import sys
import os
import os.path as osp
import json
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import numpy as np
import time
import shutil
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from .aux import sample_z, TrainingStatTracker, update_progress, update_stdout, sec2dhms
from transforms import *
from torch.distributions.normal import Normal
from .WavePDE import WavePDE
from torch.autograd import grad

torch.autograd.set_detect_anomaly(True)

angle_set = [0, 10, 20, 30, 40, 50, 60 , 70 ,80]
color_set = [180, 200, 220, 240, 260, 280 ,300 , 320, 340]
scale_set = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5 , 1.6 , 1.7, 1.8]
mnist_trans = AddRandomTransformationDims(angle_set=angle_set,color_set=color_set,scale_set=scale_set)
mnist_color = To_Color()

def log_normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( 0.5*log_var + torch.pow( x - mean, 2 ) * torch.pow( torch.exp( log_var), -1) )
    return log_normal

class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class TrainerFlowScratch(object):
    def __init__(self, params=None, exp_dir=None, use_cuda=False, multi_gpu=False,data_loader=None):
        if params is None:
            raise ValueError("Cannot build a Trainer instance with empty params: params={}".format(params))
        else:
            self.params = params
        self.use_cuda = use_cuda
        self.multi_gpu = multi_gpu
        self.data_loader=data_loader

        # Use TensorBoard
        self.tensorboard = self.params.tensorboard

        # Set output directory for current experiment (wip)
        self.wip_dir = osp.join("experiments", "wip", exp_dir)

        # Set directory for completed experiment
        self.complete_dir = osp.join("experiments", "complete", exp_dir)

        # Create log sub-directory and define stat.json file
        self.stats_json = osp.join(self.wip_dir, 'stats.json')
        if not osp.isfile(self.stats_json):
            with open(self.stats_json, 'w') as out:
                json.dump({}, out)

        # Create models sub-directory
        self.models_dir = osp.join(self.wip_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        # Define checkpoint model file
        self.checkpoint = osp.join(self.models_dir, 'checkpoint.pt')
        # Setup TensorBoard
        if self.tensorboard:
            # Create tensorboard sub-directory
            self.tb_dir = osp.join(self.wip_dir, 'tensorboard')
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', self.tb_dir])
            self.tb_url = self.tb.launch()
            print("#. Start TensorBoard at {}".format(self.tb_url))
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        # Define cross entropy loss function
        self.cross_entropy = nn.CrossEntropyLoss()
        # Define KL Div
        self.kl_div = nn.KLDivLoss(reduction="batchmean",log_target=True)
        # Array of iteration times
        self.iter_times = np.array([])

        # Set up training statistics tracker
        self.stat_tracker = TrainingStatTracker()

    def get_starting_iteration(self, support_sets, reconstructor, generator,prior):
        """Check if checkpoint file exists (under `self.models_dir`) and set starting iteration at the checkpoint
        iteration; also load checkpoint weights to `support_sets` and `reconstructor`. Otherwise, set starting
        iteration to 1 in order to train from scratch.

        Returns:
            starting_iter (int): starting iteration

        """
        starting_iter = 1
        if osp.isfile(self.checkpoint):
            checkpoint_dict = torch.load(self.checkpoint)
            starting_iter = checkpoint_dict['iter']
            support_sets.load_state_dict(checkpoint_dict['support_sets'])
            prior.load_state_dict(checkpoint_dict['prior'])
            reconstructor.load_state_dict(checkpoint_dict['reconstructor'])
            #state_dict_new = {}
            #for k, v in checkpoint_dict['vae'].items():
            #    state_dict_new[k[len("module.encoder."):]] = v
            generator.load_state_dict(checkpoint_dict['vae'])
        return starting_iter

    def loss_fn(self,recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(self.params.batch_size, -1), x.view(self.params.batch_size, -1), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + KLD) / x.size(0)

    def bce(self,recon_x, x):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(self.params.batch_size, -1),
                                                       x.view(self.params.batch_size, -1), reduction='sum')
        return BCE / x.size(0)

    def KL_loss(self, mean1, var1, mean2, var2):

        KLD = 0.5 * torch.sum(torch.log(var2) - torch.log(var1) -1 + ((mean2-mean1)**2)/var2 + var1/var2)

        return KLD

    def log_progress(self, iteration, mean_iter_time, elapsed_time, eta):
        """Log progress in terms of batch accuracy, classification and regression losses and ETA.

        Args:
            iteration (int)        : current iteration
            mean_iter_time (float) : mean iteration time
            elapsed_time (float)   : elapsed time until current iteration
            eta (float)            : estimated time of experiment completion

        """
        # Get current training stats (for the previous `self.params.log_freq` steps) and flush them
        stats = self.stat_tracker.get_means()

        # Update training statistics json file
        #with open(self.stats_json) as f:
        #    stats_dict = json.load(f)
        #stats_dict.update({iteration: stats})
        #with open(self.stats_json, 'w') as out:
        #    json.dump(stats_dict, out)

        # Flush training statistics tracker
        self.stat_tracker.flush()

        update_progress("  \\__.Training [bs: {}] [iter: {:06d}/{:06d}] ".format(
            self.params.batch_size, iteration, self.params.max_iter), self.params.max_iter, iteration + 1)
        if iteration < self.params.max_iter - 1:
            print()
        print("      \\__Batch accuracy Index      : {:.03f}".format(stats['accuracy_index']))
        print("      \\__Batch accuracy Time      : {:.03f}".format(stats['accuracy_time']))
        print("      \\__Classification loss : {:.08f}".format(stats['classification_loss']))
        print("      \\__Regression loss     : {:.08f}".format(stats['regression_loss']))
        print("      \\__Total loss          : {:.08f}".format(stats['total_loss']))
        print("         ===================================================================")
        print("      \\__Mean iter time      : {:.3f} sec".format(mean_iter_time))
        print("      \\__Elapsed time        : {}".format(sec2dhms(elapsed_time)))
        print("      \\__ETA                 : {}".format(sec2dhms(eta)))
        print("         ===================================================================")
        update_stdout(10)

    def train(self, generator, support_sets, reconstructor, prior):
        """Training function.

        Args:
            generator     :
            support_sets  :
            reconstructor :

        """
        # Save initial `support_sets` model as `support_sets_init.pt`
        torch.save(support_sets.state_dict(), osp.join(self.models_dir, 'support_sets_init.pt'))

        # Set `generator` to evaluation mode, `support_sets` and `reconstructor` to training mode, and upload
        # models to GPU if `self.use_cuda` is set (i.e., if args.cuda and torch.cuda.is_available is True).
        if self.use_cuda:
            generator.cuda().train()
            support_sets.cuda().train()
            reconstructor.cuda().train()
            prior.cuda().train()
        else:
            generator.train()
            support_sets.train()
            reconstructor.train()
            prior.train()

        # Set support sets optimizer
        support_sets_optim = torch.optim.Adam([{'params': support_sets.parameters()},{'params': prior.parameters()}], lr=self.params.support_set_lr)

        # Set VAE optimizer
        vae_optimizer = torch.optim.Adam(generator.parameters(), lr=self.params.reconstructor_lr)

        # Set shift predictor optimizer
        reconstructor_optim = torch.optim.Adam(reconstructor.parameters(), lr=self.params.reconstructor_lr)

        # Get starting iteration
        starting_iter = self.get_starting_iteration(support_sets, reconstructor,generator,prior)

        # Parallelize `generator` and `reconstructor` into multiple GPUs, if available and `multi_gpu=True`.
        if self.multi_gpu:
            print("#. Parallelize G, R over {} GPUs...".format(torch.cuda.device_count()))
            generator = DataParallelPassthrough(generator)
            reconstructor = DataParallelPassthrough(reconstructor)
            cudnn.benchmark = True

        # Check starting iteration
        if starting_iter == self.params.max_iter:
            print("#. This experiment has already been completed and can be found @ {}".format(self.wip_dir))
            print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
            try:
                shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
                print("  \\__Done!")
            except IOError as e:
                print("  \\__Already exists -- {}".format(e))
            sys.exit()
        print("#. Start training from iteration {}".format(starting_iter))

        # Get experiment's start time
        t0 = time.time()
        #self.params.max_iter = self.params.max_iter - starting_iter + 1
        #starting_iter = 0
        # Start training
        iteration = starting_iter
        while iteration<=self.params.max_iter:
        #for iteration in range(starting_iter, self.params.max_iter + 1):
            for i, (data, index) in enumerate(self.data_loader):
                index = index[0]
                next_index = torch.randint(0, self.params.num_support_sets, (1, 1), requires_grad=False)
                self.data_loader.dataset.index = next_index
                iteration = iteration + 1
                # Set gradients to zero
                vae_optimizer.zero_grad()
                support_sets.zero_grad()
                reconstructor.zero_grad()
                if self.use_cuda:
                    data = [t.cuda() for t in data]
                x = data[0]
                #print(x.mean())
                iter_t0 = time.time()
                half_range = self.params.num_support_dipoles // 2
                #If you want to have x_0 start with different angle/color/scale
                #x = mnist_trans(x, torch.randint(0, self.params.num_support_sets,(1,1)), torch.randint(0, half_range//2,(1,1)) )
                recon_x, mean, log_var, z = generator(x)
                #prior probability
                std = torch.exp(log_var / 2.0)
                prob_z = Normal(0.0, 1.0)
                rho = prob_z.log_prob(z)
                #posterior probability
                prob_zt = Normal(mean, std)
                rho_t = prob_zt.log_prob(z)
                vae_loss = self.loss_fn(recon_x, x, mean, log_var)
                for t in range(1, half_range):
                    x_t = data[t]
                    time_stamp = t * torch.ones(1, 1, requires_grad=True)
                    energy, loss_wave_tmp, uz, uzz = support_sets(index, z, time_stamp)

                    _, _, _, uzz_prior = prior(index, z, time_stamp, rho)

                    rho_t = rho_t - (uzz+1).abs().log()
                    rho = rho - (uzz_prior+1).abs().log()
                    z = z + uz

                    rho_loss1_tmp = self.kl_div(torch.nn.functional.log_softmax(rho_t.exp(), dim=-1),
                                                torch.nn.functional.log_softmax(rho.exp(), dim=-1))
                    img_shifted = generator.inference(z)

                    vae_loss +=  self.bce(img_shifted,x_t)

                    if t==1:
                        loss_wave = loss_wave_tmp
                        rho_loss1 = rho_loss1_tmp
                    else:
                        loss_wave += loss_wave_tmp
                        rho_loss1 += rho_loss1_tmp

                loss = vae_loss + rho_loss1 + loss_wave
                # Update statistics tracker
                self.stat_tracker.update(
                    accuracy_index=0.0, #torch.mean((torch.argmax(mean_k, dim=1) ==            index).to(torch.float32)).detach(),
                    accuracy_time=0.0,
                    classification_loss=rho_loss1.item(),
                    regression_loss=vae_loss.item(), #+ latent_loss.item(),
                    wave_loss=loss_wave.item(),
                    total_loss=loss.item())
                loss.backward()

                support_sets_optim.step()
                reconstructor_optim.step()
                vae_optimizer.step()



                # Update tensorboard plots for training statistics
                if self.tensorboard:
                    for key, value in self.stat_tracker.get_means().items():
                        self.tb_writer.add_scalar(key, value, iteration)

                # Get time of completion of current iteration
                iter_t = time.time()

                # Compute elapsed time for current iteration and append to `iter_times`
                self.iter_times = np.append(self.iter_times, iter_t - iter_t0)

                # Compute elapsed time so far
                elapsed_time = iter_t - t0

                # Compute rolling mean iteration time
                mean_iter_time = self.iter_times.mean()

                # Compute estimated time of experiment completion
                eta = elapsed_time * ((self.params.max_iter - iteration) / (iteration - starting_iter + 1))

                # Log progress in stdout
                if iteration % self.params.log_freq == 0:
                    self.log_progress(iteration, mean_iter_time, elapsed_time, eta)

                # Save checkpoint model file and support_sets / reconstructor model state dicts after current iteration
                if iteration % self.params.ckp_freq == 0:
                    # Build checkpoint dict
                    checkpoint_dict = {
                        'iter': iteration,
                        'support_sets': support_sets.state_dict(),
                        'prior': prior.state_dict(),
                        'vae': generator.state_dict(),
                        'reconstructor': reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict()
                    }
                    torch.save(checkpoint_dict, self.checkpoint)
        # === End of training loop ===

        # Get experiment's total elapsed time
        elapsed_time = time.time() - t0

        # Save final support sets model
        support_sets_model_filename = osp.join(self.models_dir, 'support_sets.pt')
        torch.save(support_sets.state_dict(), support_sets_model_filename)

        # Save final shift predictor model
        reconstructor_model_filename = osp.join(self.models_dir, 'reconstructor.pt')
        torch.save(reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict(),
                   reconstructor_model_filename)

        for _ in range(10):
            print()
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))

        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
            print("  \\__Done!")
        except IOError as e:
            print("  \\__Already exists -- {}".format(e))

    def eval(self, generator, support_sets, reconstructor, prior):

        neg_likelihood = []
        # starting_iter = self.get_starting_iteration(support_sets, reconstructor, generator,prior)
        support_sets.eval()
        reconstructor.eval()
        generator.eval()
        prior.eval()
        prior_z0 = Normal(0.0, 1.0)
        for i, (data, index) in enumerate(self.data_loader):
            index = index[0]
            next_index = torch.randint(0, self.params.num_support_sets, (1, 1), requires_grad=False)
            self.data_loader.dataset.index = next_index
            with torch.no_grad():
                if self.use_cuda:
                    data = [t.cuda() for t in data]
                x = data[0]
                recon_x, mean, log_var, z = generator(x)
                std = torch.exp(log_var / 2.0)
                q = Normal(mean, std)
                z_rs = q.rsample()
                log_q_z = q.log_prob(z_rs).flatten(start_dim=1).sum(-1, keepdim=True)
                log_p_z = prior_z0.log_prob(z_rs).flatten(start_dim=1).sum(-1, keepdim=True)
                p = Normal(loc=recon_x, scale=1.0)
                neg_logpx_z = -1 * p.log_prob(x).flatten(start_dim=1).sum(-1, keepdim=True) + log_p_z - log_q_z
                neg_likelihood.append(neg_logpx_z.sum() / recon_x.size(0))
        print("logpx", sum(neg_likelihood) / len(neg_likelihood))
        neg_likelihood_transformed = []
        for i, (data, index) in enumerate(self.data_loader):
            index = index[0]
            next_index = torch.randint(0, self.params.num_support_sets, (1, 1), requires_grad=False)
            self.data_loader.dataset.index = next_index
            with torch.no_grad():
                if self.use_cuda:
                    data = [t.cuda() for t in data]
                x = data[0]
                recon_x, mean, log_var, z = generator(x)
                std = torch.exp(log_var / 2.0)
                q = Normal(mean, std)
                log_q_z = q.log_prob(z)
                log_p_z = prior_z0.log_prob(z)
            for t in range(1, self.params.num_support_dipoles // 2):
                x_t = data[t]
                time_stamp = t * torch.ones(1, 1, requires_grad=True)
                _, uz, uzz = support_sets.inference(index, z, time_stamp)
                _, _, uzz_prior = prior.inference(index, z, time_stamp)
                with torch.no_grad():
                    log_q_z -= (uzz + 1).abs().log()
                    log_p_z -= (uzz_prior + 1).abs().log()
                    z += uz
                    img_shifted = generator.inference(z)
                    p = Normal(loc=img_shifted, scale=1.0)
                    neg_logpx_z = -1 * p.log_prob(x_t).flatten(start_dim=1).sum(-1, keepdim=True) + log_p_z.flatten(
                        start_dim=1).sum(-1, keepdim=True) - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True)
                    neg_likelihood_transformed.append(neg_logpx_z.sum() / recon_x.size(0))
        print("logpx transformed", sum(neg_likelihood_transformed) / len(neg_likelihood_transformed))
        eq_err_all_traverse0 = []
        eq_err_all_traverse1 = []
        eq_err_all_traverse2 = []
        eq_err_all_traverse3 = []
        eq_err_all_traverse4 = []
        eq_err_all_traverse5 = []
        eq_err_all_traverse6 = []
        # for index in range(0, 5):
        for i, (data, index) in enumerate(self.data_loader):
            index = index[0]
            next_index = torch.randint(0, self.params.num_support_sets, (1, 1), requires_grad=False)
            self.data_loader.dataset.index = next_index
            with torch.no_grad():
                if self.use_cuda:
                    data = [t.cuda() for t in data]
                x = data[0]
                _, _, _, z = generator(x)
            eq_loss_traverse0 = 0.0
            eq_loss_traverse1 = 0.0
            eq_loss_traverse2 = 0.0
            eq_loss_traverse3 = 0.0
            eq_loss_traverse4 = 0.0
            eq_loss_traverse5 = 0.0
            eq_loss_traverse6 = 0.0
            for t in range(1, self.params.num_support_dipoles // 2):
                with torch.no_grad():
                    x_t = data[t]
                    recon_xt, _, _, _ = generator(x_t)
                with torch.no_grad():
                    img_shifted = generator.inference(z)
                if index == 0:
                    eq_loss_traverse0 += (img_shifted.detach() - x_t).abs().sum() / x_t.size(0)
                elif index == 1:
                    eq_loss_traverse1 += (img_shifted.detach() - x_t).abs().sum() / x_t.size(0)
                elif index == 2:
                    eq_loss_traverse2 += (img_shifted.detach() - x_t).abs().sum() / x_t.size(0)
                elif index == 3:
                    eq_loss_traverse3 += (img_shifted.detach() - x_t).abs().sum() / x_t.size(0)
                elif index == 4:
                    eq_loss_traverse4 += (img_shifted.detach() - x_t).abs().sum() / x_t.size(0)
                elif index == 5:
                    eq_loss_traverse5 += (img_shifted.detach() - x_t).abs().sum() / x_t.size(0)
                elif index == 6:
                    eq_loss_traverse6 += (img_shifted.detach() - x_t).abs().sum() / x_t.size(0)
            if index == 0:
                eq_err_all_traverse0.append(eq_loss_traverse0)
            elif index == 1:
                eq_err_all_traverse1.append(eq_loss_traverse1)
            elif index == 2:
                eq_err_all_traverse2.append(eq_loss_traverse2)
            elif index == 3:
                eq_err_all_traverse3.append(eq_loss_traverse3)
            elif index == 4:
                eq_err_all_traverse4.append(eq_loss_traverse4)
            elif index == 5:
                eq_err_all_traverse5.append(eq_loss_traverse5)
            elif index == 6:
                eq_err_all_traverse6.append(eq_loss_traverse6)
        print("eq err traverse 0", sum(eq_err_all_traverse0) / len(eq_err_all_traverse0))
        print("eq err traverse 1", sum(eq_err_all_traverse1) / len(eq_err_all_traverse1))
        print("eq err traverse 2", sum(eq_err_all_traverse2) / len(eq_err_all_traverse2))
        print("eq err traverse 3", sum(eq_err_all_traverse3) / len(eq_err_all_traverse3))
        print("eq err traverse 4", sum(eq_err_all_traverse4) / len(eq_err_all_traverse4))
        print("eq err traverse 5", sum(eq_err_all_traverse5) / len(eq_err_all_traverse5))
        print("eq err traverse 6", sum(eq_err_all_traverse6) / len(eq_err_all_traverse6))
        return None