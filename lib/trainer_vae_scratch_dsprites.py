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


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class TrainerVAEScratchDsprites(object):
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
        #self.kl_div = nn.functional.kl_div()
        # Array of iteration times
        self.iter_times = np.array([])

        # Set up training statistics tracker
        self.stat_tracker = TrainingStatTracker()

    def get_starting_iteration(self, support_sets, reconstructor, generator):
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
            reconstructor.load_state_dict(checkpoint_dict['reconstructor'])
            generator.load_state_dict(checkpoint_dict['vae'])
        return starting_iter

    def loss_fn(self,recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(self.params.batch_size, -1), x.view(self.params.batch_size, -1), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + KLD) / x.size(0)

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
        with open(self.stats_json) as f:
            stats_dict = json.load(f)
        stats_dict.update({iteration: stats})
        with open(self.stats_json, 'w') as out:
            json.dump(stats_dict, out)

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

    def train(self, generator, support_sets, reconstructor):
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
        else:
            generator.train()
            support_sets.train()
            reconstructor.train()

        # Set support sets optimizer
        support_sets_optim = torch.optim.Adam(support_sets.parameters(), lr=self.params.support_set_lr)

        # Set VAE optimizer
        vae_optimizer = torch.optim.Adam(generator.parameters(), lr=self.params.reconstructor_lr)

        # Set shift predictor optimizer
        reconstructor_optim = torch.optim.Adam(reconstructor.parameters(), lr=self.params.reconstructor_lr)

        # Get starting iteration
        starting_iter = self.get_starting_iteration(support_sets, reconstructor,generator)

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
            for i, (data, y, index, step) in enumerate(self.data_loader):
                data = data.squeeze(0)
                y = y.squeeze(0)
                #print(y.size())
                iteration = iteration + 1
                # Get current iteration's start time
                iter_t0 = time.time()
            # Set gradients to zero
                vae_optimizer.zero_grad()
                support_sets.zero_grad()
                reconstructor.zero_grad()
                # Sample latent codes from standard (truncated) Gaussian -- torch.Size([batch_size, generator.dim_z])
                #z = sample_z(batch_size=self.params.batch_size, dim_z=generator.latent_size, truncation=self.params.z_truncation)
                if self.use_cuda:
                    data = data.cuda()
                print(y[0,0],y[0,1],y[0,2])
                x = data[:,:,0]
                x_t = data[:,:,1]
                x_t1 = data[:,:,2]


                # Generate images the shifted latent codes
                #index = torch.randint(0,self.params.num_support_sets,(1,1),requires_grad=False)
                #time_stamp = 0.5*torch.rand([self.params.batch_size,1],requires_grad=True).to(z)
                #time_stamp = torch.randint(0, self.params.num_support_dipoles, (1, 1), dtype=z.dtype,requires_grad=True).to(z)
                #half_range = self.params.num_support_dipoles // 2
                time_stamp = step * torch.ones(self.params.batch_size, 1, dtype=x.dtype, requires_grad=True).to(x)

                recon_x, mean, log_var, z = generator(x)
                if iteration > 20000:
                    recon_xt, mean_t, log_var_t, z_t = generator(x_t)
                    recon_xt1, mean_t1, log_var_t1, z_t1 = generator(x_t1)

                    #vae_loss = self.loss_fn(recon_x, x, mean, log_var) #+ self.loss_fn(recon_xt1, x_t1, mean_t1, log_var_t1)

                    energy, latent1, latent2, loss_wave = support_sets(index, z, time_stamp, generator.decoder)

                    img_shifted = generator.inference(latent1)
                    img_shifted2 = generator.inference(latent2)

                    if iteration > 40000:
                        img_detached = img_shifted.detach().clone().requires_grad_()
                        recon_xt1_hat, mean_t1_hat, log_var_t1_hat, z_t1_hat = generator(img_detached)
                        vae_loss = self.loss_fn(recon_x, x, mean, log_var) + self.loss_fn(recon_xt1_hat, img_detached,
                                                                                          mean_t1_hat, log_var_t1_hat)
                    else:
                        vae_loss = self.loss_fn(recon_x, x, mean, log_var) + self.loss_fn(recon_xt1, x_t1, mean_t1,
                                                                                          log_var_t1)
                    # Predict support sets indices and shift magnitudes
                    mean, var = reconstructor(torch.cat([img_shifted,img_shifted2],dim=1))
                    #predicted_support_sets_indices = reconstructor.reparameterize(mean,var)
                    classification_loss = self.cross_entropy(mean,
                                                             index.repeat(self.params.batch_size, 1).squeeze())

                    latent_loss = torch.mean(torch.square(z_t1-z_t+latent1-latent2)) + torch.mean(torch.square(z_t-latent1))
                    loss =  vae_loss + self.params.lambda_cls * classification_loss + self.params.lambda_pde * (loss_wave) + latent_loss
                    self.stat_tracker.update(
                        accuracy_index=torch.mean((torch.argmax(mean, dim=1) ==
                                                   index).to(torch.float32)).detach(),
                        accuracy_time=0.0,
                        classification_loss=classification_loss.item(),
                        regression_loss=vae_loss.item(),
                        wave_loss=loss_wave.item(),
                        total_loss=loss.item())
                    loss.backward()

                    # Perform optimization step (parameter update)
                    support_sets_optim.step()
                    reconstructor_optim.step()
                    vae_optimizer.step()
                else:
                    loss = self.loss_fn(recon_x, x, mean, log_var)
                    self.stat_tracker.update(
                        accuracy_index=0.0,
                        accuracy_time=0.0,
                        classification_loss=0.0,
                        regression_loss=0.0,
                        wave_loss=0.0,
                        total_loss=loss.item())
                    loss.backward()
                    vae_optimizer.step()


                # Update statistics tracker

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

    def eval(self, generator, support_sets, reconstructor):

        neg_likelihood = []
        starting_iter = self.get_starting_iteration(support_sets, reconstructor, generator)
        support_sets.eval()
        reconstructor.eval()
        generator.eval()
        prior = Normal(0.0, 1.0)
        for i, (data, y, index, step) in enumerate(self.data_loader):
            with torch.no_grad():
                data = data.squeeze(0)
                if self.use_cuda:
                    data = data.cuda()
                x = data[:, :, 0]
                recon_x, mean, log_var, z = generator(x)
                std = torch.exp(log_var / 2.0)
                q = Normal(mean, std)
                z_rs = q.rsample()
                log_q_z = q.log_prob(z_rs).flatten(start_dim=1).sum(-1, keepdim=True)
                log_p_z = prior.log_prob(z_rs).flatten(start_dim=1).sum(-1, keepdim=True)
                p = Normal(loc=recon_x, scale=1.0)
                neg_logpx_z = -1 * p.log_prob(x).flatten(start_dim=1).sum(-1, keepdim=True) + log_p_z - log_q_z
                neg_likelihood.append(neg_logpx_z.sum()/recon_x.size(0))
        print("logpx",sum(neg_likelihood)/len(neg_likelihood))
        return None