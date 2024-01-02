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

# from transforms import *
from torch.distributions.normal import Normal
from torch.autograd import grad


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot


def log_normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (0.5 * log_var + torch.pow(x - mean, 2) * torch.pow(torch.exp(log_var), -1))
    return log_normal


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class TrainerOTScratchWeaklyShapes(object):
    def __init__(
        self,
        params=None,
        exp_dir=None,
        use_cuda=False,
        multi_gpu=False,
        data_loader=None,
        dataset=None,
    ):
        if params is None:
            raise ValueError("Cannot build a Trainer instance with empty params: params={}".format(params))
        else:
            self.params = params
        self.use_cuda = use_cuda
        self.multi_gpu = multi_gpu
        self.data_loader = data_loader
        self.dataset = dataset
        # Use TensorBoard
        self.tensorboard = self.params.tensorboard

        # Set output directory for current experiment (wip)
        self.wip_dir = osp.join("experiments", "wip", exp_dir)

        # Set directory for completed experiment
        self.complete_dir = osp.join("experiments", "complete", exp_dir)

        # Create log sub-directory and define stat.json file
        self.stats_json = osp.join(self.wip_dir, "stats.json")
        if not osp.isfile(self.stats_json):
            with open(self.stats_json, "w") as out:
                json.dump({}, out)

        # Create models sub-directory
        self.models_dir = osp.join(self.wip_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        # Define checkpoint model file
        self.checkpoint = osp.join(self.models_dir, "checkpoint.pt")
        # Setup TensorBoard
        if self.tensorboard:
            # Create tensorboard sub-directory
            self.tb_dir = osp.join(self.wip_dir, "tensorboard")
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, "--logdir", self.tb_dir])
            self.tb_url = self.tb.launch()
            print("#. Start TensorBoard at {}".format(self.tb_url))
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        # Define cross entropy loss function
        self.cross_entropy = nn.CrossEntropyLoss()
        # Define KL Div
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.kl_index = nn.KLDivLoss(reduction="batchmean", log_target=False)
        # Array of iteration times
        self.iter_times = np.array([])

        # Set up training statistics tracker
        self.stat_tracker = TrainingStatTracker()

    def get_starting_iteration(self, support_sets, reconstructor, generator, prior):
        """Check if checkpoint file exists (under `self.models_dir`) and set starting iteration at the checkpoint
        iteration; also load checkpoint weights to `support_sets` and `reconstructor`. Otherwise, set starting
        iteration to 1 in order to train from scratch.

        Returns:
            starting_iter (int): starting iteration

        """
        starting_iter = 1
        if osp.isfile(self.checkpoint):
            checkpoint_dict = torch.load(self.checkpoint)
            starting_iter = checkpoint_dict["iter"]
            support_sets.load_state_dict(checkpoint_dict["support_sets"])
            prior.load_state_dict(checkpoint_dict["prior"])
            reconstructor.load_state_dict(checkpoint_dict["reconstructor"])

            generator.load_state_dict(checkpoint_dict["vae"])
        return starting_iter

    def loss_fn(self, recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(self.params.batch_size, -1),
            x.view(self.params.batch_size, -1),
            reduction="sum",
        )
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + KLD) / x.size(0)

    def bce(self, recon_x, x):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(self.params.batch_size, -1),
            x.view(self.params.batch_size, -1),
            reduction="sum",
        )
        return BCE / x.size(0)

    def KL_loss(self, mean1, var1, mean2, var2):
        KLD = 0.5 * torch.sum(torch.log(var2) - torch.log(var1) - 1 + ((mean2 - mean1) ** 2) / var2 + var1 / var2)

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
        # with open(self.stats_json) as f:
        #    stats_dict = json.load(f)
        # stats_dict.update({iteration: stats})
        # with open(self.stats_json, 'w') as out:
        #    json.dump(stats_dict, out)

        # Flush training statistics tracker
        self.stat_tracker.flush()

        update_progress(
            "  \\__.Training [bs: {}] [iter: {:06d}/{:06d}] ".format(
                self.params.batch_size, iteration, self.params.max_iter
            ),
            self.params.max_iter,
            iteration + 1,
        )
        if iteration < self.params.max_iter - 1:
            print()
        print("      \\__Classification loss : {:.08f}".format(stats["classification_loss"]))
        print("      \\__Regression loss     : {:.08f}".format(stats["regression_loss"]))
        print("      \\__Total loss          : {:.08f}".format(stats["total_loss"]))
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
        torch.save(support_sets.state_dict(), osp.join(self.models_dir, "support_sets_init.pt"))

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
        support_sets_optim = torch.optim.Adam(
            [{"params": support_sets.parameters()}, {"params": prior.parameters()}],
            lr=self.params.support_set_lr,
        )

        # Set VAE optimizer
        vae_optimizer = torch.optim.Adam(generator.parameters(), lr=self.params.reconstructor_lr)

        # Set shift predictor optimizer
        reconstructor_optim = torch.optim.Adam(reconstructor.parameters(), lr=self.params.reconstructor_lr)

        # Get starting iteration
        starting_iter = self.get_starting_iteration(support_sets, reconstructor, generator, prior)

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
                shutil.copytree(
                    src=self.wip_dir,
                    dst=self.complete_dir,
                    ignore=shutil.ignore_patterns("checkpoint.pt"),
                )
                print("  \\__Done!")
            except IOError as e:
                print("  \\__Already exists -- {}".format(e))
            sys.exit()
        print("#. Start training from iteration {}".format(starting_iter))

        # Get experiment's start time
        t0 = time.time()
        # self.params.max_iter = self.params.max_iter - starting_iter + 1
        # starting_iter = 0
        # Start training
        iteration = starting_iter
        while iteration <= self.params.max_iter:
            # for iteration in range(starting_iter, self.params.max_iter + 1):
            for i, (_, factors) in enumerate(self.data_loader):
                # data = data.squeeze()
                factors = factors.squeeze()
                factors = factors.transpose(0, 1)
                index = torch.randint(0, self.params.num_support_sets, (1, 1), requires_grad=False)
                data = self.dataset.sequence_by_index(index, factors.cpu().numpy())
                data = data.squeeze()

                iteration = iteration + 1
                # Get current iteration's start time
                iter_t0 = time.time()

                # Set gradients to zero
                vae_optimizer.zero_grad()
                support_sets_optim.zero_grad()
                reconstructor_optim.zero_grad()
                if self.use_cuda:
                    data = data.cuda()
                x = data[0]

                # Generate one-hot index
                onehot_idx = torch.zeros(x.size(0), self.params.num_support_sets, requires_grad=False)
                onehot_idx[:, index] = 1.0
                half_range = self.params.num_support_dipoles // 2
                recon_x, mean, log_var, z = generator(x)
                # prior distribution
                std = torch.exp(log_var / 2.0)
                prob_z = Normal(0.0, 1.0)
                rho = prob_z.log_prob(z)
                # posterior distribution
                prob_zt = Normal(mean, std)
                rho_t = prob_zt.log_prob(z)
                # generate sequence and classify
                x_seq = x
                for t in range(1, half_range + 1):
                    x_t = data[t]
                    x_seq = torch.cat([x_seq, x_t], dim=1)
                index_pred = reconstructor(x_seq, iteration)
                vae_loss = self.loss_fn(recon_x, x, mean, log_var) + self.kl_index(
                    (index_pred + 1e-20).log(), (onehot_idx + 1e-20)
                )

                for t in range(1, half_range + 1):
                    with torch.no_grad():
                        x_t = data[t]

                    time_stamp = t * torch.ones(1, 1, requires_grad=True)
                    energy, loss_pde_tmp, uz, uzz = support_sets.index_forward(index_pred, z, time_stamp)
                    _, _, _, uzz_prior = prior.index_forward(index_pred, z, time_stamp, rho)

                    # Update rho and z
                    rho_t = rho_t - (uzz + 1).abs().log()
                    rho = rho - (uzz_prior + 1).abs().log()
                    z = z + uz

                    rho_loss1_tmp = self.kl_div(
                        torch.nn.functional.log_softmax(rho_t.exp(), dim=-1),
                        torch.nn.functional.log_softmax(rho.exp(), dim=-1),
                    )
                    img_shifted = generator.inference(z)

                    vae_loss += self.bce(img_shifted, x_t)
                    if t == 1:
                        loss_pde = loss_pde_tmp
                        rho_loss1 = rho_loss1_tmp
                    else:
                        loss_pde += loss_pde_tmp
                        rho_loss1 += rho_loss1_tmp

                loss = vae_loss + rho_loss1 + loss_pde
                # Update statistics tracker
                self.stat_tracker.update(
                    classification_loss=rho_loss1.item(),
                    regression_loss=vae_loss.item(),
                    pde_loss=loss_pde,
                    total_loss=loss.item(),
                )
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(support_sets.parameters(), 5)
                # torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
                # Perform optimization step (parameter update)
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
                        "iter": iteration,
                        "support_sets": support_sets.state_dict(),
                        "prior": prior.state_dict(),
                        "vae": generator.state_dict(),
                        "reconstructor": reconstructor.module.state_dict()
                        if self.multi_gpu
                        else reconstructor.state_dict(),
                    }
                    torch.save(checkpoint_dict, self.checkpoint)
        # === End of training loop ===

        # Get experiment's total elapsed time
        elapsed_time = time.time() - t0

        # Save final support sets model
        support_sets_model_filename = osp.join(self.models_dir, "support_sets.pt")
        torch.save(support_sets.state_dict(), support_sets_model_filename)

        # Save final shift predictor model
        reconstructor_model_filename = osp.join(self.models_dir, "reconstructor.pt")
        torch.save(
            reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict(),
            reconstructor_model_filename,
        )

        for _ in range(10):
            print()
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))

        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:
            shutil.copytree(
                src=self.wip_dir,
                dst=self.complete_dir,
                ignore=shutil.ignore_patterns("checkpoint.pt"),
            )
            print("  \\__Done!")
        except IOError as e:
            print("  \\__Already exists -- {}".format(e))

    def eval(self, generator, support_sets, reconstructor, prior):
        neg_likelihood = []
        starting_iter = self.get_starting_iteration(support_sets, reconstructor, generator, prior)
        support_sets.eval()
        reconstructor.eval()
        generator.eval()
        prior.eval()
        prior_z0 = Normal(0.0, 1.0)
        iteration = 0
        for i, (_, factors) in enumerate(self.data_loader):
            factors = factors.squeeze()
            factors = factors.transpose(0, 1)
            index = torch.randint(0, self.params.num_support_sets, (1, 1), requires_grad=False)
            data = self.dataset.sequence_by_index(index, factors.cpu().numpy())
            data = data.squeeze()
            with torch.no_grad():
                if self.use_cuda:
                    data = data.cuda()
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
        for i, (_, factors) in enumerate(self.data_loader):
            factors = factors.squeeze()
            factors = factors.transpose(0, 1)
            # for index in range(0, 5):
            index = torch.randint(0, self.params.num_support_sets, (1, 1), requires_grad=False)
            with torch.no_grad():
                data = self.dataset.sequence_by_index(index, factors.cpu().numpy())
                data = data.squeeze()
                if self.use_cuda:
                    data = data.cuda()
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
                    neg_logpx_z = (
                        -1 * p.log_prob(x_t).flatten(start_dim=1).sum(-1, keepdim=True)
                        + log_p_z.flatten(start_dim=1).sum(-1, keepdim=True)
                        - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True)
                    )
                    neg_likelihood_transformed.append(neg_logpx_z.sum() / recon_x.size(0))
        print(
            "logpx transformed",
            sum(neg_likelihood_transformed) / len(neg_likelihood_transformed),
        )
        eq_err_all_traverse0 = []
        eq_err_all_traverse1 = []
        eq_err_all_traverse2 = []
        eq_err_all_traverse3 = []
        eq_err_all_traverse4 = []
        # for index in range(0,5):
        for i, (_, factors) in enumerate(self.data_loader):
            eq_loss_traverse0 = 0.0
            eq_loss_traverse1 = 0.0
            eq_loss_traverse2 = 0.0
            eq_loss_traverse3 = 0.0
            eq_loss_traverse4 = 0.0
            factors = factors.squeeze()
            factors = factors.transpose(0, 1)
            index = torch.randint(0, self.params.num_support_sets, (1, 1), requires_grad=False)
            data = self.dataset.sequence_by_index(index, factors.cpu().numpy())
            data = data.squeeze()
            if self.use_cuda:
                data = data.cuda()
            x = data[0]
            x_seq = x
            with torch.no_grad():
                for t in range(1, self.params.num_support_dipoles // 2 + 1):
                    x_t = data[t]
                    x_seq = torch.cat([x_seq, x_t], dim=1)
            index_pred = reconstructor(x_seq, iter=100001)
            recon_x, mean, log_var, z = generator(x)
            # std = torch.exp(log_var / 2.0)
            # prob_zt = Normal(mean, std)
            # eq_loss = 0.0
            # eq_loss_traverse = 0.0
            # rho_z = prob_zt.log_prob(z)
            for t in range(1, self.params.num_support_dipoles // 2 + 1):
                with torch.no_grad():
                    x_t = data[t]
                    recon_xt, _, _, _ = generator(x_t)
                _, shift, u_zz = support_sets.index_inference(index_pred, z, t * torch.ones(1, 1, requires_grad=True))
                z += shift
                # rho_z = rho_z - (u_zz + 1).abs().log()
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
        print("eq err traverse 0", sum(eq_err_all_traverse0) / len(eq_err_all_traverse0))
        print("eq err traverse 1", sum(eq_err_all_traverse1) / len(eq_err_all_traverse1))
        print("eq err traverse 2", sum(eq_err_all_traverse2) / len(eq_err_all_traverse2))
        print("eq err traverse 3", sum(eq_err_all_traverse3) / len(eq_err_all_traverse3))
        print("eq err traverse 4", sum(eq_err_all_traverse4) / len(eq_err_all_traverse4))

        return None
