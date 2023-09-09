import argparse
import torch
from lib import *
vae import VAE, Encoder, ConvVAE, ConvEncoder, ConvEncoder2, ConvVAE2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import numpy as np
import os
from transforms import *
from Falor3D import Falor3D
from Isaac3D import Isaac3D

def main():
   

    parser = argparse.ArgumentParser(description="PDE training script")

    # === PDEs (S) ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, help="set number of PDEs")
    parser.add_argument('-D', '--num-timesteps', type=int, help="set number of timesteps")
    parser.add_argument('--support-set-lr', type=float, default=1e-4, help="set learning rate")
    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=128, help="set batch size")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
    parser.add_argument("--isaac", type=bool, default=False)
    # === CUDA ======================================================================================================= #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args)

    # CUDA
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')


    if args.isaac == True:
        G = ConvVAE2(num_channel=3, latent_size=18 * 18, img_size=128)
        print("Intialize DSPRITES VAE")
    else:
        G = ConvVAE2(num_channel=3, latent_size=18 * 18, img_size=128)
        print("Intialize MNIST VAE")

    # Build Support Sets model S
    print("#. Build Support Sets S...")
    print("  \\__Number of Support Sets    : {}".format(args.num_support_sets))
    print("  \\__Number of Support Dipoles : {}".format(args.num_support_dipoles))
    print("  \\__Support Vectors dim       : {}".format(G.latent_size))
    print("  \\__Learn RBF alphas          : {}".format(args.learn_alphas))
    print("  \\__Learn RBF gammas          : {}".format(args.learn_gammas))
    if not args.learn_gammas:
        print("  \\__RBF gamma                 : {}".format(1.0 / G.latent_size if args.gamma is None else args.gamma))

    if args.isaac == True:
        S = HJPDE(num_support_sets=args.num_support_sets,
                        num_support_dipoles=args.num_support_dipoles,
                        support_vectors_dim=G.latent_size,
                        learn_alphas=args.learn_alphas,
                        learn_gammas=args.learn_gammas,
                        gamma=1.0 / G.latent_size if args.gamma is None else args.gamma,
                        img_size=64
                  )
    else:
        S = HJPDE(num_support_sets=args.num_support_sets,
                  num_support_dipoles=args.num_support_dipoles,
                  support_vectors_dim=G.latent_size,
                  learn_alphas=args.learn_alphas,
                  learn_gammas=args.learn_gammas,
                  gamma=1.0 / G.latent_size if args.gamma is None else args.gamma
                  )

    S_Prior = WavePDE(num_support_sets=args.num_support_sets,
              num_support_dipoles=args.num_support_dipoles,
              support_vectors_dim=G.latent_size,
              learn_alphas=args.learn_alphas,
              learn_gammas=args.learn_gammas,
              gamma=1.0 / G.latent_size if args.gamma is None else args.gamma)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Build reconstructor model R
    print("#. Build reconstructor model R...")

    if args.isaac == True:
        R = ConvEncoder2(n_cin=6*3, s_dim=18 * 18, n_hw=128)
    else:
        R = ConvEncoder2(n_cin=6*3, s_dim=18 * 18, n_hw=128)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    if args.isaac:
        print("ISSAC DATASET LOADING")
        train_tx = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        dataset = Isaac3D(root='/nfs/data_chaos/ysong/Isaac3D_down128/images', train=True, transform=train_tx)
        data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator(device='cuda'))
        trn = TrainerFlowScratch(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
                               data_loader=data_loader)

    else:
        train_tx = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        print("FALOR DATASET LOADING")
        dataset = Falor3D(root='/nfs/data_chaos/ysong/Falcor3D_down128/images', train=True, transform=train_tx)
        data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator(device='cuda'))
        trn = TrainerFlowScratch(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
                                data_loader=data_loader)

    # Train
    trn.train(generator=G, support_sets=S, reconstructor=R, prior = S_Prior)
    trn.eval(generator=G, support_sets=S, reconstructor=R, prior = S_Prior)


if __name__ == '__main__':
    main()
