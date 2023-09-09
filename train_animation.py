import argparse
import torch
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2, build_sngan
from models.vae import VAE, Encoder, ConvVAE, ConvEncoder, ConvEncoder2, ConvVAE2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import numpy as np
import os
from transforms import *
from Falor3D import Falor3D
from Isaac3D import Isaac3D
from frames_dataset import DatasetRepeater,FramesDataset

def main():
    """WarpedGANSpace -- Training script.

    Options:
        ===[ Pre-trained GAN Generator (G) ]============================================================================
        --gan-type                 : set pre-trained GAN type
        --z-truncation             : set latent code sampling truncation parameter. If set, latent codes will be sampled
                                     from a standard Gaussian distribution truncated to the range [-args.z_truncation,
                                     +args.z_truncation]
        --biggan-target-classes    : set list of classes to use for conditional BigGAN (see BIGGAN_CLASSES in
                                     lib/config.py). E.g., --biggan-target-classes 14 239.
        --stylegan2-resolution     : set StyleGAN2 generator output images resolution:  256 or 1024 (default: 1024)
        --shift-in-w-space         : search latent paths in StyleGAN2's W-space (otherwise, look in Z-space)

        ===[ Support Sets (S) ]=========================================================================================
        -K, --num-support-sets     : set number of support sets; i.e., number of warping functions -- number of
                                     interpretable paths
        -D, --num-support-dipoles  : set number of support dipoles per support set
        --learn-alphas             : learn RBF alpha params
        --learn-gammas             : learn RBF gamma params
        -g, --gamma                : set RBF gamma param (by default, gamma will be set to the inverse of the latent
                                     space dimensionality)
        --support-set-lr           : set learning rate for learning support sets

        ===[ Reconstructor (R) ]========================================================================================
        --reconstructor-type       : set reconstructor network type
        --min-shift-magnitude      : set minimum shift magnitude
        --max-shift-magnitude      : set maximum shift magnitude
        --reconstructor-lr         : set learning rate for reconstructor R optimization

        ===[ Training ]=================================================================================================
        --max-iter                 : set maximum number of training iterations
        --batch-size               : set training batch size
        --lambda-cls               : classification loss weight
        --lambda-reg               : regression loss weight
        --log-freq                 : set number iterations per log
        --ckp-freq                 : set number iterations per checkpoint model saving
        --tensorboard              : use TensorBoard

        ===[ CUDA ]=====================================================================================================
        --cuda                     : use CUDA during training (default)
        --no-cuda                  : do NOT use CUDA during training
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="WarpedGANSpace training script")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan-type', type=str, help='set GAN generator model type')
    parser.add_argument('--z-truncation', type=float, help="set latent code sampling truncation parameter")
    parser.add_argument('--biggan-target-classes', nargs='+', type=int, help="list of classes for conditional BigGAN")
    parser.add_argument('--stylegan2-resolution', type=int, default=1024, choices=(256, 1024),
                        help="StyleGAN2 image resolution")
    parser.add_argument('--shift-in-w-space', action='store_true', help="search latent paths in StyleGAN2's W-space")

    # === Support Sets (S) ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, help="set number of support sets (warping functions)")
    parser.add_argument('-D', '--num-support-dipoles', type=int, help="set number of support dipoles per support set")
    parser.add_argument('--learn-alphas', action='store_true', help='learn RBF alpha params')
    parser.add_argument('--learn-gammas', action='store_true', help='learn RBF gamma params')
    parser.add_argument('-g', '--gamma', type=float, help="set RBF gamma param; when --learn-gammas is set, this will "
                                                          "be the initial value of gammas for all RBFs")
    parser.add_argument('--support-set-lr', type=float, default=1e-4, help="set learning rate")

    # === Reconstructor (R) ========================================================================================== #
    parser.add_argument('--reconstructor-type', type=str, default='ResNet',
                        help='set reconstructor network type')
    parser.add_argument('--min-shift-magnitude', type=float, default=0.25, help="set minimum shift magnitude")
    parser.add_argument('--max-shift-magnitude', type=float, default=0.45, help="set shifts magnitude scale")
    parser.add_argument('--reconstructor-lr', type=float, default=1e-4,
                        help="set learning rate for reconstructor R optimization")

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=128, help="set batch size")
    parser.add_argument('--lambda-cls', type=float, default=1.00, help="classification loss weight")
    parser.add_argument('--lambda-reg', type=float, default=1.00, help="regression loss weight")
    parser.add_argument('--lambda-pde', type=float, default=1.00, help="regression loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
    parser.add_argument("--isaac", type=bool, default=False)
    parser.add_argument("--shapes3d", type=bool, default=False)
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


    # === BigGAN ===
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
