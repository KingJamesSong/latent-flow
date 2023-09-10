import argparse
import torch
from lib import *
from vae import ConvVAE,ConvEncoder2
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import numpy as np
import os
from transforms import *
from Shapes3d import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main():
     """
        ===[ PDEs ]=========================================================================================
        -K, --num-support-sets     : set number of PDEs
        -D, --num-timesteps        : set number of timesteps
        --support-set-lr           : set learning rate for learning PDEs
        
        ===[ Training ]=================================================================================================
        --max-iter                 : set maximum number of training iterations
        --batch-size               : set training batch size
        --log-freq                 : set number iterations per log
        --ckp-freq                 : set number iterations per checkpoint model saving
        --tensorboard              : use TensorBoard
        
        ===[ CUDA ]=====================================================================================================
        --cuda                     : use CUDA during training (default)
        --no-cuda                  : do NOT use CUDA during training
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="Latent Flow training script")

    # === PDEs ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, help="set number of PDEs")
    parser.add_argument('-D', '--num-timesteps', type=int, help="set number of timesteps")
    parser.add_argument('--support-set-lr', type=float, default=1e-4, help="set learning rate for learning PDEs")
    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=128, help="set batch size")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
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



    if args.shapes3d == True:
        G = ConvVAE(num_channel=3,latent_size=15*15+1,img_size=64)
        G.load_state_dict(torch.load("vae_shapes3d.pt", map_location='cpu'))
        print("Initialize Shapes3D VAE")
    elif:
        G = ConvVAE(num_channel=3, latent_size=18*18, img_size=28)
        print("Intialize MNIST VAE")

    # Build PDEs
    print("#. Build Support Sets S...")
    print("  \\__Number of Support Sets    : {}".format(args.num_support_sets))
    print("  \\__Number of Timesteps       : {}".format(args.num_timesteps))
    print("  \\__Latent    Dimension       : {}".format(G.latent_size))

    S = HJPDE(num_support_sets=args.num_support_sets,
              num_timesteps=args.num_timesteps,
              support_vectors_dim=G.latent_size)

    S_Prior = DiffPDE(num_support_sets=args.num_support_sets,
              num_timesteps=args.num_timesteps,
              support_vectors_dim=G.latent_size)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))


    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    if args.shapes3d:
        print("SHAPES3D DATASET LOADING")
        train_tx = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = Shapes3D(root='/nfs/data_lambda/ysong/3dshapes.h5', train=True, transform=None)
        #train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)],generator=torch.Generator(device='cuda'))
        data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator(device='cuda'))
        trn = TrainerOTScratchShapes(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
                               data_loader=data_loader,dataset=dataset)
    else:
        print("MNIST DATASET LOADING")
        #train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=data_config['batch_size'])
        dataset = MNIST(root='/nfs/data_lambda/ysong/', train=True, transform=transforms.ToTensor(),download=True)
        #train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)],generator=torch.Generator(device='cuda'))
        #dataset = DSprites(root='/nfs/data_chaos/ysong/simplegan_experiments/dataset', transform=transforms.ToTensor())
        data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator(device='cuda'))
        trn = TrainerOTScratch(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
                                data_loader=data_loader)

    # Train
    trn.train(generator=G, support_sets=S, prior = S_Prior)
    trn.eval(generator=G, support_sets=S, prior = S_Prior)


if __name__ == '__main__':
    main()
