import argparse
import torch
import torchvision

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from latent_flow.trainers.aux import create_exp_dir
from latent_flow.datasets.Falor3D import Falor3D
from latent_flow.datasets.Isaac3D import Isaac3D
from latent_flow.models.vae import ConvVAE2
from latent_flow.models.DiffPDE import DiffPDE
from latent_flow.models.HJPDE import HJPDE
from latent_flow.trainers.trainer_falcol_isaac_scratch import TrainerFalcolIsaacScratch


def main():
    parser = argparse.ArgumentParser(description="PDE training script")

    # === PDEs (S) ======================================================================== #
    parser.add_argument("-K", "--num-support-sets", type=int, help="set number of PDEs")
    parser.add_argument("-D", "--num-timesteps", type=int, help="set number of timesteps")
    parser.add_argument("--support-set-lr", type=float, default=1e-4, help="set learning rate")
    # === Training =================================================================================================== #
    parser.add_argument("--max-iter", type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument("--batch-size", type=int, default=128, help="set batch size")
    parser.add_argument("--log-freq", default=10, type=int, help="set number iterations per log")
    parser.add_argument("--ckp-freq", default=1000, type=int, help="set number iterations per checkpoint model saving")
    parser.add_argument("--tensorboard", action="store_true", help="use tensorboard")
    parser.add_argument("--isaac", type=bool, default=False)
    # === CUDA ======================================================================================================= #
    parser.add_argument("--cuda", dest="cuda", action="store_true", help="use CUDA during training")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="do NOT use CUDA during training")
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
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print(
                "*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                "                 Run with --cuda for optimal training speed."
            )
            torch.set_default_tensor_type("torch.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    G = ConvVAE2(num_channel=3, latent_size=18 * 18, img_size=128)
    print("Intialize MNIST VAE")

    # Build PDEs
    print("#. Build PDEs S...")
    print("  \\__Number of Support Sets    : {}".format(args.num_support_sets))
    print("  \\__Number of Timesteps : {}".format(args.num_timesteps))
    print("  \\__Support Vectors dim       : {}".format(G.latent_size))

    S = HJPDE(
        num_support_sets=args.num_support_sets, num_timesteps=args.num_timesteps, support_vectors_dim=G.latent_size
    )

    S_Prior = DiffPDE(
        num_support_sets=args.num_support_sets, num_timesteps=args.num_timesteps, support_vectors_dim=G.latent_size
    )

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    if args.isaac:
        print("ISSAC DATASET LOADING")
        train_tx = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        dataset = Isaac3D(root="./data/Isaac3D_down128/images", train=True, transform=train_tx)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator(device="cuda"),
        )
        trn = TrainerFalcolIsaacScratch(
            params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu, data_loader=data_loader
        )

    else:
        train_tx = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
        print("FALOR DATASET LOADING")
        dataset = Falor3D(root="./data/Falcor3D_down128/images", train=True, transform=train_tx)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator(device="cuda"),
        )
        trn = TrainerFalcolIsaacScratch(
            params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu, data_loader=data_loader
        )

    # Train
    trn.train(generator=G, support_sets=S, prior=S_Prior)
    trn.eval(generator=G, support_sets=S, prior=S_Prior)


if __name__ == "__main__":
    main()
