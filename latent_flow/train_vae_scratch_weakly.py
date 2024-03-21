import argparse
import torch

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from latent_flow.trainers.aux import create_exp_dir
from latent_flow.datasets.Shapes3d import Shapes3D
from latent_flow.models.vae import ConvVAE, ConvEncoder3
from latent_flow.models.DiffPDE import DiffPDE
from latent_flow.models.HJPDE import HJPDE
from latent_flow.trainers.trainer_ot_scratch_weakly import TrainerOTScratchWeakly
from latent_flow.trainers.trainer_ot_scratch_weakly_shapes import TrainerOTScratchWeaklyShapes

torch.set_default_tensor_type("torch.cuda.FloatTensor")


def main():
    parser = argparse.ArgumentParser(description="Laten Flow training script")

    # === PDEs ======================================================================== #
    parser.add_argument("-K", "--num-support-sets", type=int, help="set number of PDEs")
    parser.add_argument("-D", "--num-timesteps", type=int, help="set number of timesteps")
    parser.add_argument("--support-set-lr", type=float, default=1e-4, help="set learning rate")

    # === Training =================================================================================================== #
    parser.add_argument("--max-iter", type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument("--batch-size", type=int, default=128, help="set batch size")
    parser.add_argument("--log-freq", default=10, type=int, help="set number iterations per log")
    parser.add_argument("--ckp-freq", default=1000, type=int, help="set number iterations per checkpoint model saving")
    parser.add_argument("--tensorboard", action="store_true", help="use tensorboard")
    parser.add_argument("--dsprites", type=bool, default=False)
    parser.add_argument("--shapes3d", type=bool, default=False)
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

    if args.shapes3d == True:
        G = ConvVAE(num_channel=3, latent_size=15 * 15 + 1, img_size=64)
        print("Intialize Shapes3D VAE")
    else:
        G = ConvVAE(num_channel=3, latent_size=18 * 18, img_size=28)
        print("Intialize MNIST VAE")

    # Build PDEs
    print("#. Build PDEs...")
    print("  \\__Number of Support Sets    : {}".format(args.num_support_sets))
    print("  \\__Number of Support Dipoles : {}".format(args.num_timesteps))
    print("  \\__Support Vectors dim       : {}".format(G.latent_size))

    S = HJPDE(num_support_sets=args.num_support_sets, num_timesteps=args.timesteps, support_vectors_dim=G.latent_size)

    S_Prior = DiffPDE(
        num_support_sets=args.num_support_sets, num_timesteps=args.num_timesteps, support_vectors_dim=G.latent_size
    )

    # Build transformation index predictor
    print("#. Build index predictor R...")

    if args.shapes3d == True:
        R = ConvEncoder3(s_dim=15 * 15 + 1, n_cin=3 * 8, n_hw=64, latent_size=5)
    else:
        R = ConvEncoder3(s_dim=18 * 18, n_cin=3 * 9, n_hw=28, latent_size=3)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    if args.shapes3d:
        print("SHAPES3D DATASET LOADING")
        train_tx = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((64, 64)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = Shapes3D(root="./data/ysong/3dshapes.h5", train=True, transform=None)
        # train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)],generator=torch.Generator(device='cuda'))
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator(device="cuda"),
        )
        trn = TrainerOTScratchWeaklyShapes(
            params=args,
            exp_dir=exp_dir,
            use_cuda=use_cuda,
            multi_gpu=multi_gpu,
            data_loader=data_loader,
            dataset=dataset,
        )
    else:
        print("MNIST DATASET LOADING")
        # train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=data_config['batch_size'])
        dataset = MNIST(root="./data/ysong/", train=True, transform=transforms.ToTensor(), download=True)
        train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)],generator=torch.Generator(device='cuda'))
        data_loader_train = DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator(device="cuda"),
        )
        data_loader_val = DataLoader(
            dataset=val_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator(device="cuda"),
        )
        trn = TrainerOTScratchWeakly(
            params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu, data_loader=data_loader_train
        )

    # Train
    trn.train(generator=G, support_sets=S, reconstructor=R, prior=S_Prior)
    trn.data_loader = data_loader_val
    trn.eval(generator=G, support_sets=S, reconstructor=R, prior=S_Prior)


if __name__ == "__main__":
    main()
