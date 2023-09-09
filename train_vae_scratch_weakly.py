import argparse
import torch
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2, build_sngan
from models.vae import VAE, Encoder, ConvVAE, ConvEncoder, ConvEncoder2, ConvEncoder3
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import numpy as np
import os
from transforms import *
from dsprites import *
from Shapes3d import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DSprites(torch.utils.data.Dataset):

    def __init__(self,root, transform):
        super().__init__()
        data_dir = root
        self.data = np.load(os.path.join(data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), encoding='bytes')
        self.images = self.data['imgs']
        self.latents_values = self.data['latents_values']
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index:index+1]
        # to tensor
        img = torch.from_numpy(img.astype('float32'))
        # normalize
        #img = img.mul(2).sub(1)
        #img = self.transform(img)
        return img, self.latents_values[index]

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)

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
    parser.add_argument("--dsprites", type=bool, default=False)
    parser.add_argument("--shapes3d", type=bool, default=False)
    parser.add_argument("--madelung", type=bool, default=False)
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
    if args.dsprites == True:
        G = ConvVAE(num_channel=1,latent_size=15*15+1,img_size=64)
        #G = ConvVAE(num_channel=1,latent_size=256)
        #G.load_state_dict(torch.load("vae_dsprites.pt", map_location='cpu'))
        print("Intialize DSPRITES VAE")
    elif args.shapes3d == True:
        G = ConvVAE(num_channel=3,latent_size=15*15+1,img_size=64)
        #G = ConvVAE(num_channel=1,latent_size=256)
        #G.load_state_dict(torch.load("vae_dsprites.pt", map_location='cpu'))
        print("Intialize Shapes3D VAE")
    else:
        G = ConvVAE(num_channel=3, latent_size=18 * 18, img_size=28)
        #G = VAE(encoder_layer_sizes=[784*3,256],latent_size=16,decoder_layer_sizes=[256,784*3])
        #G.load_state_dict(torch.load("vae_mnist.pt", map_location='cpu'))
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

    if args.shapes3d == True:
        S = WaFlow(num_support_sets=args.num_support_sets,
                  num_support_dipoles=args.num_support_dipoles,
                  support_vectors_dim=G.latent_size,
                  learn_alphas=args.learn_alphas,
                  learn_gammas=args.learn_gammas,
                  gamma=1.0 / G.latent_size if args.gamma is None else args.gamma,
                  img_size=64,
                  madelung_flow=args.madelung
                  )
    else:
        S = WaFlow(num_support_sets=args.num_support_sets,
                  num_support_dipoles=args.num_support_dipoles,
                  support_vectors_dim=G.latent_size,
                  learn_alphas=args.learn_alphas,
                  learn_gammas=args.learn_gammas,
                  gamma=1.0 / G.latent_size if args.gamma is None else args.gamma,
                  madelung_flow=args.madelung
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

    #R = Reconstructor(reconstructor_type=args.reconstructor_type,
    #                  dim_index=S.num_support_sets,
    #                  dim_time=S.num_support_dipoles,
    #                  channels=1 if args.gan_type == 'SNGAN_MNIST' else 3)
    if args.dsprites == True:
        #R = ConvEncoder(num_channel=2,latent_size=256)
        R = ConvEncoder3(s_dim=15 * 15+1,n_cin=1*7,n_hw=64,latent_size=5)
    elif args.shapes3d == True:
        R = ConvEncoder3(s_dim=15 * 15+1,n_cin=3*8,n_hw=64,latent_size=5)
    else:
        R = ConvEncoder3(s_dim=18 * 18,n_cin=3*9,n_hw=28,latent_size=3)
        #R = Encoder(layer_sizes=[784*6,256], latent_size=args.num_support_sets)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    config = {
        'wandb_on': False,
        'lr': 1e-4,
        'momentum': 0.9,
        'max_epochs': 100,
        'eval_epochs': 5,
        'dataset': 'DSprites',
        'seq_transforms': ['posX', 'posY','orientation', 'scale', 'shape'],
        'avail_transforms': ['posX', 'posY', 'orientation', 'scale', 'shape'],
        'seed': 1,
        'n_caps': 15,
        'cap_dim': 15,
        'n_transforms': args.num_support_dipoles // 2,
        'max_transform_len': 30,
        'mu_init': 30.0,
        'n_off_diag': 0,
        'group_kernel': (10, 10, 1),
        'n_is_samples': 10
    }
    if args.dsprites:
        #data_config['dataset']='DSPRITES'
        print("DSPRITES DATASET LOADING")
        #train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=data_config['batch_size'])
        #data_loader = get_dataloader(dir='/nfs/data_lambda/ysong/',
        #                             seq_transforms=config['seq_transforms'],
        #                             avail_transforms=config['avail_transforms'],
        #                             seq_len=config['n_transforms'],
        #                             max_transform_len=config['max_transform_len'],
        #                             batch_size=args.batch_size)
        dataset = DSpritesDataset(dir='/nfs/data_lambda/ysong/', seq_transforms=config['seq_transforms'],
                                  avail_transforms=config['avail_transforms'],
                                  max_transform_len=config['max_transform_len'],
                                  seq_len=config['n_transforms'])
        data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator(device='cuda'))
        trn = TrainerOTScratchWeaklyDsprites(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
                                data_loader=data_loader,dataset=dataset)
    elif args.shapes3d:
        print("SHAPES3D DATASET LOADING")
        train_tx = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = Shapes3D(root='/nfs/data_lambda/ysong/3dshapes.h5', train=True, transform=None)
        #train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)],generator=torch.Generator(device='cuda'))
        data_loader = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator(device='cuda'))
        trn = TrainerOTScratchWeaklyShapes(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
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
        trn = TrainerOTScratchWeakly(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu,
                                data_loader=data_loader)

    # Train
    trn.train(generator=G, support_sets=S, reconstructor=R, prior = S_Prior)
    trn.eval(generator=G, support_sets=S, reconstructor=R, prior = S_Prior)


if __name__ == '__main__':
    main()
