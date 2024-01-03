import os
import os.path as osp
import argparse
import torch
import json
import numpy as np
from torch import nn
from hashlib import sha1
from torchvision.transforms import ToPILImage

from latent_flow.models.vae import ConvVAE, ConvVAE2
from latent_flow.trainers.aux import sample_z, update_progress, update_stdout


def tensor2image(tensor, adaptive=False):
    tensor = tensor.squeeze(dim=0)
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def main():
    parser = argparse.ArgumentParser(description="Sample a  VAE latent space and generate images")
    parser.add_argument("-v", "--verbose", action="store_true", help="set verbose mode on")
    parser.add_argument("-g", "--gan-type", type=str, required=True, help="VAE model type")
    parser.add_argument("--num-samples", type=int, default=4, help="number of latent codes to sample")
    parser.add_argument("--pool", type=str, help="name of latent codes/images pool")
    parser.add_argument("--cuda", dest="cuda", action="store_true", help="use CUDA during training")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false", help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Create output dir for generated images
    out_dir = osp.join("experiments", "latent_codes", args.gan_type)
    if args.pool:
        out_dir = osp.join(out_dir, args.pool)
    else:
        out_dir = osp.join(
            out_dir,
            "{}_{}".format(args.gan_type, args.num_samples),
        )
    os.makedirs(out_dir, exist_ok=True)

    # Save argument in json file
    with open(osp.join(out_dir, "args.json"), "w") as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # Set default tensor type
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        if not args.cuda:
            print(
                "*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                "                 Run with --cuda for optimal training speed."
            )
            torch.set_default_tensor_type("torch.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
    use_cuda = args.cuda and torch.cuda.is_available()

    if args.gan_type == "VAE_MNIST":
        G = ConvVAE(num_channel=3, latent_size=18 * 18, img_size=28)
    elif args.gan_type == "VAE_Shapes":
        G = ConvVAE(num_channel=3, latent_size=15 * 15 + 1, img_size=64)
    elif args.gan_type == "VAE_FALOR" or args.gan_type == "VAE_ISAAC":
        G = ConvVAE2(num_channel=3, latent_size=18 * 18, img_size=128)

    # Upload generator to GPU
    if use_cuda:
        G = G.cuda()

    # Set generator to evaluation mode
    G.eval()

    # Latent codes sampling

    zs = sample_z(batch_size=args.num_samples, dim_z=G.latent_size, truncation=args.z_truncation)

    if use_cuda:
        zs = zs.cuda()

    if args.verbose:
        print("#. Generate images...")
        print("  \\__{}".format(out_dir))

    # Iterate over given latent codes
    for i in range(args.num_samples):
        # Un-squeeze current latent code in shape [1, dim] and create hash code for it
        z = zs[i, :].unsqueeze(0)
        latent_code_hash = sha1(z.cpu().numpy()).hexdigest()

        if args.verbose:
            update_progress(
                "  \\__.Latent code hash: {} [{:03d}/{:03d}] ".format(latent_code_hash, i + 1, args.num_samples),
                args.num_samples,
                i,
            )

        # Create directory for current latent code
        latent_code_dir = osp.join(out_dir, "{}".format(latent_code_hash))
        os.makedirs(latent_code_dir, exist_ok=True)

        # Save latent code tensor under `latent_code_dir`
        torch.save(z.cpu(), osp.join(latent_code_dir, "latent_code.pt"))

        # Generate image for the given latent code z
        with torch.no_grad():
            img = G.inference(z).cpu()
        # Convert image's tensor into an RGB image and save it
        img_pil = tensor2image(img, adaptive=True)
        img_pil.save(osp.join(latent_code_dir, "image.jpg"), "JPEG", quality=95, optimize=True, progressive=True)

    if args.verbose:
        update_stdout(1)
        print()
        print()


if __name__ == "__main__":
    main()
