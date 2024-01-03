# latent-factozied-flow

NeurIPS23 "[Flow Factorized Representation Learning](https://arxiv.org/abs/2309.13167)" 

[Yue Song](https://kingjamessong.github.io/)<sup>1,2</sup>, [T. Anderson Keller](https://scholar.google.com/citations?hl=en&user=Tb86kC0AAAAJ)<sup>1</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>2</sup>, [Max Welling](https://scholar.google.com/citations?user=8200InoAAAAJ&hl=en)<sup>1</sup>  
<sup>1</sup>University of Amsterdam, the Netherlands <br>
<sup>2</sup>University of Trento, Italy <be> 




| MNIST        |                    Shapes3D                     |  Falcol3D  | Isaac3D |
:-------------------------:|:-------------------------------------------:|:-------------------------:|:-------------------------:|
|<img src="imgs/mnist_trans1 (2).gif" /><br> <img src="imgs/mnist_trans2 (2).gif" /><br><img src="imgs/mnist_trans3 (2).gif" />| <img src="imgs/shapes_trans1 (1).gif" width="140" height="28" /> <br><img src="imgs/shapes_trans2 (1).gif" width="140" height="28" /> <br> <img src="imgs/shapes_trans3 (1).gif" width="140" height="28" /> |<img src="imgs/falor1.gif" width="96" height="96" />| <img src="imgs/issac1.gif" width="96" height="96" />  <img src="imgs/issac2.gif" width="96" height="96" /> |



## Overview

<p align="center">
<img src="imgs/surface.jpg" width="500px"/>
<br>
Illustration of our flow factorized representation learning: at each point in the latent space we have a distinct set of tangent directions ∇uk which define different transformations we would like to model in the image space. For each path, the latent sample evolves to the target on the potential landscape following dynamic optimal transport.
</p>

<p align="center">
<img src="imgs/graphical_model.png" width="800px"/>
<br>
Depiction of our model in plate notation. (Left) Supervised, (Right) Weakly-supervised. White nodes denote latent variables, shaded nodes denote observed variables, solid lines denote the generative model, and dashed lines denote the approximate posterior. We see, as in a standard VAE framework, our model approximates the initial one-step posterior p(z0|x0), but additionally approximates the conditional transition distribution p(zt|zt−1, k) through dynamic optimal transport over a potential landscape.
</p>

## Setup
First, clone the repository and navigate into it:

```bash
git clone https://github.com/KingJamesSong/latent-flow.git
cd latent-flow
```

We recommend setting up a new conda environment for this project. You can do this using the following command:

```bash
conda create --name latent-flow-env python=3.11
conda activate latent-flow-env
```

Next, install the necessary dependencies. This project requires PyTorch. You can find the installation instructions on the [PyTorch setup page](https://pytorch.org/get-started/locally/).


After installing PyTorch, install the remaining dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

For development purposes, you may also want to install the dependencies listed in `requirements_dev.txt`:

```bash
pip install -r requirements_dev.txt
```
It is recommended to set your IDEs autoformatter to use *black* and to enable "format on save".

Finally, install the package itself. If you plan on modifying the code, install it in editable mode using the `-e` option:

```bash
pip install -e .
```
This will allow your changes to be immediately reflected in the installed package.

The code assumes that all datasets are placed in the `./data` folder. This folder is going to be created automatically if necessary. 
However, if you'd like to use a different folder for your datasets, you can create a symbolic link to that folder. This can be done using the following commands:

For Unix-based systems (Linux, MacOS), use the `ln` command:
```bash
ln -s /path/to/your/dataset/folder ./data
```
This command creates a symbolic link named `./data` that points to `/path/to/your/dataset/folder`.

For Windows systems, use the `mklink` command:
```cmd
mklink /D .\data C:\path\to\your\dataset\folder
```
This command creates a symbolic link named `.\data` that points to `C:\path\to\your\dataset\folder`.

Please replace `/path/to/your/dataset/folder` and `C:\path\to\your\dataset\folder` with the actual path to your dataset folder.

## Usage

Please check [the scripts folder](https://github.com/KingJamesSong/latent-flow/tree/main/scripts) for the training and evaluation codes.

## Citation

If you think the code is helpful to your research, please consider citing our paper:

```
@inproceedings{song2023flow,
  title={Flow Factorized Representation Learning},
  author={Song, Yue and Keller, Andy and Sebe, Nicu and Welling, Max},
  booktitle={NeurIPS},
  year={2023}
}
```

If you have any questions or suggestions, please feel free to contact me via `yue.song@unitn.it`.
