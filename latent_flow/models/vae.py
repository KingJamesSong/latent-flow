import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_sigmoid_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.sigmoid(y / temperature)

def gumbel_softmax(logits, temperature, categorical_dim, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, categorical_dim)

def gumbel_sigmoid(logits, temperature, categorical_dim, hard=False):
    """
    ST-gumple-sigmoid
    input: [*, n_class]
    return: flatten --> [*, n_class] an binary vector
    """
    y = gumbel_sigmoid_sample(logits, temperature)

    if not hard:
        return y.view(-1, categorical_dim)
    shape = y.size()
    #_, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard[y>0.5]=1.0
    #y_hard.scatter_(1, ind.view(y.size(0), -1), 1)
    #for i in ind:
    #    y_hard[:,i]=1.0
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, categorical_dim)


# VAE models for MNIST and Shapes3D
class ConvVAE(nn.Module):
    def __init__(self, num_channel, latent_size, img_size):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = ConvEncoder2(n_cin=num_channel, s_dim=latent_size, n_hw=img_size)
        self.decoder = ConvDecoder2(n_cout=num_channel, s_dim=latent_size, n_hw=img_size)

    def forward(self, x):
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z):
        recon_x = self.decoder(z)
        return recon_x


# VAE models for Falcol3D and Isaac3D
class ConvVAE2(nn.Module):
    def __init__(self, num_channel, latent_size, img_size):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = ConvEncoder4(n_cin=num_channel, s_dim=latent_size, n_hw=img_size)
        self.decoder = ConvDecoder4(n_cout=num_channel, s_dim=latent_size, n_hw=img_size)

    def forward(self, x):
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z):
        recon_x = self.decoder(z)
        return recon_x


class ConvEncoder2(nn.Module):
    def __init__(self, s_dim, n_cin, n_hw):
        super().__init__()

        self.s_dim = s_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 2, kernel_size=1, stride=1, padding=0),
        )

        self.linear_means = nn.Linear(s_dim * 2, s_dim)
        self.linear_log_var = nn.Linear(s_dim * 2, s_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.s_dim * 2)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class ConvDecoder2(nn.Module):
    def __init__(self, s_dim, n_cout, n_hw):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(s_dim, s_dim),  # B, 256
            View((-1, s_dim, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim * 2, s_dim * 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim * 3, s_dim * 2, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim * 2, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, n_cout, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.decoder(z)

        return x


# Encoder with Gumbel-Softmax trick
class ConvEncoder3(nn.Module):
    def __init__(self, s_dim, n_cin, n_hw, latent_size):
        super().__init__()

        self.s_dim = s_dim
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim, kernel_size=1, stride=1, padding=0),
        )
        self.temp_min = 0.05
        self.ANNEAL_RATE = 0.00003
        self.temp_ini = 1.0
        self.temp = 1.0
        self.linear1 = nn.Linear(s_dim, self.latent_size)

    def forward(self, x, iter):
        x = self.encoder(x)
        x = x.view(-1, self.s_dim)
        x = self.linear1(x)
        if iter % 100 == 1:
            self.temp = np.maximum(self.temp_ini * np.exp(-self.ANNEAL_RATE * iter), self.temp_min)
        z = gumbel_softmax(x, temperature=self.temp, categorical_dim=self.latent_size, hard=True)
        return z
        # return F.softmax(z,dim=-1)


class ConvEncoder4(nn.Module):
    def __init__(self, s_dim, n_cin, n_hw):
        super().__init__()

        self.s_dim = s_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=(n_hw // 16), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim, kernel_size=1, stride=1, padding=0),
        )

        self.linear_means = nn.Linear(s_dim, s_dim)
        self.linear_log_var = nn.Linear(s_dim, s_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.s_dim)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std


class ConvDecoder4(nn.Module):
    def __init__(self, s_dim, n_cout, n_hw):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(s_dim, s_dim),  # B, 256
            View((-1, s_dim, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=(n_hw // 16), stride=1, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(s_dim, n_cout, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.decoder(z)

        return x

#Gumbel-sigmoid trick to infer spike and slab components
class ConvEncoder3_Unsuper(nn.Module):

    def __init__(self, s_dim, n_cin, n_hw, latent_size):
        super().__init__()

        self.s_dim = s_dim
        self.latent_size = latent_size
        self.encoder =nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim, kernel_size=1, stride=1, padding=0),
        )
        self.temp_min = 0.05
        self.ANNEAL_RATE = 0.00003
        self.temp_ini = 1.0
        self.temp = 1.0
        #Spike infer
        self.linear_y = nn.Linear(s_dim, self.latent_size)
        #Slab infer
        self.encoder_g = nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim, kernel_size=1, stride=1, padding=0),
        )
        self.linear_g = nn.Linear(s_dim, self.latent_size)
        self.act_g = nn.Sigmoid()

    def forward(self, input, iter):
        if iter % 100 == 1:
            self.temp = np.maximum(self.temp_ini * np.exp(-self.ANNEAL_RATE * iter), self.temp_min)
        x = self.encoder(input)
        x = x.view(-1, self.s_dim)
        z_temp1 = self.linear_y(x)
        z1 = gumbel_sigmoid(z_temp1, temperature=self.temp, categorical_dim=self.latent_size, hard=False)

        x_g = self.encoder_g(input)
        x_g = x_g.view(-1, self.s_dim)
        g = 2*self.act_g(self.linear_g(x_g))
        return z1,g

#Gumbel-sigmoid trick to infer spike and slab components (two spike components for rotational and gradient fields)
class ConvEncoder3_Unsuper2(nn.Module):

    def __init__(self, s_dim, n_cin, n_hw, latent_size):
        super().__init__()

        self.s_dim = s_dim
        self.latent_size = latent_size
        self.encoder =nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim, kernel_size=1, stride=1, padding=0),
        )
        self.temp_min = 0.05
        self.ANNEAL_RATE = 0.00003
        self.temp_ini = 1.0
        self.temp = 1.0
        self.linear_y1 = nn.Linear(s_dim, self.latent_size)
        self.linear_y2 = nn.Linear(s_dim, self.latent_size)

        self.encoder_g = nn.Sequential(
            nn.Conv2d(n_cin, s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim, s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim * 3, kernel_size=(n_hw // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 3, s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(s_dim * 2, s_dim, kernel_size=1, stride=1, padding=0),
        )
        self.linear_g = nn.Linear(s_dim, self.latent_size)
        self.act_g = nn.Sigmoid()

        #self.linear_g2 = nn.Linear(s_dim, self.latent_size)
        #self.act_g2 = nn.Sigmoid()
        #self.linear_gtilde_means = nn.Linear(s_dim, self.latent_size)
        #self.linear_gtilde_vars = nn.Linear(s_dim, self.latent_size)
        #self.linear1 = nn.ModuleList([nn.Linear(s_dim, 2) for i in range(self.latent_size)])

    def forward(self, input, iter):
        if iter % 100 == 1:
            self.temp = np.maximum(self.temp_ini * np.exp(-self.ANNEAL_RATE * iter), self.temp_min)
        x = self.encoder(input)
        x = x.view(-1, self.s_dim)
        z_temp1 = self.linear_y1(x)
        z_temp2 = self.linear_y2(x)
        z1 = gumbel_sigmoid(z_temp1, temperature=self.temp, categorical_dim=self.latent_size, hard=False)
        z2 = gumbel_sigmoid(z_temp2, temperature=self.temp, categorical_dim=self.latent_size, hard=False)

        x_g = self.encoder_g(input)
        x_g = x_g.view(-1, self.s_dim)

        g1 = 2 * self.act_g(self.linear_g(x_g))
        return z1,z2,g1
