import torch
from torch import nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings,self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = (self.dim) // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Diffusion_Log(nn.Module):
    def __init__(self, n_in,timestep):
        super(Diffusion_Log,self).__init__()

        self.D = torch.ones(1, 1, requires_grad=True)
        self.mu = nn.Parameter(torch.zeros(timestep, 1, requires_grad=True))
        self.sigma = nn.Parameter(torch.ones(timestep, 1, requires_grad=True))
        #Assume rho follows gaussian distribution

    def forward(self, rho, z, time_step):
        phi = -self.D * rho
        if time_step == 1:
            nabla_phi = self.D * z
            nabla_nabla_phi = self.D
        else:
            nabla_phi =  self.D * (z-self.mu[time_step-1]) / (self.sigma[time_step-1]**2 + 1e-5)
            nabla_nabla_phi = self.D / (self.sigma[time_step-1] ** 2 + 1e-5 )

        return phi, nabla_phi, nabla_nabla_phi


class WavePDE(nn.Module):
    def __init__(self, num_support_sets, num_support_dipoles, support_vectors_dim,
                 learn_alphas=False, learn_gammas=False, gamma=None):
        """SupportSets class constructor.

        Args:
            num_support_sets (int)    : number of support sets (each one defining a warping function)
            num_support_dipoles (int) : number of support dipoles per support set (per warping function)
            support_vectors_dim (int) : dimensionality of support vectors (latent space dimensionality, z_dim)
            learn_alphas (bool)       : learn RBF alphas
            learn_gammas (bool)       : learn RBF gammas
            gamma (float)             : RBF gamma parameter (by default set to the inverse of the latent space
                                        dimensionality)
        """
        super(WavePDE, self).__init__()
        self.num_support_sets = num_support_sets
        self.num_support_dipoles = num_support_dipoles
        self.support_vectors_dim = support_vectors_dim
        self.MLP_SET = nn.ModuleList([Diffusion_Log(n_in=support_vectors_dim, timestep=num_support_dipoles//2) for i in range(num_support_sets)])


    def loss_pde(self, mlp, z, t, rho):
        z = z.clone().requires_grad_()
        #z.requires_grad = True
        rho = rho.clone().requires_grad_()
        u,u_z,u_zz = mlp(rho, z, t)
        return 0.0, u_z, u, u_zz

    #def forward(self, index, z, t, generator):
    #    mse_ic = self.loss_ic(self.MLP_SET[index], z)
    #    _, u_z, u = self.loss_pde(self.MLP_SET[index], z, t, self.c[index], generator)
    #    mse_jvp = self.loss_jvp(self.MLP_SET[index], z, t, generator)
    #    mse_pde = 0.0
    #    half_range = self.num_support_dipoles // 2
    #    for i in range(0, half_range):
    #        t_index = i * torch.ones(1, 1, requires_grad=True)
    #        mse_pde_t_index, _ , _ = self.loss_pde(self.MLP_SET[index],z,t_index,self.c[index],generator)
    #        mse_pde = mse_pde + mse_pde_t_index
    #    loss = mse_ic + mse_pde / self.num_support_dipoles - mse_jvp
    #    return u, u_z, loss

    def index_forward(self, index_pred, z, t, rho):
        mse_pde_t_index, u_z, u, u_zz = 0.0,0.0,0.0,0.0
        for index in range(self.num_support_sets):
            mse_pde_t_index_temp, u_z_temp, u_temp, u_zz_temp = self.loss_pde(self.MLP_SET[index], z, t[0].long(), rho)
            mse_pde_t_index += index_pred[:,index:index+1].mean() * mse_pde_t_index_temp
            u_z += index_pred[:,index:index+1] * u_z_temp
            u += index_pred[:,index:index+1] * u_temp
            u_zz += index_pred[:,index:index+1] * u_zz_temp
        loss_step = mse_pde_t_index
        return u, loss_step, u_z, u_zz

    def forward(self, index, z, t, rho):
        mse_pde_t_index, u, u_z, u_zz = self.loss_pde(self.MLP_SET[index], z, t[0].long(), rho)
        loss_step = mse_pde_t_index
        return u, loss_step, u_z, u_zz

    def inference(self, index, z, t):
        z = z.clone().requires_grad_()
        t = t.clone().requires_grad_()
        u, u_z, u_zz =  self.MLP_SET[index](torch.zeros_like(z),z, int(t[0]))
        return u, u_z, u_zz
