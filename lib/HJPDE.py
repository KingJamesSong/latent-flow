import torch
from torch import nn
import math
from torch.autograd import grad


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = (self.dim) // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super(MLP, self).__init__()

        self.layer_x = nn.Linear(n_in, n_in)
        self.activation1 = nn.Tanh()

        # Time position embedding used in Transformers and Diffusion Models
        self.layer_pos = SinusoidalPositionEmbeddings(n_in)
        self.layer_time = nn.Linear(n_in, n_in)
        self.activation2 = nn.GELU()
        # self.activation2 = nn.Tanh()
        self.layer_time2 = nn.Linear(n_in, n_in)

        self.layer_fusion = nn.Linear(n_in, n_in)
        self.activation3 = nn.Tanh()
        self.layer_out = nn.Linear(n_in, n_out)
        self.activation4 = nn.Tanh()

    def forward(self, x, time):
        x = self.layer_x(x)
        x = self.activation1(x)

        time = self.layer_pos(time)
        time = self.layer_time(time)
        time = self.activation2(time)
        time = self.layer_time2(time)

        x = self.layer_fusion(x + time)
        x = self.activation3(x)
        x = self.layer_out(x)
        x = self.activation4(x)
        return x


class HJPDE(nn.Module):
    def __init__(self, num_support_sets, num_timesteps, support_vectors_dim):
        super(HJPDE, self).__init__()
        self.num_support_sets = num_support_sets
        self.num_timesteps = num_timesteps
        self.support_vectors_dim = support_vectors_dim

        self.MLP_SET = nn.ModuleList([MLP(n_in=support_vectors_dim, n_out=1) for i in range(num_support_sets)])
        self.FORCE_SET = nn.ModuleList([MLP(n_in=support_vectors_dim, n_out=1) for i in range(num_support_sets)])

    # initial condition loss
    def loss_ic(self, mlp, z, force):
        z = z.clone().requires_grad_()
        u = mlp(z, torch.FloatTensor([0]).to(z))
        u_z = grad(u.sum(), z, create_graph=True)[0]

        f = force(z, torch.FloatTensor([0]).to(z))
        mse_0 = torch.mean(torch.square(u_z)) + torch.mean(torch.square(f))

        return mse_0

    # Loss of PDE and Div
    def loss_pde(self, mlp, z, t, force):
        z = z.clone().requires_grad_()
        # z.requires_grad = True
        t = t.clone().requires_grad_()
        # t.requires_grad = True

        u = mlp(z, t)
        u_t = grad(u.sum(), t, create_graph=True)[0]

        u_z = grad(u.sum(), z, create_graph=True)[0]
        u_zz = grad(u_z.sum(), z, create_graph=True)[0]

        f = force(z, t)
        # PDE
        pde = u_t + 0.5 * torch.mean(u_z**2) + (f**2)
        mse_pde = torch.mean(torch.square(pde))
        return mse_pde, u_z, u, u_zz

    # weakly-supervised training
    def index_forward(self, index_pred, z, t):
        mse_pde_t_index, u_z, u, u_zz = 0.0, 0.0, 0.0, 0.0
        for index in range(self.num_support_sets):
            mse_pde_t_index_temp, u_z_temp, u_temp, u_zz_temp = self.loss_pde(
                self.MLP_SET[index], z, t, self.FORCE_SET[index]
            )
            mse_pde_t_index += index_pred[:, index : index + 1].mean() * mse_pde_t_index_temp
            u_z += index_pred[:, index : index + 1] * u_z_temp
            u += index_pred[:, index : index + 1] * u_temp
            u_zz += index_pred[:, index : index + 1] * u_zz_temp
        loss_step = mse_pde_t_index
        if t[0] == 1:
            mse_ic = 0.0
            for index in range(self.num_support_sets):
                mse_ic_temp = self.loss_ic(self.MLP_SET[index], z, self.FORCE_SET[index])
                mse_ic = index_pred[:, index : index + 1].mean() * mse_ic_temp
            loss_step += mse_ic
        return u, loss_step, u_z, u_zz

    def forward(self, index, z, t):
        mse_pde_t_index, u_z, u, u_zz = self.loss_pde(self.MLP_SET[index], z, t, self.FORCE_SET[index])
        loss_step = mse_pde_t_index
        if t[0] == 1:
            mse_ic = self.loss_ic(self.MLP_SET[index], z, self.FORCE_SET[index])
            loss_step += mse_ic
        return u, loss_step, u_z, u_zz

    def inference(self, index, z, t):
        z = z.clone().requires_grad_()
        t = t.clone().requires_grad_()
        u = self.MLP_SET[index](z, t)
        u_z = grad(u.sum(), z, create_graph=True)[0]
        u_zz = grad(u_z.sum(), z, create_graph=True)[0]

        return u, u_z, u_zz

    # weakly-supervised inference
    def index_inference(self, index_pred, z, t):
        z = z.clone().requires_grad_()
        t = t.clone().requires_grad_()
        u, u_z, u_zz = 0.0, 0.0, 0.0
        for index in range(self.num_support_sets):
            u_temp = self.MLP_SET[index](z, t)
            u_z_temp = grad(u_temp.sum(), z, create_graph=True)[0]
            u_zz_temp = grad(u_z_temp.sum(), z, create_graph=True)[0]
            u += index_pred[:, index : index + 1] * u_temp
            u_z += index_pred[:, index : index + 1] * u_z_temp
            u_zz += index_pred[:, index : index + 1] * u_zz_temp
        return u, u_z, u_zz
