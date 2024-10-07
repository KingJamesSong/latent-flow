import torch
from torch import nn
import math
from torch.autograd import grad
from .divfree import *
from torch.func import vmap

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


class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super(MLP,self).__init__()

        self.layer_x = nn.Linear(n_in, n_in)
        self.activation1 = nn.Tanh()

        #Time position embedding used in Transformers and Diffusion Models
        self.layer_pos = SinusoidalPositionEmbeddings(n_in)
        self.layer_time = nn.Linear(n_in, n_in)
        self.activation2 = nn.GELU()
        #self.activation2 = nn.Tanh()
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

        x = self.layer_fusion(x+time)
        x = self.activation3(x)
        x = self.layer_out(x)
        x = self.activation4(x)
        return x

class MLP_DIV(nn.Module):
    def __init__(self, n_in, n_out):
        super(MLP_DIV,self).__init__()

        self.layer_x = nn.Linear(n_in, n_in)
        self.activation1 = nn.Tanh()

        self.layer_out = nn.Linear(n_in, n_out)
        self.activation2 = nn.Tanh()

    def forward(self, x):
        x = self.layer_x(x)
        x = self.activation1(x)
        x = self.layer_out(x)
        x = self.activation2(x)
        return x

#Unsupervised latent flow framework 
class HJPDE2(nn.Module):
    def __init__(self, num_support_sets, num_timesteps, support_vectors_dim):

        super(HJPDE2, self).__init__()
        self.num_support_sets = num_support_sets
        self.num_timesteps = num_timesteps
        self.support_vectors_dim = support_vectors_dim
        #gradient flow
        self.MLP_SET= nn.ModuleList([MLP(n_in=support_vectors_dim,n_out=1) for i in range(num_support_sets)])
        #irrotational flow
        self.DIV_MLP = nn.ModuleList([MLP_DIV(n_in=support_vectors_dim,n_out=support_vectors_dim) for i in range(num_support_sets)])

    def loss_ic(self,mlp,z):
        z = z.clone().requires_grad_()
        u = mlp(z,torch.FloatTensor([0]).to(z))
        u_z = grad(u.sum(), z, create_graph=True)[0]

        mse_0 =  torch.mean(torch.square(u_z)) 

        return mse_0

    #Loss of PDE and Div
    def loss_pde(self,mlp,z,t,mlp_div):

        z = z.clone().requires_grad_()
        t = t.clone().requires_grad_()
        #gradient field
        u = mlp(z, t)
        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_z = grad(u.sum(), z, create_graph=True)[0]
        u_zz = grad(u_z.sum(), z, create_graph=True)[0]
        pde = u_t + 0.5 * torch.mean(u_z**2) #+ (f**2)
        #divergence-free vector fields
        div_u_fn, params, A_fn, F_fn = build_divfree_vector_field(mlp_div)
        div_u = mlp_div(z)
        div_z = div(lambda x: F_fn(params, x))
        divergence = vmap(div_z)(z)
      
        mse_pde = torch.mean(torch.square(pde)) + torch.mean(torch.square(divergence))

        return mse_pde, u_z, div_u, u, u_zz

    #Separate controls of  two flow fields
    def index_forward(self, index_pred1, index_pred2, z, t):
        mse_pde_t_index, u_z, u, u_zz = 0.0,0.0,0.0,0.0
        for index in range(self.num_support_sets):
            mse_pde_t_index_temp, u_z_temp, div_u_temp, u_temp, u_zz_temp = self.loss_pde(self.MLP_SET[index], z, t, self.DIV_MLP[index])
            mse_pde_t_index += (index_pred1[:,index:index+1].squeeze() * mse_pde_t_index_temp).mean()
            u_z += index_pred1[:,index:index+1] * u_z_temp + index_pred2[:,index:index+1] * div_u_temp
            u += index_pred1[:,index:index+1] * u_temp
            u_zz += index_pred1[:,index:index+1] * u_zz_temp
        loss_step = mse_pde_t_index
        if t[0] == 1:
            mse_ic = 0.0
            for index in range(self.num_support_sets):
                mse_ic_temp = self.loss_ic(self.MLP_SET[index], z)
                mse_ic = (index_pred1[:,index:index+1].squeeze() * mse_ic_temp).mean()
            loss_step += mse_ic
        return u, loss_step, u_z, u_zz
        
    #Single control of both flow fields
    def index_forward_single(self, index_pred, z, t):
        mse_pde_t_index, u_z, u, u_zz = 0.0,0.0,0.0,0.0
        for index in range(self.num_support_sets):
            mse_pde_t_index_temp, u_z_temp, div_u_temp, u_temp, u_zz_temp = self.loss_pde(self.MLP_SET[index], z, t, self.DIV_MLP[index])
            mse_pde_t_index += (index_pred[:,index:index+1].squeeze() * mse_pde_t_index_temp).mean()
            u_z += index_pred[:,index:index+1] * u_z_temp + index_pred[:,index:index+1] * div_u_temp
            u += index_pred[:,index:index+1] * u_temp
            u_zz += index_pred[:,index:index+1] * u_zz_temp
        loss_step = mse_pde_t_index
        if t[0] == 1:
            mse_ic = 0.0
            for index in range(self.num_support_sets):
                mse_ic_temp = self.loss_ic(self.MLP_SET[index], z)
                mse_ic = (index_pred[:,index:index+1].squeeze() * mse_ic_temp).mean()
            loss_step += mse_ic
        return u, loss_step, u_z, u_zz

    def forward(self, index, z, t):
        mse_pde_t_index, u_z, div_u, u, u_zz = self.loss_pde(self.MLP_SET[index],z,t,self.DIV_MLP[index])
        loss_step = mse_pde_t_index
        if t[0]==1:
            mse_ic = self.loss_ic(self.MLP_SET[index], z)
            loss_step += mse_ic
        return u, loss_step, u_z+div_u, u_zz

    def inference(self, index, z, t):

        z = z.clone().requires_grad_()
        t = t.clone().requires_grad_()
        u = self.MLP_SET[index](z, t)
        u_z = grad(u.sum(), z, create_graph=True)[0]
        u_zz = grad(u_z.sum(), z, create_graph=True)[0]
        div_u = self.DIV_MLP[index](z)

        return u, u_z+div_u, u_zz
        
    #Two controls
    def index_inference(self, index_pred1,index_pred2, z, t):
        z = z.clone().requires_grad_()
        t = t.clone().requires_grad_()
        u, div_u, u_z, u_zz=0.0,0.0,0.0,0.0
        for index in range(self.num_support_sets):
            u_temp = self.MLP_SET[index](z, t)
            div_u_temp = self.DIV_MLP[index](z)
            u_z_temp = grad(u_temp.sum(), z, create_graph=True)[0]
            u_zz_temp = grad(u_z_temp.sum(), z, create_graph=True)[0]
            u+= index_pred1[:,index:index+1] * u_temp
            u_z+= index_pred1[:,index:index+1] * u_z_temp
            div_u += index_pred2[:,index:index+1] * div_u_temp
            u_zz+= index_pred1[:,index:index+1] * u_zz_temp
        return u, u_z+div_u, u_zz
        
    #Single control
    def index_inference_single(self, index_pred, z, t):
        z = z.clone().requires_grad_()
        t = t.clone().requires_grad_()
        u, div_u, u_z, u_zz=0.0,0.0,0.0,0.0
        for index in range(self.num_support_sets):
            u_temp = self.MLP_SET[index](z, t)
            div_u_temp = self.DIV_MLP[index](z)
            u_z_temp = grad(u_temp.sum(), z, create_graph=True)[0]
            u_zz_temp = grad(u_z_temp.sum(), z, create_graph=True)[0]
            u+= index_pred[:,index:index+1] * u_temp
            u_z+= index_pred[:,index:index+1] * u_z_temp
            div_u += index_pred[:,index:index+1] * div_u_temp
            u_zz+= index_pred[:,index:index+1] * u_zz_temp
        return u, u_z+div_u, u_zz
