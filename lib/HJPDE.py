import torch
from torch import nn
import math
from torch.autograd import grad

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

class MLP_Force(nn.Module):
    def __init__(self, n_in, n_out, img_size):
        super(MLP_Force,self).__init__()

        self.s_dim =64
        self.layer_x = nn.Sequential(
            nn.Conv2d(3, self.s_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.s_dim, self.s_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.s_dim * 2, self.s_dim * 3, kernel_size=(img_size // 4), stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(self.s_dim * 3, self.s_dim * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(self.s_dim * 2, self.s_dim, kernel_size=1, stride=1, padding=0),
        )
        self.layer_x_linear = nn.Linear(self.s_dim, n_out)

        self.layer_z = nn.Linear(n_in, n_in)
        self.activation1 = nn.Tanh()

        self.layer_pos = SinusoidalPositionEmbeddings(n_in)
        self.layer_time = nn.Linear(n_in, n_in)
        self.activation2 = nn.GELU()
        self.layer_time2 = nn.Linear(n_in, n_in)

        self.layer_fusion = nn.Linear(n_in, n_in)
        self.activation3 = nn.Tanh()
        self.layer_out = nn.Linear(n_in, n_out)
        self.activation4 = nn.Tanh()

    def forward(self, x, z, time):

        x = self.layer_x(x)
        x = x.view(-1, self.s_dim)
        x = self.layer_x_linear(x)

        z = self.layer_z(z)
        z = self.activation1(z)

        time = self.layer_pos(time)
        time = self.layer_time(time)
        time = self.activation2(time)
        time = self.layer_time2(time)

        x = self.layer_fusion(x+z+time)
        x = self.activation3(x)
        x = self.layer_out(x)
        x = self.activation4(x)
        return x


class HJPDE(nn.Module):
    def __init__(self, num_support_sets, num_support_dipoles, support_vectors_dim,
                 learn_alphas=False, learn_gammas=False, gamma=None,img_size=28):

        super(HJPDE, self).__init__()
        self.num_support_sets = num_support_sets
        self.num_support_dipoles = num_support_dipoles
        self.support_vectors_dim = support_vectors_dim

        self.MLP_SET= nn.ModuleList([MLP(n_in=support_vectors_dim,n_out=1) for i in range(num_support_sets)])
        self.FORCE_SET = nn.ModuleList([MLP(n_in=support_vectors_dim, n_out=1) for i in range(num_support_sets)])

    def loss_ic(self,mlp,z,force):
        z = z.clone().requires_grad_()
        #z.requires_grad = True
        u = mlp(z,torch.FloatTensor([0]).to(z))
        u_z = grad(u.sum(), z, create_graph=True)[0]

        f = force(z,torch.FloatTensor([0]).to(z))
        mse_0 =  torch.mean(torch.square(u_z)) + torch.mean(torch.square(f))

        return mse_0

    #Loss of PDE and Div
    def loss_pde(self,mlp,z,t,force):

        z = z.clone().requires_grad_()
        #z.requires_grad = True
        t = t.clone().requires_grad_()
        #t.requires_grad = True

        u = mlp(z, t)
        u_t = grad(u.sum(), t, create_graph=True)[0]

        u_z = grad(u.sum(), z, create_graph=True)[0]
        u_zz = grad(u_z.sum(), z, create_graph=True)[0]

        f = force(z,t)
        #PDE
        pde = u_t + 0.5 * torch.mean(u_z**2) + (f**2)
        mse_pde = torch.mean(torch.square(pde))
        return mse_pde, u_z, u, u_zz


    def index_forward(self, index_pred, z, t):
        mse_pde_t_index, u_z, u, u_zz = 0.0,0.0,0.0,0.0
        for index in range(self.num_support_sets):
            mse_pde_t_index_temp, u_z_temp, u_temp, u_zz_temp = self.loss_pde(self.MLP_SET[index], z, t, self.FORCE_SET[index])
            mse_pde_t_index += index_pred[:,index:index+1].mean() * mse_pde_t_index_temp
            u_z += index_pred[:,index:index+1] * u_z_temp
            u += index_pred[:,index:index+1] * u_temp
            u_zz += index_pred[:,index:index+1] * u_zz_temp
        loss_step = mse_pde_t_index
        if t[0] == 1:
            mse_ic = 0.0
            for index in range(self.num_support_sets):
                mse_ic_temp = self.loss_ic(self.MLP_SET[index], z, self.FORCE_SET[index])
                mse_ic = index_pred[:,index:index+1].mean() * mse_ic_temp
            loss_step += mse_ic
        return u, loss_step, u_z, u_zz

    #def forward(self, index, z, t, rho):
    #    for i in range(1,t[0].long()+1):
    #        t_index = i*torch.ones(1,1,requires_grad=True)
            #if i == 1:
            #    rho_z = grad(rho.exp().sum(), z, create_graph=True)[0]
            #    rho_zz = grad(rho_z.sum(), z, create_graph=True)[0]
    #        mse_pde_t_index, u_z , u, _ = self.loss_pde(self.MLP_SET[index],z,t_index,self.FORCE_SET[index],self.s[index])
            #print(i,grad(((rho_temp + 1).abs().log()).exp().sum(), (z), create_graph=True)[0])
            #print(i, rho_z)
    #        if i == t[0]:
    #            energy = u
    #            latent1 = z + u_z
    #            uzz = grad(u_z.sum(), z, create_graph=True)[0]
    #            loss_step = mse_pde_t_index
            #rho_temp = grad(u_z.sum(), z, create_graph=True)[0]
            #rho_z = grad(rho.exp().sum(), z, create_graph=True)[0]
            #print(i,rho_z)
            #rho = rho - (rho_temp + 1).abs().log()
            #rho_z = grad(rho.exp().sum(), z, create_graph=True)[0]
            #rho_zz = grad(rho_z.sum(), z, create_graph=True)[0]
    #        z = z + u_z
            #print(rho_t)
            #mse_pde = mse_pde + mse_pde_t_index
        #loss = mse_ic + mse_pde/half_range #- mse_jvp
    #    if t[0]==1:
    #        mse_ic = self.loss_ic(self.MLP_SET[index], z, self.FORCE_SET[index])
    #        loss_step += mse_ic
    #    latent2 = z
    #    return energy, latent1, latent2, loss_step, rho, uzz

    def forward(self, index, z, t):
        mse_pde_t_index, u_z , u, u_zz = self.loss_pde(self.MLP_SET[index],z,t,self.FORCE_SET[index])
        loss_step = mse_pde_t_index
        if t[0]==1:
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

    def index_inference(self, index_pred, z, t):
        z = z.clone().requires_grad_()
        t = t.clone().requires_grad_()
        u, u_z, u_zz=0.0,0.0,0.0
        for index in range(self.num_support_sets):
            u_temp = self.MLP_SET[index](z, t)
            u_z_temp = grad(u_temp.sum(), z, create_graph=True)[0]
            u_zz_temp = grad(u_z_temp.sum(), z, create_graph=True)[0]
            u+= index_pred[:,index:index+1] * u_temp
            u_z+= index_pred[:,index:index+1] * u_z_temp
            u_zz+= index_pred[:,index:index+1] * u_zz_temp
        return u, u_z, u_zz
