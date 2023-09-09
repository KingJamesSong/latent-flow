import torch
from torch import nn
import math
from torch.autograd import grad
from torch.distributions.normal import Normal
import numpy as np

class TruncNormal(object):
    def __init__(self, mu=0.0, sigma=1.0, a=-10.0, b=10.0):
        self.mu = mu
        self.sigma = sigma
        self.a = torch.Tensor([a])
        self.b = torch.Tensor([b])
        self.phi_b = 0.5 * (1 + torch.special.erf((self.b - mu) / sigma / np.sqrt(2)))
        self.phi_a = 0.5 * (1 + torch.special.erf((self.a - mu) / sigma / np.sqrt(2)))

    def inv_phi(self, x):
        x = 2 * x - 1
        return torch.special.erfinv(x) * np.sqrt(2)

    def phi(self, x):
        return 0.5 * (1 + torch.special.erf(x / np.sqrt(2)))

    def phi_x(self, x):
        # out-of-boundary check
        x = torch.where(x > self.b, x + 100, x)
        x = torch.where(x < self.a, x - 100, x)
        x = (x - self.mu) / self.sigma
        scaling = 1 / np.sqrt(2 * np.pi)
        exp_part = torch.exp(-0.5 * (x ** 2))
        output = scaling * exp_part
        # out-of-boundary check
        output[x > self.b] = 0.
        output[x < self.a] = 0.
        # non-slit check
        #for batch_index in range(x.size(0)):
        #    if x[batch_index, 0] > 1.5 and x[batch_index, 1] > -1 and x[batch_index, 1] < 1:
        #        output[batch_index] = 0.
        #    elif x[batch_index, 0] > -1.5 and x[batch_index, 1] > -1 and x[batch_index, 1] < 1:
        #        output[batch_index] = 0.
        #    elif x[batch_index, 0] > -1 and x[batch_index, 0] < 1 and x[batch_index, 1] > -1 and \
        #            x[batch_index, 1] < 1:
        #        output[batch_index] = 0.

        return output

    def prob(self, x):
        x_prob = self.phi_x(x)
        return x_prob / (self.phi_b - self.phi_a) + 1e-10

    def log_prob(self, x):
        x_prob = self.phi_x(x)
        prob = x_prob / (self.phi_b - self.phi_a)
        return (prob + 1e-10).log()

    def rsample(self, x):
        eps = torch.rand_like(x)
        phi_b = self.phi(self.b)
        phi_a = self.phi(self.a)
        inv_input = eps * (phi_b - phi_a) + phi_a
        inv = self.inv_phi(inv_input)
        inv = torch.where(inv > self.b, self.b, inv)
        inv = torch.where(inv < self.a, self.a, inv)
        return inv * self.sigma + self.mu

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings,self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = (self.dim) // 2
        embeddings = math.log(10000) / (half_dim)
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

class WasserteinFlow(nn.Module):
    def __init__(self, n_in,timestep,madelung_flow=False,group=3):
        super(WasserteinFlow,self).__init__()
        self.g = group
        self.mlp = nn.ModuleList([MLP(self.g, n_out=1) for i in range(n_in//self.g)])
        #self.mu = nn.Parameter(torch.zeros(timestep+1, n_in, requires_grad=True))
        #self.sigma = nn.Parameter(torch.ones(timestep+1, n_in, requires_grad=True))
        #self.V =  nn.ModuleList([MLP_Diff(2, n_out=1) for i in range(n_in//2)])
        self.madelung_flow = madelung_flow
        self.set_num = n_in // self.g
        self.beta = nn.Parameter(torch.ones(self.set_num, 1, requires_grad=True))
        self.coupling = nn.Parameter(torch.eye(self.set_num, requires_grad=True))
        #self.slack = nn.Parameter(torch.zeros(1, n_in, requires_grad=True))
        #Assume rho follows gaussian distribution

    def forward(self, z, time_step, mu, sigma):
        t = time_step * torch.ones(1, 1, requires_grad=True)
        prob = TruncNormal(mu, sigma)
        nabla_log_phi = - (z - mu) / (sigma ** 2 + 1e-5)
        delta_log_phi = - 1.0 / (sigma ** 2 + 1e-5 )
        delta_phi_over_phi =((z - mu)**2 - sigma ** 2 )/(sigma ** 4 + 1e-5)
        rho = prob.prob(z).squeeze(1)
        for i in range(self.set_num):
            z_temp = z[:,i*self.g:i*self.g+self.g]
            rho_temp = rho[:,i*self.g:i*self.g+self.g]
            S = self.mlp[i](z_temp, t)
            u_temp = S - self.beta[i] * rho_temp.prod(dim=1, keepdim=True).log()
            s_z = grad(S.sum(), z_temp, create_graph=True)[0]
            u_z_temp = (s_z - self.beta[i]*nabla_log_phi[:,i*self.g:i*self.g+self.g])
            #u_t = grad((S - self.beta[i] * rho_temp.prod(dim=1, keepdim=True).log()).sum(), t, create_graph=True)[0]
            u_t = grad(S.sum(), t, create_graph=True)[0] + self.beta[i]*(grad((rho_temp*u_z_temp).sum(), z_temp, create_graph=True)[0]) / (rho_temp+1e-5)
            u_zz_temp = (grad(s_z.sum(), z_temp, create_graph=True)[0] - self.beta[i]*delta_log_phi[:,i*self.g:i*self.g+self.g])
            if self.madelung_flow:
                #v_temp = self.V[i](z_temp)
                z_tmp_max, _ = torch.max(z_temp, dim=1, keepdim=True)
                z_tmp_min, _ = torch.min(z_temp, dim=1, keepdim=True)
                #v_temp = torch.zeros_like(z_tmp_max)
                v_temp = z_temp.unsqueeze(1).bmm(self.coupling.unsqueeze(0)).bmm(z[:,(i+1)*self.g:(i+1)*self.g+self.g].unsqueeze(2))
                v_temp = v_temp.squeeze()
                #Double slits experiments
                v_temp[z_tmp_max> 10.] = 100.
                v_temp[z_tmp_min<-10.] = 100.
                pde_temp = u_t + 0.5 * (u_z_temp ** 2) -  0.125 * (self.beta[i]**2)* ((nabla_log_phi[:,i*self.g:i*self.g+self.g] ** 2) - 2 * delta_phi_over_phi[:,i*self.g:i*self.g+self.g]) + v_temp
                #v_target_tmp = torch.zeros_like(v_temp)
                #v_target_tmp[z_tmp_max>5.]  = 100.
                #v_target_tmp[z_tmp_min<-5.] = 100.
            else:
                pde_temp = u_t + 0.5 * (u_z_temp ** 2) + 0.125 * (self.beta[i]**2)*((nabla_log_phi[:,i*self.g:i*self.g+self.g] ** 2) - 2 * delta_phi_over_phi[:,i*self.g:i*self.g+self.g])
            if i==0:
                pde,u_z,u,u_zz = pde_temp,u_z_temp,u_temp,u_zz_temp
                #if self.madelung_flow:
                #    v,v_target = v_temp,v_target_tmp
            else:
                pde = torch.cat([pde, pde_temp],dim=1)
                u_z = torch.cat([u_z, u_z_temp], dim=1)
                u = torch.cat([u, u_temp], dim=1)
                u_zz = torch.cat([u_zz, u_zz_temp], dim=1)
                #if self.madelung_flow:
                #    v = torch.cat([v, v_temp], dim=1)
                #    v_target = torch.cat([v_target, v_target_tmp], dim=1)
        #if self.madelung_flow:
        #    bound = v - v_target
        #    pde = torch.cat([pde, bound], dim=1)

        #mse_pde = torch.mean(torch.square(pde)) #+ torch.mean(torch.square(madelung_flow-self.slack**2))
        return pde, u_z, u, u_zz

    def energy_density(self,z,time_step,mu,sigma):
        t = time_step * torch.ones(1, 1, requires_grad=True)
        prob = TruncNormal(mu,sigma)
        rho = prob.prob(z).squeeze(1)
        for i in range(self.set_num):
            z_temp = z[:, i * self.g:i * self.g + self.g]
            S = self.mlp[i](z_temp, t)
            rho_temp = rho[:, i * self.g:i * self.g + self.g]
            u_temp = S - self.beta[i] * rho_temp.prod(dim=1, keepdim=True).log()
            if i == 0:
                u = u_temp
            else:
                u = torch.cat([u, u_temp], dim=1)
        return u, rho


    #def boundary_loss(self,z,time_step,mu,sigma):
    #    t = time_step * torch.ones(1, 1, requires_grad=True)
    #    bound1 =  torch.abs(torch.rand_like(z, requires_grad=True))*5 + 5
    #    bound2 = -torch.abs(torch.rand_like(z, requires_grad=True))*5 - 5
        #u1_z = torch.zeros_like(z)
        #u2_z = torch.zeros_like(z)
    #    nabla_log_phi1 = - (bound1 - mu) / (sigma ** 2 + 1e-5)
    #    nabla_log_phi2 = - (bound2 - mu) / (sigma ** 2 + 1e-5)
    #    for i in range(self.set_num):
    #        bound1_temp = bound1[:,i*self.g:i*self.g+self.g]
    #        S1 = self.mlp[i](bound1_temp, t)
    #        S1_z = grad(S1.sum(), bound1_temp, create_graph=True)[0]
    #        u1_z_temp = (S1_z - self.beta[i] * nabla_log_phi1[:,i*self.g:i*self.g+self.g]).squeeze(0)
    #        bound2_temp = bound2[:, i * self.g:i * self.g + self.g]
    #        S2 = self.mlp[i](bound2_temp, t)
    #        S2_z = grad(S2.sum(), bound2_temp, create_graph=True)[0]
    #        u2_z_temp = (S2_z - self.beta[i] * nabla_log_phi2[:,i*self.g:i*self.g+self.g]).squeeze(0)
    #        if i==0:
    #            u1_z, u2_z = u1_z_temp,u2_z_temp
    #        else:
    #            u1_z = torch.cat([u1_z, u1_z_temp], dim=1)
    #            u2_z = torch.cat([u2_z, u2_z_temp], dim=1)

    #    return torch.mean(torch.square(u1_z)) + torch.mean(torch.square(u2_z))

    #def forward(self, z, time_step):
        #print(time_step)
        #if time_step == 1:
        #    posterior = Normal(0.0, 1.0)
        #else:
        #    #print(self.mu[time_step],self.sigma[time_step])
        #    posterior = Normal(self.mu[time_step], self.sigma[time_step] ** 2)
        #log_rho = posterior.log_prob(z)
    #    if time_step == 1:
    #        nabla_phi = self.D * z
    #        nabla_nabla_phi = self.D * 1.0
    #    else:
    #        nabla_phi =  self.D * (z-self.mu[time_step-1]) / (self.sigma[time_step-1]**2 + 1e-5)
    #        nabla_nabla_phi = self.D * 1.0 / (self.sigma[time_step-1] ** 2 + 1e-5 )
    #    A = self.mlp(z,time_step*torch.ones(1,1,requires_grad=True))
    #    potential = A.sum()
    #    velocity = (grad(potential, z, create_graph=True)[0] + nabla_phi).squeeze()
    #    nabla_velocity = (grad(velocity.sum(), z, create_graph=True)[0] + nabla_nabla_phi).squeeze()

    #    return velocity, potential, nabla_velocity


class WaFlow(nn.Module):
    def __init__(self, num_support_sets, num_support_dipoles, support_vectors_dim,
                 learn_alphas=False, learn_gammas=False, gamma=None,img_size=28,madelung_flow=False,group=3):

        super(WaFlow, self).__init__()
        self.num_support_sets = num_support_sets
        self.num_support_dipoles = num_support_dipoles
        self.support_vectors_dim = support_vectors_dim
        self.MLP_SET= nn.ModuleList([WasserteinFlow(n_in=support_vectors_dim,timestep=num_support_dipoles//2,madelung_flow=madelung_flow,group=group) for i in range(num_support_sets)])


    def index_forward(self, index_pred, z, t,mu, sigma ):
        mse_pde_t_index, u_z, u, u_zz = 0.0,0.0,0.0,0.0
        t = t.long()
        for index in range(self.num_support_sets):
            pde_temp, u_z_temp, u_temp, u_zz_temp = self.MLP_SET[index](z, t,mu, sigma)
            #u_zt_temp = self.MLP_SET[index].mlp(zt, (t+1)*torch.ones(1, 1, requires_grad=True))
            #rho_z_temp = self.MLP_SET[index].density(z, t)
            #rho_zt_temp = self.MLP_SET[index].density(zt, t + 1)
            mse_pde_t_index_temp = torch.mean(torch.square(pde_temp))
            #mse_pde_t_index_temp =  torch.mean(torch.square(rho_z_temp*(u_z_temp + pde_temp) - rho_zt_temp*u_zt_temp))
            mse_pde_t_index += index_pred[:, index:index + 1].mean() * mse_pde_t_index_temp
            u_z += index_pred[:,index:index+1] * u_z_temp
            u += index_pred[:,index:index+1] * u_temp
            u_zz += index_pred[:,index:index+1] * u_zz_temp
        return u, mse_pde_t_index, u_z, u_zz

    def forward(self, index, z, t, mu, sigma):
        t = t.long()
        pde, u_z, u, u_zz = self.MLP_SET[index](z, t, mu, sigma)
        #s_zt = self.MLP_SET[index].mlp(zt,(t+1)*torch.ones(1, 1, requires_grad=True))
        #rho_z = self.MLP_SET[index].density(z,t)
        #rho_zt = self.MLP_SET[index].density(zt, t+1)
        #u_zt = s_zt - rho_zt.log()
        mse_pde = torch.mean(torch.square(pde)) #+ self.MLP_SET[index].boundary_loss(z,t, mu, sigma)
        #mse_pde  =  torch.mean(torch.square( rho_z*(u+pde) - rho_zt*u_zt ))
        return u, mse_pde, u_z, u_zz

    def inference(self, index, z, t, mu, sigma):
        t = t.long()
        z = z.clone().requires_grad_()
        #t = t.clone().requires_grad_()
        _, u_z, u, u_zz = self.MLP_SET[index](z, t, mu, sigma)
        prob = TruncNormal(mu, sigma)
        rho =prob.prob(z).squeeze(1)
        #u_z = grad(u.sum(), z, create_graph=True)[0]
        #u_zz = grad(u_z.sum(), z, create_graph=True)[0]

        return u, u_z, rho

    def inference_with_rho(self, index, z, t, mu, sigma):
        t = t.long()
        #z = z.clone().requires_grad_()
        #t = t.clone().requires_grad_()
        #_, u_z, u, u_zz = self.MLP_SET[index](z, t, mu, sigma)
        u, rho = self.MLP_SET[index].energy_density(z, t, mu, sigma)
        #u_z = grad(u.sum(), z, create_graph=True)[0]
        #u_zz = grad(u_z.sum(), z, create_graph=True)[0]
        #S = self.MLP_SET[index].mlp(z, t*torch.ones(1, 1, requires_grad=True))
        #if t == 1:
        #  prob = TruncNormal(0.0,1.0)
        #else:
        #  prob = TruncNormal(mu, sigma)
        #rho =prob.prob(z).squeeze(1)
        #u = S - self.MLP_SET[index].beta * rho.prod(dim=1, keepdim=True).log()

        #v = self.MLP_SET[index].V(z,t)
        return u, rho.sqrt()

    def index_inference(self, index_pred, z, t):
        t = t.long()
        z = z.clone().requires_grad_()
        #t = t.clone().requires_grad_()
        u, u_z, u_zz= 0.0, 0.0, 0.0
        for index in range(self.num_support_sets):
            mse_pde_t_index_temp, u_z_temp, u_temp, u_zz_temp = self.MLP_SET[index](z, t)
            u+= index_pred[:,index:index+1] * u_temp
            u_z+= index_pred[:,index:index+1] * u_z_temp
            u_zz+= index_pred[:,index:index+1] * u_zz_temp
        return u, u_z, u_zz
