import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def torch_binom(n, k):
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sigmoid_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.sigmoid(y / temperature)

#Gumbel-Sigmoid trick Allows for generating binary spike variables
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

#Infer spike variable from a pair of images
class ConvEncoder3_Unsuper(nn.Module):

    def __init__(self, s_dim, n_cin, n_hw, latent_size):
        super().__init__()

        self.s_dim = s_dim
        self.latent_size = latent_size #number of motions
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
        self.linear_y = nn.Linear(s_dim, self.latent_size)

    def forward(self, input, iter):
        if iter % 100 == 1:
            self.temp = np.maximum(self.temp_ini * np.exp(-self.ANNEAL_RATE * iter), self.temp_min)
        x = self.encoder(input)
        x = x.view(-1, self.s_dim)
        z_temp1 = self.linear_y(x)
        z1 = gumbel_sigmoid(z_temp1, temperature=self.temp, categorical_dim=self.latent_size, hard=False)
        return z1

#A basic VAE, please specify the encoder and decoder
class VAE(nn.Module):

    def __init__(self, num_channel, latent_size, img_size):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = None
        self.decoder = None

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

#BCE Reconstruction + KL Div
def vae_loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(x.size(0), -1), x.view(x.size(0), -1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

#BCE Reconstruction
def bce(recon_x, x):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x.view(x.size(0), -1),x.view(x.size(0), -1), reduction='sum')
    return BCE / x.size(0)

#Define 3 latent flow fields
num_vector = 3
latent_vector = nn.ModuleList([MLP for i in range(num_vector)])#Please define your MLP
reconstructor = ConvEncoder3_Unsuper(s_dim=64,n_cin=3*2,n_hw=128,latent_size=3)#3motions inferred from 2 neignboring images
generator = VAE(num_channel=3,latent_size=128,img_size=128)
#KL Loss for spike variable
kl_index = nn.KLDivLoss(reduction="batchmean", log_target=False)

def training_function(data_loader,optimizer, generator, reconstructor, latent_vector,num_steps,num_vector):
    iteration = 0
    for i, (data, index) in enumerate(data_loader):
        iteration = iteration + 1
        # Set gradients to zero
        optimizer.zero_grad()
        #Reconstruction of X0
        x = data[0]
        recon_x, mean, log_var, z = generator(x)
        vae_loss = vae_loss_fn(recon_x, x, mean, log_var)
        #assume a 3-dimensional Bernoulli variable with a initial switch-on probability of 0.3
        #each logit has the probability of 0.7 to keep the value, and has the probability of 0.3 to switch to the other state
        rej_prob = 1. / 3. + 2 * 0.3 * 0.7 * 1. / 3. + 0.09 * 1. / 3
        intial_prob = rej_prob
        target_porb = 0.0
        x_t1 = x
        for t in range(1, num_steps + 1):
            x_t = x_t1.clone()
            x_t1 = data[t]
            x_pair = torch.cat([x_t, x_t1], dim=1)
            #infer the spike variable
            y = reconstructor(x_pair, iteration)
            uz = 0.0
            for index in range(num_vector):
                uz += y[:, index:index + 1] * latent_vector[index]
            z = z + uz
            img_shifted = generator.inference(z)
            #Reconstruction of xt
            vae_loss += bce(img_shifted, x_t1)

            if t == 1:
                y_set = y
            else:
                y_set = torch.cat([y_set, y], dim=0)
            #update the expectation of the spike variable with rejection sampling
            target_porb = target_porb + intial_prob
            #The term after torch_binom denotes the expectation if all logits turn zero
            intial_prob = (intial_prob * 0.7 + (1 - intial_prob) * 0.3) + torch_binom(torch.FloatTensor([3.]).to(z),torch.FloatTensor([intial_prob * 3]).to(z)) * \
                          (0.3 ** (intial_prob * 3)) * (0.7 ** (3 - intial_prob * 3)) * (1. / 3. + 2 * 0.3 * 0.7 * 1. / 3. + 0.09 * 1. / 3.)
        #calculate the KL of the spike variable
        prob_one = torch.norm(y_set, p=1) / y_set.size(0) / y_set.size(1)
        prob_k = torch.Tensor([prob_one, 1.0 - prob_one]).to(z)
        target_porb = target_porb / num_steps
        target_k = torch.Tensor([target_porb, 1.0 - target_porb]).to(z)

        loss = vae_loss + kl_index((prob_k + 1e-20).log(), target_k)

        loss.backward()
        optimizer.step()