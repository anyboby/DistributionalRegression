from numpy.random.mtrand import sample
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from torch.autograd import Variable
import math 
from scipy.misc import derivative
from scipy.signal import savgol_filter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.device(0)
print(f'Using {device} device')

def init_weights_xav(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight, gain=1)
        m.bias.data.fill_(0.0)


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).float()

        if not v.is_cuda and cuda:
            v = v.cuda()
        
        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)
        
        out.append(v)
    return out

def mse_loss(output, target):
    mse = (output - target)**2
    return mse.mean()

def normal(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2)) * np.exp(- (x-mean)**2 / sigma**2)

class Net(nn.Module):
    def __init__(self, n_in, n_outs, n_hidden, n_layers, lr, weight_decay=0, loss_type='quantile', v_min = -10, v_max = 10):
        super(Net, self).__init__()
    
        self.n_outs = n_outs

        self.layers = nn.ModuleList([nn.Linear(n_in, n_hidden)])
        self.layers.extend([nn.Linear(n_hidden, n_hidden) for i in range(1, n_layers)])

        self.loss_type = loss_type

        if loss_type == "quantile":
            # quantile regression loss type
            self.loss = self.quant_loss

            #define quantiles
            # taus = np.arange(2 * n_outs + 1) / (2 * n_outs)
            # self.taus, = to_variable(var=(taus,), cuda=True)
            # self.n_taus = len(self.taus)

            taus = np.arange(n_outs) / (n_outs-1)
            self.taus, = to_variable(var=(taus,), cuda=True)
            self.layers.append(nn.Linear(n_hidden, self.n_outs))

        elif loss_type == "projection":
            self.loss = self.proj_loss #projection_loss
            self.v_min, self.v_max = v_min, v_max
            self.zs = torch.linspace(v_min, v_max, n_outs).cuda()
            self.delta_z = (self.zs[1] - self.zs[0]).cuda()
            
            self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
            self.softmax = nn.Softmax(dim=1)
            
            # simply for smoothing predictions a bit
            self.unif = D.uniform.Uniform(-self.delta_z , self.delta_z)
            self.layers.append(nn.Linear(n_hidden, self.n_outs))


        elif loss_type == "evidential":
            # quantile regression loss
            self.loss = self.evidential_loss
            taus = np.arange(n_outs) / (n_outs-1)
            self.taus, = to_variable(var=(taus,), cuda=True)
            self.mus = nn.Linear(n_hidden, self.n_outs)
            self.logvars = nn.Linear(n_hidden, self.n_outs)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.activation = F.silu

    def forward(self, x):
        for l in range(len(self.layers)-1):
            x = self.layers[l](x)
            x = self.activation(x)
        
        x=self.layers[-1](x)

        if self.loss_type == "evidential":
            x = self.activation(x)
            mu = self.mus(x)
            logvar = self.logvars(x)
            x = self.reparametrize(mu, logvar), mu, logvar

        return x
    
    def forward_sample(self, x, size=1, include_latent = False):
        x = self.forward(x)
    
        if self.loss_type == "quantile":
            quant_idx = torch.randint(low=0, high=self.n_outs, size=(x.shape[0],size))
            x_idx = torch.arange(0, len(x))[...,None]

            samples = x[x_idx, quant_idx]

        elif self.loss_type == "projection":
            probs = self.softmax(x)
            dists = D.categorical.Categorical(probs)
            idx_samples = dists.sample((size,)).swapaxes(0,1)

            unif_samples = self.unif.sample((x.shape[0],size))

            samples = self.v_min + self.delta_z * idx_samples + unif_samples

        elif self.loss_type == "evidential":
            x, mu, logvar = x
            quant_idx = torch.randint(low=0, high=self.n_outs, size=(x.shape[0],size))
            x_idx = torch.arange(0, len(x))[...,None]

            samples = x[x_idx, quant_idx]

            if include_latent:
                samples = samples, mu, logvar.exp()

        return samples


    def fit(self, x, y):
        x, y = to_variable(var=(x,y), cuda=True)

        out = self.forward(x)

        #reset grad and loss
        self.optimizer.zero_grad()
        loss = self.loss(out, y)

        loss.backward()
        self.optimizer.step()

        return loss

    def hl(self, u, kappa):
        delta = (abs(u) <= kappa).float()
        hl = delta * (u * u / 2) + (1 - delta) * (
            kappa * (abs(u) - kappa / 2)
        )

    def rho_tau(self, u, tau, kappa=1):
        delta = (u < 0).float()
        if kappa == 0:
            return (tau - delta) * u
        else:
            return abs(tau - delta) * self.hl(u=u, kappa=kappa)

    def normal_elbo(self, mu, logvar):
        elbo = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return elbo

    def quant_loss(self, outs, targets):
        thetas = outs
        loss = torch.mean(
            self.rho_tau(targets - thetas, self.taus, kappa = 0.0) 
            )
        return loss

    def proj_loss(self, outs, targets):
        p_zs = outs
        targets = torch.clamp(1 - abs(targets - self.zs) / self.delta_z, 0 , 1)
        ce_loss = self.ce_loss(p_zs, targets)
        return ce_loss

    def evidential_loss(self, outs, targets):
        logits, mu, logvar = outs
        quant_loss = self.quant_loss(logits, targets)
        elbo_loss = self.normal_elbo(mu, logvar)

        return quant_loss + 0.05 * elbo_loss

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

### TRAINING DATA ###
### more complex mixture samples
train_size = 10000

# generate xs
x_clusters = torch.tensor([-3, 7]).float()
x_stds = torch.tensor([1.5,1.5]).float()
mix = D.Categorical(torch.ones(2,))
comp = D.Normal(x_clusters, x_stds)
gmm = MixtureSameFamily(mix, comp)

x_samples = gmm.sample(sample_shape=(train_size,1))

# model repose samples
# define mean and std functions
def f_mu_1(x, scale=1):
    res = scale * torch.sin(2 * x) * torch.cos(x / 2)
    return res
def f_sig_1(x, scale=1e-1):
    res = scale * (.5 * torch.sin(x) + 1)
    return res

def f_mu_2(x, scale=1):
    res = scale * torch.cos(x_samples) * torch.sin(x_samples / 2 + 2)
    return res
def f_sig_2(x, scale=1e-1):
    res = scale * (.5 * torch.cos(x) + 1)
    return res

## define mixture components. Stds in x-direct are kept small. 
mu_x1 = f_mu_1(x_samples, scale=1.5)
mu_x1 = torch.cat((x_samples, mu_x1), dim = -1)
sig_x1 = f_sig_1(x_samples, scale=2.5e-1)
sig_x1 = torch.cat((1e-5 * torch.ones_like(x_samples), sig_x1), dim = -1)

mu_x2 = f_mu_2(x_samples, scale=1.5)
mu_x2 = torch.cat((x_samples, mu_x2), dim = -1)
sig_x2 = f_sig_2(x_samples, scale=1.5e-1)
sig_x2 = torch.cat((1e-5 * torch.ones_like(x_samples), sig_x2), dim = -1)

mu_mix = torch.cat((mu_x1, mu_x2), dim = 0)
sig_mix = torch.cat((sig_x1, sig_x2), dim = 0)

mix_y = D.Categorical(torch.ones(2 * train_size,))
comps_y = D.Independent(D.Normal(
    loc = mu_mix, scale = sig_mix), 1)
gmm_y = MixtureSameFamily(mix_y, comps_y)

samples = gmm_y.sample(sample_shape=(train_size,))

### MODEL
torch.manual_seed(1)

n_outs = 50
models = []

# proj_net = Net(n_in=1, n_outs=n_outs, n_hidden=200, n_layers=3, lr=5e-3, weight_decay=1e-7, loss_type='projection', v_min=-4.0, v_max=4.0).cuda()
# proj_net.apply(init_weights_xav)

# models.append(proj_net)

# qreg_net = Net(n_in=1, n_outs= n_outs, n_hidden=200, n_layers=3, lr=1e-3, weight_decay=1e-7,).cuda()
# qreg_net.apply(init_weights_xav)

# models.append(qreg_net)

evid_net = Net(n_in=1, n_outs= n_outs, n_hidden=200, n_layers=3, lr=1e-3, weight_decay=1e-7, loss_type="evidential").cuda()
evid_net.apply(init_weights_xav)

models.append(evid_net)

bs = 64
epochs = 3000

for i in range(epochs):
    sub_idx = np.random.choice(np.arange(0, train_size), size=bs, replace=True)

    x_train, y_train = samples[sub_idx,0:1],samples[sub_idx,1:2]
    # y_train = x_n[sub_idx]
    # x_train = np.zeros_like(y_train)

    losses = [m.fit(x_train, y_train) for m in models]


    if i % 200 == 0:
        print(i, [l.cpu().data.numpy() for l in losses])
        #print('Epoch %4d, Train loss projection = %6.3f, loss quantile = %6.3f, loss evidential = %6.3f' % \
        #    (i, proj_loss.cpu().data.numpy(), qreg_loss.cpu().data.numpy(), evid_loss.cpu().data.numpy())
        #    )


### PREDICTION / EVALUATION
# create test values of x
x_test = torch.linspace(-25, 25, steps=1000)[...,None].float().cuda()

# predictions are inverse of the CDF
# quants_pred = qreg_loss(x_test).cpu().detach().numpy()[0]
# quants_cdf = np.linspace(0, 1, 2 * n_q + 1)

# draw samples from CDF by drawing from CDF^-1(u), u ~ U[0,1]
sample_size = 100
x_samples = x_test.repeat((1,sample_size)).flatten().cpu().detach().numpy()
y_preds = [m.forward_sample(x_test, size=sample_size).flatten().cpu().detach().numpy() for m in models]


# epistemic uncertainty
_, _, quant_var  = models[0].forward_sample(x_test, size=sample_size, include_latent=True)
quant_var = torch.mean(quant_var, dim= -1).cpu().detach().numpy()

#y_preds_qreg = qreg_net.forward_sample(x_test, size=sample_size).flatten().cpu().detach().numpy()
#y_preds_proj = proj_net.forward_sample(x_test, size=sample_size).flatten().cpu().detach().numpy()

# smoothen to get a pdf, but this doesn't really work atm and is a bit unnecessary
# quants_pred_smooth = savgol_filter(quants_pred, 51, 3)
# quants_pdf = np.diff(quants_cdf) / np.diff(quants_pred_smooth)

plt.figure(figsize = (6, 5))
plt.style.use('default')

samples_np = samples.cpu().detach().numpy()
plt.scatter(samples_np[:,0], samples_np[:,1], s = 10, marker = 'x', color = 'green', alpha = 0.5, label='data')
for y_pred in y_preds:
    plt.scatter(x_samples, y_pred, s=2, alpha=.2,)


# epistemic ucnertainty
plt.plot(x_test.cpu().detach().numpy(), quant_var, color='red')

# plt.plot(np.linspace(xmin, xmax, n_testsamples))
# plt.fill_between(np.linspace(xmin, xmax, n_testsamples), means + aleatoric, means + total_unc, color = c[0], alpha = 0.3, label = r"$\sigma(y'|x')$")
# plt.fill_between(np.linspace(xmin, xmax, n_testsamples), means - total_unc, means - aleatoric, color = c[0], alpha = 0.3)
# plt.fill_between(np.linspace(xmin, xmax, n_testsamples), means - aleatoric, means + aleatoric, color = c[1], alpha = 0.4, label = r'$\mathbb{E}_{q(\mathbf{w})}[\sigma^2]^{1/2}$')
# plt.plot(np.linspace(xmin, xmax, n_testsamples), means, color = 'black', linewidth = 1, label='means')

plt.xlim([-25, 25])
plt.ylim([-3, 6])

# plt.xlabel('$x$', fontsize=5)
# plt.title('map ensemble', fontsize=40)
# plt.tick_params(labelsize=5)
# plt.xticks(np.arange(xmin, xmax, 2))
# plt.yticks(np.arange(-4, 7, 2))
# plt.gca().yaxis.grid(alpha=0.3)
# plt.gca().xaxis.grid(alpha=0.3)
# plt.savefig(f'map_ensemble{i}.pdf', bbox_inches = 'tight')

plt.show()