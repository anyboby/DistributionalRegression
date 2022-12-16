import math
from cProfile import label
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random.mtrand import sample
from scipy.misc import derivative
from scipy.signal import savgol_filter
from torch import distributions as D
from torch import logit
from torch.autograd import Variable
from torch.distributions.mixture_same_family import MixtureSameFamily

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.device(0)
print(f'Using {device} device')

def init_weights_xav(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight, gain=1)
        # m.bias.data.fill_(0.0)


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
    def __init__(self, n_in, n_outs, n_hidden, n_layers, lr, weight_decay=0, loss_type='quantile', v_min = -10, v_max = 10, last_layer_rbf=False):
        super(Net, self).__init__()
    
        self.n_outs = n_outs

        self.layers = nn.ModuleList([nn.Linear(n_in, n_hidden)])
        self.layers.extend([nn.Linear(n_hidden, n_hidden) for i in range(1, n_layers)])

        self.loss_type = loss_type
        self.last_layer_rbf = last_layer_rbf

        if loss_type == "quantile":
            # quantile regression loss type
            self.loss = self.quant_loss

            taus = (np.arange(n_outs)+0.5)/n_outs
            self.taus, = to_variable(var=(taus,), cuda=True)
            self.layers.append(nn.Linear(n_hidden, self.n_outs))
        
        if loss_type == "implicit_quantile":
            # quantile regression loss type
            self.loss = self.impl_quant_loss

            taus = (np.arange(n_outs)+0.5)/n_outs
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
            self.counts = nn.Linear(n_hidden, self.n_outs)
            
            def count_act(x):
                return torch.exp(-x**2)-.5            
            
            #Bias-less scaling layer
            self.counts_scale = nn.Linear(self.n_outs, self.n_outs, bias=False)
            self.sp_fn = nn.Softplus()
            # self.counts_scale = nn.Parameter(torch.ones(self.n_outs, requires_grad=True).cuda())
            
            if last_layer_rbf:
                #RBF activation before last layer
                self.counts_activation = count_act
            else:
                self.counts_activation = F.relu

        elif loss_type == "mse":
            self.loss = mse_loss
            self.layers.append(nn.Linear(n_hidden, self.n_outs))
            self.zs
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.activation = F.relu
        
        def weirdact(x):
            return x/(torch.abs(x)+0.01) * (1+x)
        def weirdact2(x):
            return torch.sign(x)
        # self.activation = F.sigmoid
        self.activation = weirdact2
        # self.activation = torch.sin

    def forward(self, x, reparametrize = True):
        for l in range(len(self.layers)-1):
            x = self.layers[l](x)
            x = self.activation(x)
        
        x=self.layers[-1](x)

        if self.loss_type == "evidential":
            x = self.activation(x)
            mu = self.mus(x)
            counts = self.counts(x)
            counts = self.counts_activation(counts)
            if self.last_layer_rbf:
                counts =  self.counts_scale(counts)
            # counts = self.counts_scale_act(counts)
            # counts = torch.exp(self.counts_scale) * counts
            if reparametrize:
                x = self.reparametrize(mu, counts), mu, counts
            else: 
                x = mu, mu, counts
        return x
    
    def forward_sample(self, x, size=1, include_latent = False, reparametrize=False, logits=False):
        x = self.forward(x, reparametrize=reparametrize)
    
        if self.loss_type == "quantile" or self.loss_type == "implicit_quantile":
            quant_idx = torch.randint(low=0, high=self.n_outs, size=(x.shape[0],size))
            x_idx = torch.arange(0, len(x))[...,None]
            if reparametrize:
                samples = x[x_idx, quant_idx]
            else:
                samples = x.mean(dim=-1, keepdim=True).expand((quant_idx.shape))

        elif self.loss_type == "projection":
            if logits:
                logits = x
                logit_idx = torch.randint(low=0, high=self.n_outs, size=(logits.shape[0],size))
                x_idx = torch.arange(0, len(logits))[...,None]
                if reparametrize:
                    samples = logits[x_idx, logit_idx]
                else:
                    samples = logits.mean(dim=-1, keepdim=True).expand((logit_idx.shape))

            else:
                probs = self.softmax(x)
                dists = D.categorical.Categorical(probs)
                idx_samples = dists.sample((size,)).swapaxes(0,1)

                unif_samples = self.unif.sample((x.shape[0],size))

                samples = self.v_min + self.delta_z * idx_samples + unif_samples

                if reparametrize:
                    samples = samples
                else:
                    samples = torch.sum(probs * self.zs.expand(probs.shape), dim=-1, keepdim=True)
                    samples = samples.expand((samples.shape[0], size))

        elif self.loss_type == "evidential":
            x, mu, counts = x
            quant_idx = torch.randint(low=0, high=self.n_outs, size=(x.shape[0],size))
            x_idx = torch.arange(0, len(x))[...,None]

            samples = x[x_idx, quant_idx]

            if include_latent:
                samples = samples, mu, torch.exp(counts)
        elif self.loss_type == "mse":
            samples = x.repeat((1, size))

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

    def normal_elbo(self, mu, counts, reduce=True):
        # elbo = -0.5 * torch.mean(5 + logcounts - mu.pow(2) - logcounts.exp())
        if reduce:
            # reg = torch.mean(torch.exp(counts))
            reg = torch.mean(counts)
        else:
            # reg = torch.exp(counts)
            reg = counts
        return reg

    def quant_loss(self, outs, targets, reduce=True):
        thetas = outs
        # std = torch.std(thetas, dim = -1)[...,None].detach()

        if reduce:
            loss = torch.mean(
                self.rho_tau(targets - thetas, self.taus, kappa = 0.0) #/ std
                )
        else:
            loss = self.rho_tau(targets - thetas, self.taus, kappa = 0.0) # / std
        
        return loss

    def impl_quant_loss(self, outs, targets, reduce=True):
        thetas = outs
        thetas = torch.sort(thetas, descending=False, dim=-1)[0]
        # std = torch.std(thetas, dim = -1)[...,None].detach()

        if reduce:
            loss = torch.mean(
                self.rho_tau(targets - thetas, self.taus, kappa = 0.0) #/ std
                )
        else:
            loss = self.rho_tau(targets - thetas, self.taus, kappa = 0.0) # / std
        
        # loss += -torch.clip(torch.mean(torch.sum(thetas * torch.log(self.taus+0.01), dim=-1)),0, 0.5)
        return loss


    def proj_loss(self, outs, targets):
        p_zs = outs
        targets = torch.clamp(1 - abs(targets - self.zs) / self.delta_z, 0 , 1)
        ce_loss = self.ce_loss(p_zs, targets)
        return ce_loss

    def evidential_loss(self, outs, targets):
        logits, mu, counts = outs
        # std = torch.std(mu, dim = -1)[...,None]
        quant_loss = self.quant_loss(logits, targets, reduce=False) # / std.detach()
        elbo_loss = self.normal_elbo(mu, counts, reduce=False)
        
        acc_loss_weight = torch.mean(quant_loss.detach())
        ev_loss_weight = torch.mean(elbo_loss.detach())
        cur_weight = ev_loss_weight / acc_loss_weight
        reweight = 1e-1 * cur_weight

        ev_loss = torch.mean(quant_loss) + 1e-3 * torch.mean(elbo_loss) #-1e-2 * torch.mean(elbo_loss)# + -1e-2 * torch.mean(elbo_loss)# + reweight * elbo_loss / std**2) #0 * 1e-2 * quant_loss * elbo_loss / std)
        return ev_loss

    def reparametrize(self, mu, counts):
        # std_hat = torch.std(mu, dim=-1)[...,None]
        std_pseudo = torch.exp(- .5 * counts) #* std_hat.detach()
        eps = torch.randn_like(std_pseudo)
        # mu = mu * (1+torch.randn_like(std))
        return mu + eps * std_pseudo

### TRAINING DATA ###
### more complex mixture samples
n_dim = 24
train_size = 250000

# generate xs
x_train = torch.normal(mean=torch.zeros(size=(train_size, n_dim)), std=torch.ones(size=(train_size, n_dim)))
x_train = x_train / torch.sqrt(torch.sum(x_train**2, dim=-1, keepdim=True))
y_train = torch.ones(size=(train_size, 1))

### MODEL
torch.manual_seed(1)

n_outs = 51
models = []
loss_type = "quantile"
ensemble_size = 100
for i in range(ensemble_size):
    net = Net(n_in=n_dim, n_outs= n_outs, n_hidden=512, n_layers=1, lr=5e-4, weight_decay=0, loss_type=loss_type, last_layer_rbf=False).cuda()
    net.apply(init_weights_xav)
    models.append(net)

bs = 128
epochs = 5000

for i in range(epochs):
    losses = []
    for m in range(len(models)):
        sub_idx = np.random.choice(np.arange(0, train_size), size=bs, replace=True)
        x_mb, y_mb = x_train[sub_idx],y_train[sub_idx]
        losses.append(models[m].fit(x_mb, y_mb))

    if i % 100 == 0:
        print(i, [l.cpu().data.numpy() for l in losses])

### evaluation: searching for ensemble agreement on a wider hypersphere
sweeps = 10000
eval_size = 128

test_r = 2.0 #radius of hypersphere

## get reference var
ref_var = 0
for j in range(train_size//bs):
    x_ref = x_train[j*bs:(j+1)*bs].to("cuda")
    ref_mean = torch.mean(torch.stack([models[i](x_ref) for i in range(len(models))]), dim=-1).cpu().detach()
    ref_var += torch.mean(torch.var(ref_mean, dim=0))
ref_var = ref_var/(train_size//bs)

threshold = ref_var.mean()

adv_vecs_2 = 0
adv_vecs_3 = 0
adv_vecs_4 = 0
adv_vecs_5 = 0
adv_vecs_7 = 0
adv_vecs_10 = 0
adv_vecs_15 = 0
adv_vecs_25 = 0
adv_vecs_33 = 0
adv_vecs_50 = 0
adv_vecs_75 = 0
adv_vecs_100 = 0

for j in range(sweeps):
    if j%(sweeps//3)==0:
        print(f"Running sweep{j}")

    # generate xs
    x = torch.normal(mean=torch.zeros(size=(eval_size, n_dim)), std=torch.ones(size=(eval_size, n_dim)))
    x = x / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True)) * test_r

    means = torch.stack([models[i](x.to("cuda")) for i in range(len(models))])
    means = torch.mean(means, dim=-1)
    
    ens_means_2 = means[0:2]
    ens_means_3 = means[0:3]
    ens_means_4 = means[0:4]
    ens_means_5 = means[0:5]
    ens_means_7 = means[0:7]
    ens_means_10 = means[0:10]
    ens_means_15 = means[0:15]
    ens_means_25 = means[0:25]
    ens_means_33 = means[0:33]
    ens_means_50 = means[0:50]
    ens_means_75 = means[0:75]
    ens_means_100 = means[0:100]

    ens_var_2 =   torch.var(ens_means_2, dim=0)
    ens_var_3 =   torch.var(ens_means_3, dim=0)
    ens_var_4 =   torch.var(ens_means_4, dim=0)
    ens_var_5 =   torch.var(ens_means_5, dim=0)
    ens_var_7 =   torch.var(ens_means_7, dim=0)
    ens_var_10 =  torch.var(ens_means_10, dim=0)
    ens_var_15 =  torch.var(ens_means_15, dim=0)
    ens_var_25 =  torch.var(ens_means_25, dim=0)
    ens_var_33 =  torch.var(ens_means_33, dim=0)
    ens_var_50 =  torch.var(ens_means_50, dim=0)
    ens_var_75 =  torch.var(ens_means_75, dim=0)
    ens_var_100 = torch.var(ens_means_100, dim=0)

    adv_vecs_2 += (ens_var_2<threshold).sum()
    adv_vecs_3 += (ens_var_3<threshold).sum()
    adv_vecs_4 += (ens_var_4<threshold).sum()
    adv_vecs_5 += (ens_var_5<threshold).sum()
    adv_vecs_7 += (ens_var_7<threshold).sum()
    adv_vecs_10 += (ens_var_10<threshold).sum()
    adv_vecs_15 += (ens_var_15<threshold).sum()
    adv_vecs_25 += (ens_var_25<threshold).sum()
    adv_vecs_33 += (ens_var_33<threshold).sum()
    adv_vecs_50 += (ens_var_50<threshold).sum()
    adv_vecs_75 += (ens_var_75<threshold).sum()
    adv_vecs_100 += (ens_var_100<threshold).sum()

print(f"Reference ens variance on training set ( {n_dim}-dimensional unit hypershell): {ref_var.mean()}")
print(f"Out of {sweeps*eval_size} evaluation points on hypersphere with radius {test_r}:")
print(f"2-sized ensemble predictions, with disagreement less than {threshold}:   {adv_vecs_2}")
print(f"3-sized ensemble predictions, with disagreement less than {threshold}:   {adv_vecs_3}")
print(f"4-sized ensemble predictions, with disagreement less than {threshold}:   {adv_vecs_4}")
print(f"5-sized ensemble predictions, with disagreement less than {threshold}:   {adv_vecs_5}")
print(f"7-sized ensemble predictions, with disagreement less than {threshold}:   {adv_vecs_7}")
print(f"10-sized ensemble predictions, with disagreement less than {threshold}:  {adv_vecs_10}")
print(f"15-sized ensemble predictions, with disagreement less than {threshold}:  {adv_vecs_15}")
print(f"25-sized ensemble predictions, with disagreement less than {threshold}:  {adv_vecs_25}")
print(f"33-sized ensemble predictions, with disagreement less than {threshold}:  {adv_vecs_33}")
print(f"50-sized ensemble predictions, with disagreement less than {threshold}:  {adv_vecs_50}")
print(f"75-sized ensemble predictions, with disagreement less than {threshold}:  {adv_vecs_75}")
print(f"100-sized ensemble predictions, with disagreement less than {threshold}: {adv_vecs_100}")
