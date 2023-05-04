import math
from random import randint
from typing import List
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import rcParams
from numpy.random.mtrand import sample
from scipy.misc import derivative
from scipy.signal import savgol_filter
from torch import distributions as D
from torch.autograd import Variable
from torch.distributions.mixture_same_family import MixtureSameFamily
from scipy.interpolate import make_interp_spline, BSpline
import scipy.signal

device = 'cuda' if th.cuda.is_available() else 'cpu'
th.cuda.device(0)
print(f'Using {device} device')


live_draw = False

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",  # use serif/main font for text elements
    "font.size" : 6,
    # "font.family": "Times New Roman",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })

figsize = (2.0, 2.2)

colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0']
darkcolors = ['#0b7ad1', '#cc7a00', '#458c3f', '#d82411', '#7c1e92']

def init_weights_xav(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight, gain=1)
        # m.bias.data.fill_(0.0)

def softmax_t(input, t=1.0):
    ex = th.exp(input/t)
    sum = th.sum(ex, axis=-1, keepdim=True)
    return ex / sum

def cross_entropy_t(target, dist):
    ce = -th.sum(target * th.log(dist), dim=-1)
    return ce.mean()

def sigmoid_t(input, t=1.0):
    ex = th.exp(input/t)
    return ex / (ex+1)

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = th.from_numpy(v).float()

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
    def __init__(self, n_in, n_outs, n_hidden, n_layers, lr, weight_decay=0, loss_type='quantile', v_min = -10, v_max = 10, temp=1, last_layer_rbf=False):
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

        if loss_type == "expectile":
            # expectile regression loss type
            self.loss = self.exp_loss

            taus = (np.arange(n_outs)+0.5)/n_outs
            self.taus, = to_variable(var=(taus,), cuda=True)
            self.layers.append(nn.Linear(n_hidden, self.n_outs))

        elif loss_type == "categorical":
            self.loss = self.cat_loss #projected categorical loss
            self.v_min, self.v_max = v_min, v_max
            self.zs = th.linspace(v_min, v_max, n_outs).cuda()
            self.delta_z = (self.zs[1] - self.zs[0]).cuda()
            
            # self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
            self.ce_loss = cross_entropy_t
            # self.softmax = nn.Softmax(dim=1)
            self.softmax = softmax_t
            self.sm_temp = temp
            # simply for smoothing predictions a bit
            # self.unif = D.uniform.Uniform(-self.delta_z , self.delta_z)
            self.layers.append(nn.Linear(n_hidden, self.n_outs))
        
        elif loss_type == "binary":
            self.loss = self.bin_loss #projected categorical loss
            self.v_min, self.v_max = v_min, v_max
            self.zs = th.linspace(v_min, v_max, n_outs).cuda()
            self.delta_z = (self.zs[1] - self.zs[0]).cuda()
            
            self.sigmoid = sigmoid_t
            self.sm_temp = temp

            # simply for smoothing predictions a bit
            self.unif = D.uniform.Uniform(0 , 1)
            self.layers.append(nn.Linear(n_hidden, self.n_outs))

        elif loss_type == "evidential":
            # quantile regression loss
            self.loss = self.evidential_loss
            taus = np.arange(n_outs) / (n_outs-1)
            self.taus, = to_variable(var=(taus,), cuda=True)
            self.mus = nn.Linear(n_hidden, self.n_outs)
            self.counts = nn.Linear(n_hidden, self.n_outs)
            
            def count_act(x):
                return th.exp(-x**2)-.5            
            
            #Bias-less scaling layer
            self.counts_scale = nn.Linear(self.n_outs, self.n_outs, bias=False)
            self.sp_fn = nn.Softplus()
            # self.counts_scale = nn.Parameter(th.ones(self.n_outs, requires_grad=True).cuda())
            
            if last_layer_rbf:
                #RBF activation before last layer
                self.counts_activation = count_act
            else:
                self.counts_activation = F.relu

        elif loss_type == "mse":
            self.loss = mse_loss
            self.layers.append(nn.Linear(n_hidden, self.n_outs))
            self.zs
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # self.activation = F.relu
        self.activation = F.relu

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
            # counts = th.exp(self.counts_scale) * counts
            if reparametrize:
                x = self.reparametrize(mu, counts), mu, counts
            else: 
                x = mu, mu, counts
        return x
    
    def forward_sample(self, x, size=1, include_latent = False, reparametrize=False, sample_analytical = False):
        x = self.forward(x, reparametrize=reparametrize)
    
        if self.loss_type == "quantile" or self.loss_type == "implicit_quantile" or self.loss_type == "expectile":
            quant_idx = th.randint(low=0, high=self.n_outs, size=(x.shape[0],size))
            x_idx = th.arange(0, len(x))[...,None]
            if reparametrize:
                samples = x[x_idx, quant_idx]
                if sample_analytical:
                    tau = (th.arange(size)/size+1/(2*size)).to(x.device)
                    samples = th.quantile(x, tau, dim=1).swapaxes(0,1)
            else:
                samples = x.mean(dim=-1, keepdim=True)#.expand((quant_idx.shape))

        elif self.loss_type == "categorical":
            probs = self.softmax(x, t=self.sm_temp)

            if reparametrize:                
                if sample_analytical:                    
                    tau = (th.arange(size)/size+1/(2*size)).to(x.device)
                    cdf = th.cumsum(probs, dim=1)

                    tau_expanded = tau.unsqueeze(0).repeat(cdf.shape[0],1)
                    zs_exp = self.zs.unsqueeze(0).repeat(cdf.shape[0],1)
                    
                    pairw_delta = (cdf.unsqueeze(1)-tau_expanded.unsqueeze(2))
                    pairw_delta_pos = pairw_delta.clone()
                    pairw_delta_neg = pairw_delta.clone()
                    pairw_delta_pos[pairw_delta_pos<0] = 1e9
                    pairw_delta_neg[pairw_delta_neg>0] = -1e9

                    lev_u, ind_u = pairw_delta_pos.min(dim=2)
                    lev_l, ind_l = pairw_delta_neg.max(dim=2)
                    lev_l = th.abs(lev_l)

                    lev_u_norm = lev_u/(lev_l + lev_u)
                    lev_l_norm = lev_l/(lev_l + lev_u)
                    
                    z_l = th.gather(zs_exp, index=ind_l, dim=1)
                    z_u = th.gather(zs_exp, index=ind_u, dim=1)

                    z = lev_u_norm * z_l + lev_l_norm * z_u
                    
                    samples = z

            else:
                # samples
                samples = th.sum(probs * self.zs.expand(probs.shape), dim=-1, keepdim=True)
                # samples = samples.expand((samples.shape[0], size))

        elif self.loss_type == "binary":
            probs = self.sigmoid(x, t=self.sm_temp)

            unif_samples = self.unif.sample((x.shape[0],size)).to(device)
            inds = th.abs(unif_samples.unsqueeze(-1)-probs.unsqueeze(-2)).argmin(dim=-1)
            samples = self.v_min + self.delta_z * inds

            if reparametrize:
                samples = samples
            else:
                samples = samples
                samples = th.mean(samples, dim=-1, keepdim=True)
                samples = samples.expand((samples.shape[0], size))

        elif self.loss_type == "evidential":
            x, mu, counts = x
            quant_idx = th.randint(low=0, high=self.n_outs, size=(x.shape[0],size))
            x_idx = th.arange(0, len(x))[...,None]

            samples = x[x_idx, quant_idx]

            if include_latent:
                samples = samples, mu, th.exp(counts)
        elif self.loss_type == "mse":
            samples = x.repeat((1, size))

        return samples

    def forward_std(self, x, reparametrize):
        x = self.forward(x, reparametrize=reparametrize)
    
        if self.loss_type == "quantile" or self.loss_type == "implicit_quantile" or self.loss_type == "expectile":
            std = th.std(x, dim=1)
        elif self.loss_type == "categorical":
            probs = self.softmax(x, t=self.sm_temp)
            mean = th.sum(probs * self.zs.expand(probs.shape), dim=-1, keepdim=True)
            var = th.sum(probs * (self.zs.expand(probs.shape) - mean)**2, dim=1)
            std = th.sqrt(var)
        
        return std


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
        # elbo = -0.5 * th.mean(5 + logcounts - mu.pow(2) - logcounts.exp())
        if reduce:
            # reg = th.mean(th.exp(counts))
            reg = th.mean(counts)
        else:
            # reg = th.exp(counts)
            reg = counts
        return reg

    def quant_loss(self, outs, targets, reduce=True):
        thetas = outs
        # std = th.std(thetas, dim = -1)[...,None].detach()

        if reduce:
            loss = th.mean(
                self.rho_tau(targets - thetas, self.taus, kappa = 0.0) #/ std
                )
        else:
            loss = self.rho_tau(targets - thetas, self.taus, kappa = 0.0) # / std
        
        return loss

    def impl_quant_loss(self, outs, targets, reduce=True):
        thetas = outs
        thetas = th.sort(thetas, descending=False, dim=-1)[0]
        # std = th.std(thetas, dim = -1)[...,None].detach()

        if reduce:
            loss = th.mean(
                self.rho_tau(targets - thetas, self.taus, kappa = 0.0) #/ std
                )
        else:
            loss = self.rho_tau(targets - thetas, self.taus, kappa = 0.0) # / std
        
        # loss += -th.clip(th.mean(th.sum(thetas * th.log(self.taus+0.01), dim=-1)),0, 0.5)
        return loss

    def exp_loss(self, outs, targets):
        thetas = outs
        # std = th.std(thetas, dim = -1)[...,None].detach()

        err = (targets-thetas)
        sq_error = err**2
        weight = (err<0).float()
        exp_loss = th.abs((self.taus - weight) * sq_error)

        return exp_loss.mean()

    def cat_loss(self, outs, targets):
        p_zs = outs
        targets = th.clamp(1 - abs(targets - self.zs) / self.delta_z, 0 , 1)
        ce_loss = self.ce_loss(targets, self.softmax(p_zs, self.sm_temp))
        return ce_loss

    def bin_loss(self, outs, targets):
        p_zs = self.sigmoid(outs, t=self.sm_temp)
        # targets = th.clamp(1 - abs(targets - self.zs) / self.delta_z, 0 , 1)
        targets = (targets<self.zs).float()

        bin_loss = mse_loss(p_zs, targets)
        return th.sum(bin_loss)

    def evidential_loss(self, outs, targets):
        logits, mu, counts = outs
        # std = th.std(mu, dim = -1)[...,None]
        quant_loss = self.quant_loss(logits, targets, reduce=False) # / std.detach()
        elbo_loss = self.normal_elbo(mu, counts, reduce=False)
        
        acc_loss_weight = th.mean(quant_loss.detach())
        ev_loss_weight = th.mean(elbo_loss.detach())
        cur_weight = ev_loss_weight / acc_loss_weight
        reweight = 1e-1 * cur_weight

        ev_loss = th.mean(quant_loss) + 1e-3 * th.mean(elbo_loss) #-1e-2 * th.mean(elbo_loss)# + -1e-2 * th.mean(elbo_loss)# + reweight * elbo_loss / std**2) #0 * 1e-2 * quant_loss * elbo_loss / std)
        return ev_loss

    def reparametrize(self, mu, counts):
        # std_hat = th.std(mu, dim=-1)[...,None]
        std_pseudo = th.exp(- .5 * counts) #* std_hat.detach()
        eps = th.randn_like(std_pseudo)
        # mu = mu * (1+th.randn_like(std))
        return mu + eps * std_pseudo

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

### TRAINING DATA ###
### more complex mixture samples
train_size = 100

# generate xs
# x_clusters = th.tensor([-6, 0.2, 7]).float()
x_clusters = th.tensor([-2, 2]).float()
# x_stds = th.tensor([0.5,0.5, 0.5]).float()
x_stds = th.tensor([.6, .6]).float()

mix = D.Categorical(th.ones_like(x_clusters))
comp = D.Normal(x_clusters, x_stds)
gmm = MixtureSameFamily(mix, comp)
x_samples = gmm.sample(sample_shape=(train_size,1))

# x_interval = [-3,3]
# uni = D.Uniform(x_interval[0], x_interval[1])
# x_samples = uni.sample(sample_shape=(train_size,1))

# model repose samples
# define mean and std functions
def f_lin_1(x, scale=1):
    res = scale * x
    return res

def f_sin_1(x, scale=1):
    res = th.sin(scale*x)
    return res


def f_const_1(x, scale=1):
    res = th.ones_like(x) * scale
    return res

def f_mu_1(x, scale=1):
    res = scale * th.sin(x) * th.cos(x / 2)
    return res
def f_sig_1(x, scale=1e-1):
    # res = scale * (.5 * th.sin(x) + 1) + 3.5 * th.exp( - (x-35)**2 / 2)
    # res = scale * (th.sin(2*x)+1.5)
    res = scale * (th.sin(0.3*x)+1.5)
    # res = th.ones_like(x) * scale
    return res

def f_mu_2(x, scale=1):
    res = scale * th.cos(x_samples) * th.sin(x_samples / 2 + 2)
    return res
def f_sig_2(x, scale=1e-1):                     ## add a high noise cluster for evaluation
    res = scale * (.5 * th.cos(x) + 1) + 3.5 * th.exp( - (x-35)**2 / 2)
    return res

## define mixture components. Stds in x-direct are kept small. 
# mu_x1 = f_mu_1(x_samples, scale=0.6)
mu_x1 = 0.7 * f_sin_1(x_samples, scale=0.8)
mu_x1 = th.cat((x_samples, mu_x1), dim = -1)
# sig_x1 = f_sig_1(x_samples, scale=0.2e-1)
sig_x1 = f_const_1(x_samples, scale=0.15)
# sig_x1 = th.cat((1e-5 * th.ones_like(x_samples), sig_x1), dim = -1)
sig_x1 = th.cat((sig_x1, sig_x1), dim = -1)

mu_x2 = f_mu_2(x_samples, scale=0.6)
mu_x2 = th.cat((x_samples, mu_x2), dim = -1)
sig_x2 = f_sig_2(x_samples, scale=0.2e-1)
sig_x2 = th.cat((1e-5 * th.ones_like(x_samples), sig_x2), dim = -1)

sig_x2, mu_x2 = sig_x1, mu_x1

mu_mix = th.cat((mu_x1, mu_x2), dim = 0)
sig_mix = th.cat((sig_x1, sig_x2), dim = 0)

mix_y = D.Categorical(th.ones(2 * train_size,))
comps_y = D.Independent(D.Normal(
    loc = mu_mix, scale = sig_mix), 1)
gmm_y = MixtureSameFamily(mix_y, comps_y)

samples = gmm_y.sample(sample_shape=(train_size,))
# samples[...,1] = th.zeros_like(samples[...,1]) #<---- for zeros
# samples[...,1] = th.sgn(samples[...,0]) #<---- for stage

### LIVE PLOTTING
if live_draw:
    plt.ion()

### PREDICTION / EVALUATION
# create test values of x
x_test = th.linspace(-15, 15, steps=500)[...,None].float().cuda()
sample_size = 51
old_y_pred_vars1 = np.squeeze(np.zeros_like(x_test.cpu().detach().numpy()))

# draw samples from CDF by drawing from CDF^-1(u), u ~ U[0,1]
x_samples = x_test.repeat((1,sample_size)).flatten().cpu().detach().numpy()

# plot settings
# Neurips textwidth is 5.5
fig = plt.figure(figsize = figsize)

# training data plot
samples_np = samples.cpu().detach().numpy()

# eval plot placeholders
x_test_np = x_test.cpu().detach().numpy()
# ep_plt1 = plt.plot(x_test_np, np.zeros_like(x_test_np), color='orange')[0]
# ep_plt2 = plt.plot(x_test_np, np.zeros_like(x_test_np), color='yellow')[0]

# postpred_sc = plt.scatter(x_samples, np.zeros_like(x_samples), color=colors[0], s=0.5, alpha=1, label="Categorical", edgecolors="none")
# postpred_sc2 = plt.scatter(x_samples, np.zeros_like(x_samples), color=colors[1], s=0.5, alpha=1, label="Quantile", edgecolors="none")
# postpred_sc3 = plt.scatter(x_samples, np.zeros_like(x_samples), color='orange', s=1, alpha=.2, label="QR3")
# postpred_sc4 = plt.scatter(x_samples, np.zeros_like(x_samples), color='red', s=1, alpha=.1, label="QR4")

# map_sc = plt.scatter(x_samples, np.zeros_like(x_samples), color=darkcolors[0], s=1, alpha=.002, edgecolors="none")
# map_sc2 = plt.scatter(x_samples, np.zeros_like(x_samples), color=darkcolors[1], s=1, alpha=.002, edgecolors="none")

# map_sc3 = plt.scatter(x_samples, np.zeros_like(x_samples), color='orange', s=1, alpha=.2,)
# map_sc4 = plt.scatter(x_samples, np.zeros_like(x_samples), color='red', s=1, alpha=.2,)

# sc = plt.scatter(samples_np[:,0], samples_np[:,1], s = 1.5, color = "black", alpha = 1, label='Training Data', edgecolors="black")

# sc.set_rasterized(True)
# postpred_sc.set_rasterized(True)
# postpred_sc2.set_rasterized(True)
# map_sc.set_rasterized(True)
# map_sc2.set_rasterized(True)

y_lim = (-1.5,1.5)
plt.xlim([-10, 10])
plt.ylim(y_lim)

legend_list = [(mpatches.Patch(color=c), a) for c,a in zip(colors[0:2], ["Categorical", "Quantile"])]
plt.legend(*zip(*legend_list), loc='lower right') #, prop={'size': 18})

# plt.title("Toy 1D-Regression")
plt.xlabel("     ")

# plt.legend(loc=2, prop={'size': 12, alpha=1})
if live_draw:
    plt.show()

def eval_and_plot_net(models, mapscatter, postpredscatter, flush=True):

    y_pred_maps, y_stds, y_pred_postpreds, quant_vars = [], [], [], []
    max_count = 0

    for i in range(len(models)):
        cur_model = models[i]
        y_pred_map = cur_model.forward_sample(x_test, size=sample_size, reparametrize=False, sample_analytical=False)
        y_pred_postpred = cur_model.forward_sample(x_test, size=sample_size, reparametrize=True, sample_analytical=True)
        y_std = cur_model.forward_std(x_test, reparametrize=False)
        y_stds.append(y_std.cpu().detach().numpy())
        y_pred_maps.append(y_pred_map.cpu().detach().numpy())
        y_pred_postpreds.append(y_pred_postpred.cpu().detach().numpy())

        # epistemic uncertainty
        if models[0].loss_type == "evidential":
            _, _, quant_var  = models[i].forward_sample(x_test, size=sample_size, include_latent=True)
            quant_var = th.mean(quant_var, dim= -1).cpu().detach().numpy()
            quant_vars.append(quant_var)
        else: 
            quant_var = th.var(y_pred_postpred, dim=-1).cpu().detach().numpy()
            quant_vars.append(quant_var)

    if models[0].loss_type == "evidential":
        quant_var_pl = np.mean(quant_vars, axis=0)
    else:
        # quant_var_pl = np.var(y_pred_maps, axis=0)
        # quant_var_pl = np.reshape(quant_var_pl, (len(x_test_np), sample_size))
        # quant_var_pl = 1/np.mean(quant_var_pl, axis=-1)

        qaunt_var_this_step = np.mean(quant_vars, axis=0)
        quant_var_pl = np.abs(qaunt_var_this_step-old_y_pred_vars1)
    # fit to plot size
    quant_var_pl = quant_var_pl #/ 3 * np.median(quant_var_pl) * y_lim[-1]

    y_pred_map_pl = np.mean(y_pred_maps, axis=0)
    y_pred_postpred_pl = y_pred_postpreds[randint(0,len(models)-1)]
    y_std = y_stds[randint(0,len(models)-1)]

    print(f"max count :{max_count}")

    # update prediction plots    
    # ep_plt.set_ydata(quant_var_pl)
    # mapscatter.set_offsets(np.stack((x_samples,y_pred_map_pl), axis=-1))
    # postpredscatter.set_offsets(np.stack((x_samples,y_pred_postpred_pl), axis=-1))
    
    if live_draw:
        fig.canvas.draw()
    if flush:
        fig.canvas.flush_events()

    return x_samples, y_pred_map_pl, y_std, y_pred_postpred_pl

### MODEL
th.manual_seed(1)

n_outs = 101
models = []
models2 = []
models3 = []
models4 = []
ensemble_size = 1
for i in range(ensemble_size):
    net = Net(n_in=1, n_outs= n_outs, n_hidden=128, n_layers=1, lr=5e-4, weight_decay=5e-3, loss_type="categorical", last_layer_rbf=False, v_min=-1.5, v_max=1.5).cuda()
    # net.apply(init_weights_xav)
    models.append(net)

ensemble_size = 1
for i in range(ensemble_size):
    net = Net(n_in=1, n_outs= n_outs, n_hidden=128, n_layers=1, lr=5e-4, weight_decay=1e-4, loss_type="quantile", last_layer_rbf=False).cuda()
    # net.apply(init_weights_xav)
    models2.append(net)

bs = 128
epochs = 20000

for i in range(epochs):
    losses = []
    for m in range(len(models)):
        # sub_idx = np.random.choice(np.arange(0, train_size), size=bs, replace=True)
        # x_train, y_train = samples[sub_idx,0:1],samples[sub_idx,1:2]
        x_train, y_train = samples[:,0:1],samples[:,1:2]
        losses.append(models[m].fit(x_train, y_train))
    for m in range(len(models2)):
        # sub_idx = np.random.choice(np.arange(0, train_size), size=bs, replace=True)
        # x_train, y_train = samples[sub_idx,0:1],samples[sub_idx,1:2]
        x_train, y_train = samples[:,0:1],samples[:,1:2]
        losses.append(models2[m].fit(x_train, y_train))
    # for m in range(len(models3)):
    #     # sub_idx = np.random.choice(np.arange(0, train_size), size=bs, replace=True)
    #     # x_train, y_train = samples[sub_idx,0:1],samples[sub_idx,1:2]
    #     x_train, y_train = samples[:,0:1],samples[:,1:2]
    #     losses.append(models3[m].fit(x_train, y_train))
    # for m in range(len(models4)):
    #     # sub_idx = np.random.choice(np.arange(0, train_size), size=bs, replace=True)
    #     # x_train, y_train = samples[sub_idx,0:1],samples[sub_idx,1:2]
    #     x_train, y_train = samples[:,0:1],samples[:,1:2]
    #     losses.append(models4[m].fit(x_train, y_train))


    if i % 50 == 0 or i==epochs-1:
        flush = i<epochs-1
        print(i, [l.cpu().data.numpy() for l in losses])
        x, y_mean1, y_std1,  y_samples1 = eval_and_plot_net(models, None, None, flush = flush)
        x, y_mean2, y_std2, y_samples2 = eval_and_plot_net(models2, None, None, flush = flush)
        # eval_and_plot_nets(models3, map_sc3, postpred_sc3, flush = flush) 
        # eval_and_plot_net(models4, map_sc4, postpred_sc4, flush = flush)
        #print('Epoch %4d, Train loss projection = %6.3f, loss quantile = %6.3f, loss evidential = %6.3f' % \
        #    (i, proj_loss.cpu().data.numpy(), qreg_loss.cpu().data.numpy(), evid_loss.cpu().data.numpy())
        #    )

x_test_np = x_test.cpu().detach().numpy().copy().squeeze()
for i in range(y_samples1.shape[-1]):
    # 300 represents number of points to make between T.min and T.max
    
    y1 = y_samples1[:,i].squeeze()
    y2 = y_samples2[:,i].squeeze()

    smoothed1 = smooth(y1, weight=0.8)
    smoothed2 = smooth(y2, weight=0.8)

    b, a = scipy.signal.butter(3, 0.1)

    smoothed13 = scipy.signal.medfilt(y1, kernel_size=5)
    smoothed23 = scipy.signal.medfilt(y2, kernel_size=5)

    smoothed12 = scipy.signal.filtfilt(b, a, y1)
    smoothed22 = scipy.signal.filtfilt(b, a, y2)

    plt.plot(x_test_np, y1, color=colors[0], linewidth=0.3, alpha=1)
    # plt.plot(x_test_np, y2, color=colors[1], linewidth=0.3, alpha=1)

plt.plot(x_test_np, y_mean1, color=darkcolors[0], linewidth=0.5, alpha=1)
plt.plot(x_test_np, y_mean2, color=darkcolors[1], linewidth=0.5, alpha=1)

plt.fill_between(x_test_np, y_mean1.squeeze()-y_std1, y_mean1.squeeze()+y_std2, alpha=0.2)
plt.fill_between(x_test_np, y_mean2.squeeze()-y_std2, y_mean2.squeeze()+y_std2, alpha=0.2)

fig.canvas.draw()
ax_list = fig.get_axes()
ax_list[0].set_facecolor('#F0F0F0')
ax_list[0].grid(color='white')
sc = plt.scatter(samples_np[:,0], samples_np[:,1], s = 1.5, color = "black", alpha = 1, label='Training Data', edgecolors="black")

# save data
np.savez('toyregression.npz', x_test_np, samples_np, y_mean1, y_samples1, y_mean2, y_samples2)

    # "font.family": "serif",
    # # Use LaTeX default serif font.
    # "font.serif": [],
    # # Use specific cursive fonts.
    # "font.cursive": ["Comic Neue", "Comic Sans MS"],

plt.tight_layout()
plt.savefig('images/toy_regression.pgf', dpi=900)
plt.savefig('images/toy_regression.png', dpi=900)
