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
from random import randint

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
            # counts = torch.exp(self.counts_scale) * counts
            if reparametrize:
                x = self.reparametrize(mu, counts), mu, counts
            else: 
                x = mu, mu, counts
        return x
    
    def forward_sample(self, x, size=1, include_latent = False, reparametrize=False):
        x = self.forward(x, reparametrize=reparametrize)
    
        if self.loss_type == "quantile" or self.loss_type == "implicit_quantile":
            quant_idx = torch.randint(low=0, high=self.n_outs, size=(x.shape[0],size))
            x_idx = torch.arange(0, len(x))[...,None]
            if reparametrize:
                samples = x[x_idx, quant_idx]
            else:
                samples = x.mean(dim=-1, keepdim=True).expand((quant_idx.shape))

        elif self.loss_type == "projection":
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
train_size = 5000

# generate xs
# x_clusters = torch.tensor([-6, 0.2, 7]).float()
x_clusters = torch.tensor([-2.5, 0, 2.5]).float()
# x_stds = torch.tensor([0.5,0.5, 0.5]).float()
x_stds = torch.tensor([.2,1.5,.2]).float()

mix = D.Categorical(torch.ones_like(x_clusters))
comp = D.Normal(x_clusters, x_stds)
gmm = MixtureSameFamily(mix, comp)
x_samples = gmm.sample(sample_shape=(train_size,1))


x_interval = [-3,3]
uni = D.Uniform(x_interval[0], x_interval[1])
x_samples = uni.sample(sample_shape=(train_size,1))

# model repose samples
# define mean and std functions
def f_lin_1(x, scale=1):
    res = scale * x
    return res

def f_sin_1(x, scale=1):
    res = torch.sin(scale*x)
    return res


def f_const_1(x, scale=1):
    res = torch.ones_like(x) * scale
    return res

def f_mu_1(x, scale=1):
    res = scale * torch.sin(x) * torch.cos(x / 2)
    return res
def f_sig_1(x, scale=1e-1):
    # res = scale * (.5 * torch.sin(x) + 1) + 3.5 * torch.exp( - (x-35)**2 / 2)
    # res = scale * (torch.sin(2*x)+1.5)
    res = scale * (torch.sin(0.3*x)+1.5)
    # res = torch.ones_like(x) * scale
    return res

def f_mu_2(x, scale=1):
    res = scale * torch.cos(x_samples) * torch.sin(x_samples / 2 + 2)
    return res
def f_sig_2(x, scale=1e-1):                     ## add a high noise cluster for evaluation
    res = scale * (.5 * torch.cos(x) + 1) + 3.5 * torch.exp( - (x-35)**2 / 2)
    return res

## define mixture components. Stds in x-direct are kept small. 
# mu_x1 = f_mu_1(x_samples, scale=0.6)
mu_x1 = f_sin_1(x_samples, scale=0.8)
mu_x1 = torch.cat((x_samples, mu_x1), dim = -1)
# sig_x1 = f_sig_1(x_samples, scale=0.2e-1)
sig_x1 = f_const_1(x_samples, scale=0.1)
# sig_x1 = torch.cat((1e-5 * torch.ones_like(x_samples), sig_x1), dim = -1)
sig_x1 = torch.cat((sig_x1, sig_x1), dim = -1)

mu_x2 = f_mu_2(x_samples, scale=0.6)
mu_x2 = torch.cat((x_samples, mu_x2), dim = -1)
sig_x2 = f_sig_2(x_samples, scale=0.2e-1)
sig_x2 = torch.cat((1e-5 * torch.ones_like(x_samples), sig_x2), dim = -1)

sig_x2, mu_x2 = sig_x1, mu_x1

mu_mix = torch.cat((mu_x1, mu_x2), dim = 0)
sig_mix = torch.cat((sig_x1, sig_x2), dim = 0)

mix_y = D.Categorical(torch.ones(2 * train_size,))
comps_y = D.Independent(D.Normal(
    loc = mu_mix, scale = sig_mix), 1)
gmm_y = MixtureSameFamily(mix_y, comps_y)

samples = gmm_y.sample(sample_shape=(train_size,))
# samples[...,1] = torch.zeros_like(samples[...,1]) #<---- for zeros

### LIVE PLOTTING
plt.ion()

### PREDICTION / EVALUATION
# create test values of x
x_test = torch.linspace(-8, 8, steps=2000)[...,None].float().cuda()
sample_size = 200
old_y_pred_vars1 = np.squeeze(np.zeros_like(x_test.cpu().detach().numpy()))

# draw samples from CDF by drawing from CDF^-1(u), u ~ U[0,1]
x_samples = x_test.repeat((1,sample_size)).flatten().cpu().detach().numpy()

# plot settings
fig = plt.figure(figsize = (4, 3))
plt.style.use('default')

# training data plot
samples_np = samples.cpu().detach().numpy()

# eval plot placeholders
x_test_np = x_test.cpu().detach().numpy()
# ep_plt1 = plt.plot(x_test_np, np.zeros_like(x_test_np), color='orange')[0]
# ep_plt2 = plt.plot(x_test_np, np.zeros_like(x_test_np), color='yellow')[0]

postpred_sc = plt.scatter(x_samples, np.zeros_like(x_samples), color='purple', s=1, alpha=.02, label="QR1")
postpred_sc2 = plt.scatter(x_samples, np.zeros_like(x_samples), color='blue', s=1, alpha=.01, label="QR2")
postpred_sc3 = plt.scatter(x_samples, np.zeros_like(x_samples), color='orange', s=1, alpha=.02, label="CDR1")
postpred_sc4 = plt.scatter(x_samples, np.zeros_like(x_samples), color='red', s=1, alpha=.01, label="CDR2")

map_sc = plt.scatter(x_samples, np.zeros_like(x_samples), color='purple', s=1, alpha=.1,)
map_sc2 = plt.scatter(x_samples, np.zeros_like(x_samples), color='blue', s=1, alpha=.1,)
map_sc3 = plt.scatter(x_samples, np.zeros_like(x_samples), color='orange', s=1, alpha=.1,)
map_sc4 = plt.scatter(x_samples, np.zeros_like(x_samples), color='red', s=1, alpha=.1,)

sc = plt.scatter(samples_np[:,0], samples_np[:,1], s = 10, color = 'green', alpha = 0.2, label='Data')


y_lim = (-3,3)
plt.xlim([-8, 8])
plt.ylim(y_lim)
leg = plt.legend(loc=3, prop={'size': 18})
for lh in leg.legendHandles: 
    lh.set_alpha(1)
    lh._sizes = [100]


# plt.legend(loc=2, prop={'size': 12, alpha=1})
plt.show()

def eval_and_plot_net(models, mapscatter, postpredscatter, flush=True):

    y_pred_maps, y_pred_postpreds, quant_vars = [], [], []
    max_count = 0

    for i in range(len(models)):
        cur_model = models[i]
        y_pred_map = cur_model.forward_sample(x_test, size=sample_size, reparametrize=False)
        y_pred_postpred = cur_model.forward_sample(x_test, size=sample_size, reparametrize=True)

        y_pred_maps.append(y_pred_map.flatten().cpu().detach().numpy())
        y_pred_postpreds.append(y_pred_postpred.flatten().cpu().detach().numpy())

        # epistemic uncertainty
        if models[0].loss_type == "evidential":
            _, _, quant_var  = models[i].forward_sample(x_test, size=sample_size, include_latent=True)
            quant_var = torch.mean(quant_var, dim= -1).cpu().detach().numpy()
            quant_vars.append(quant_var)
        else: 
            quant_var = torch.var(y_pred_postpred, dim=-1).cpu().detach().numpy()
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

    y_pred_map_pl = np.mean(y_pred_maps, axis=0 )
    y_pred_postpred_pl = y_pred_postpreds[randint(0,len(models)-1)]

    print(f"max count :{max_count}")

    # update prediction plots    
    # ep_plt.set_ydata(quant_var_pl)
    mapscatter.set_offsets(np.stack((x_samples,y_pred_map_pl), axis=-1))
    postpredscatter.set_offsets(np.stack((x_samples,y_pred_postpred_pl), axis=-1))
    
    fig.canvas.draw()
    if flush:
        fig.canvas.flush_events()

    return qaunt_var_this_step



### MODEL
torch.manual_seed(1)

n_outs = 201
models = []
models2 = []
models3 = []
models4 = []
loss_type = "implicit_quantile"
ensemble_size = 1
for i in range(ensemble_size):
    net = Net(n_in=1, n_outs= n_outs, n_hidden=512, n_layers=1, lr=1e-4, weight_decay=0, loss_type=loss_type, last_layer_rbf=False).cuda()
    net.apply(init_weights_xav)
    models.append(net)

ensemble_size2 = 1
for i in range(ensemble_size2):
    net = Net(n_in=1, n_outs= n_outs, n_hidden=512, n_layers=1, lr=1e-4, weight_decay=0, loss_type=loss_type, last_layer_rbf=False).cuda()
    net.apply(init_weights_xav)
    models2.append(net)

loss_type2 = "projection"
ensemble_size3 = 1
for i in range(ensemble_size):
    net = Net(n_in=1, n_outs= n_outs, n_hidden=512, n_layers=1, lr=1e-4, weight_decay=0, loss_type=loss_type2, last_layer_rbf=False, v_min=-3, v_max=3).cuda()
    net.apply(init_weights_xav)
    models3.append(net)

ensemble_size4 = 1
for i in range(ensemble_size2):
    net = Net(n_in=1, n_outs= n_outs, n_hidden=512, n_layers=1, lr=1e-4, weight_decay=0, loss_type=loss_type2, last_layer_rbf=False, v_min=-3, v_max=3).cuda()
    net.apply(init_weights_xav)
    models4.append(net)


bs = 5000
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
    for m in range(len(models3)):
        # sub_idx = np.random.choice(np.arange(0, train_size), size=bs, replace=True)
        # x_train, y_train = samples[sub_idx,0:1],samples[sub_idx,1:2]
        x_train, y_train = samples[:,0:1],samples[:,1:2]
        losses.append(models3[m].fit(x_train, y_train))
    for m in range(len(models4)):
        # sub_idx = np.random.choice(np.arange(0, train_size), size=bs, replace=True)
        # x_train, y_train = samples[sub_idx,0:1],samples[sub_idx,1:2]
        x_train, y_train = samples[:,0:1],samples[:,1:2]
        losses.append(models4[m].fit(x_train, y_train))


    # for m in range(len(models2)):
    #     sub_idx = np.random.choice(np.arange(0, train_size), size=bs, replace=True)
    #     # x_train = samples[sub_idx,0:1]
    #     x_train, y_train = samples[sub_idx,0:1],samples[sub_idx,1:2]

    #     # sub_idx2 = np.random.choice(np.arange(0, len(x_train)), size=bs, replace=True)

    #     # y_train = models[0].forward(x_train.to("cuda"))
    #     # x_train = x_train.expand(y_train.shape)
    #     # x_train, y_train = x_train.flatten().unsqueeze(-1), y_train.flatten().unsqueeze(-1)

    #     # y_train2 = models[1].forward(x_train.to("cuda")).mean(dim=-1, keepdim=True)
    #     # mus = (y_train1 + y_train2)/2
    #     # vars = (y_train1 - mus)**2/2 + (y_train2 - mus)**2/2
    #     # y_train = torch.normal(mus, torch.sqrt(vars))
    #     # x_train2 = x_train.expand(y_train2.shape)
    #     # x_train2, y_train2 = x_train2.flatten().unsqueeze(-1)[sub_idx2], y_train2.flatten().unsqueeze(-1)[sub_idx2]
    #     # 
    #     losses.append(models2[m].fit(x_train, y_train))


    if i % 2000 == 0 or i==epochs-1:
        flush = i<epochs-1
        print(i, [l.cpu().data.numpy() for l in losses])
        eval_and_plot_net(models, map_sc, postpred_sc, flush = flush)
        eval_and_plot_net(models2, map_sc2, postpred_sc2, flush = flush)
        eval_and_plot_net(models3, map_sc3, postpred_sc3, flush = flush) 
        eval_and_plot_net(models4, map_sc4, postpred_sc4, flush = flush)
        #print('Epoch %4d, Train loss projection = %6.3f, loss quantile = %6.3f, loss evidential = %6.3f' % \
        #    (i, proj_loss.cpu().data.numpy(), qreg_loss.cpu().data.numpy(), evid_loss.cpu().data.numpy())
        #    )
plt.savefig(f'quant_reg_{loss_type}.pdf', bbox_inches = 'tight')
# plt.savefig(f'quant_reg_{loss_type}.eps', bbox_inches = 'tight', format="eps")
plt.savefig(f'quant_reg_{loss_type}.png', dpi=300)
