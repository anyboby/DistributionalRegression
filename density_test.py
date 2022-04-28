import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.autograd import Variable, grad
import torch.autograd.functional as AGF
from tqdm import tqdm 

import numpy as np
import matplotlib.pyplot as plt

def init_weights_xav(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight, gain=0.6)
        # m.bias.data.fill_(0.0)


def init_weights_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)


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

def grad_regularization(output, input):
    # weights = abs(th.linspace(-1,1,30).view(1,-1)).cuda()
    weights = abs(th.linspace(-1,1, output.shape[-1])).unsqueeze(0).to(output.device)**2
    var = th.var(weights * output, dim=-1)
    grad_var = grad(outputs=var, inputs=input, grad_outputs=th.ones_like(var), retain_graph=True, create_graph=True)[0]
    grad2_var = grad(outputs=th.squeeze(grad_var), inputs=input, grad_outputs=th.ones_like(var), retain_graph=True, create_graph=True)[0]
    # grad_mse = var +  th.sigmoid(-grad2_var)
    # grad_mse = th.exp(-grad2_var)
    grad_mse = th.sigmoid(-grad2_var)
    # X.requires_grad = True
    # Y_net = net(X)
    # loss_g = torch.zeros([0.0])
    # # loop over minibatch
    # for idx in range(Y_net.size()[0]):
    #   dydx[idx,:] = torch.autograd.grad(Y_net[idx,0],X,create_graph=True)[0][idx,:]
    #   loss_g = loss_g + (dydx[:,torch.arange(0,Nin)]).sum()
    # loss = loss_f(Y_net, Y) + loss_g
    # optimizer.zero_grad()
    # loss.backward(retain_graph=True)
    # optimizer.step()
    return grad_mse

class Net(nn.Module):
    def __init__(self, n_in, n_outs, n_hidden, n_layers, lr, weight_decay=0, loss_type='mse'):
        super(Net, self).__init__()

        self.n_outs = n_outs

        self.layers = nn.ModuleList([nn.Linear(n_in, n_hidden)])
        self.layers.extend([nn.Linear(n_hidden, n_hidden) for i in range(1, n_layers)])

        self.loss_type = loss_type

        if loss_type == "mse" or loss_type=="mse_gradreg":
            self.loss = mse_loss
            self.layers.append(nn.Linear(n_hidden, self.n_outs))
            
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.activation = F.silu

    def forward(self, x):
        for l in range(len(self.layers)-1):
            x = self.layers[l](x)
            x = self.activation(x)
        
        x=self.layers[-1](x)

        return x
    
    def fit(self, x, y):
        x, y = to_variable(var=(x,y), cuda=True)            
        out = self.forward(x)
        
        self.optimizer.zero_grad()
        loss = self.loss(out, y)

        if self.loss_type == "mse_gradreg":
            #reset grad and loss
            grad_reg = th.mean(grad_regularization(out, x))
            loss = loss + grad_reg #self.loss(out, y) + th.clip(grad_reg, -0.1, 0.1)
        
        # grads = grad(loss, self.parameters())
        loss.backward()
        self.optimizer.step()

        return loss


net_gradreg = Net(n_in=1, n_outs=100, n_hidden=128, n_layers=3, lr=1e-3, weight_decay=0, loss_type="mse_gradreg").cuda()
net = Net(n_in=1, n_outs=30, n_hidden=128, n_layers=3, lr=1e-3, weight_decay=0, loss_type="mse").cuda()
epochs = 5000

### TRAINING DATA ###
### more complex mixture samples
train_size = 10000
batch_size = 64

# generate xs
x_clusters = th.tensor([-5.5, 0, 6.5]).float()
x_stds = th.tensor([1,0.5, 1]).float()
mix = D.Categorical(th.ones_like(x_clusters))
comp = D.Normal(x_clusters, x_stds)
gmm = MixtureSameFamily(mix, comp)

x_samples = gmm.sample(sample_shape=(train_size,1))

# model repose samples
# define mean and std functions
def f_mu_1(x, scale=1):
    res = scale * th.sin(2 * x) * th.cos(x / 2)
    return res
def f_sig_1(x, scale=1e-1):
    res = scale * (.5 * th.sin(x) + 1) + 3.5 * th.exp( - (x-35)**2 / 2)
    return res

def f_mu_2(x, scale=1):
    res = scale * th.sin(2 * x) * th.cos(x / 2) # scale * th.cos(x_samples) * th.sin(x_samples / 2 + 2)
    return res
def f_sig_2(x, scale=1e-1):                     ## add a high noise cluster for evaluation
    res = scale * (.5 * th.cos(x) + 1) + 3.5 * th.exp( - (x-35)**2 / 2)
    return res

## define mixture components. Stds in x-direct are kept small. 
mu_x1 = f_mu_1(x_samples, scale=1.5)
mu_x1 = th.cat((x_samples, mu_x1), dim = -1)
sig_x1 = f_sig_1(x_samples, scale=1e-1)
sig_x1 = th.cat((1e-5 * th.ones_like(x_samples), sig_x1), dim = -1)

mu_x2 = f_mu_2(x_samples, scale=1.5)
mu_x2 = th.cat((x_samples, mu_x2), dim = -1)
sig_x2 = f_sig_2(x_samples, scale=2.5e-1)
sig_x2 = th.cat((1e-5 * th.ones_like(x_samples), sig_x2), dim = -1)

mu_mix = th.cat((mu_x1, mu_x2), dim = 0)
sig_mix = th.cat((sig_x1, sig_x2), dim = 0)

mix_y = D.Categorical(th.ones(2 * train_size,))
comps_y = D.Independent(D.Normal(
    loc = mu_mix, scale = sig_mix), 1)
gmm_y = MixtureSameFamily(mix_y, comps_y)

samples = gmm_y.sample(sample_shape=(train_size,))

# train_size = 300
# batch_size = 64
# x_train = th.normal(-3, 0.2, size=(100,1))
# x_train = th.concat((x_train, th.normal(0.0, 0.2, size=(100,1))), dim=0)
# x_train = th.concat((x_train, th.normal(3, 0.2, size=(100,1))), dim=0).cuda()
# y_train = th.zeros_like(x_train).cuda().detach()
# samples = th.concat((x_train, y_train), -1)

train_size = 40
batch_size = 40
train_x = th.linspace(-7,7,train_size).cuda().unsqueeze(-1)
# train_y = th.sin(2 * train_x).cuda().detach()
train_y = th.zeros_like(train_x).cuda()
samples = th.concat((train_x, train_y), -1)

pbar = tqdm(range(epochs))
for i in pbar:
    sub_idx = np.random.choice(np.arange(0, train_size), size=batch_size, replace=True)
    x_train, y_train = samples[sub_idx,0:1],samples[sub_idx,1:2]
    x_train.requires_grad=True

    l_mse = net.fit(x_train.detach(), y_train)
    l_mse_gradreg = net_gradreg.fit(x_train, y_train)
    pbar.set_description(f"mse loss: {l_mse}, grad_reg loss: {l_mse_gradreg}")

test_x = th.linspace(-10, 10, 1000)[...,None].cuda()
test_y_gradreg = net_gradreg(test_x).detach().cpu().numpy()
test_y_mse = net(test_x).detach().cpu().numpy()

plt.plot(test_x.detach().cpu().numpy(), test_y_gradreg, color="red", alpha=0.4, label="mse with var regularization", zorder=1)
plt.plot(test_x.detach().cpu().numpy(), test_y_mse, color="blue", alpha=0.4, label="regular mse", zorder=1)
# plt.legend(loc="upper left")
plt.scatter(samples[:,0].detach().cpu().numpy(), samples[:,1].detach().cpu().numpy(), color='orange', zorder=2, s=0.2)
# for  i in range(5):s
    
plt.show()