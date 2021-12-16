import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math 
from scipy.misc import derivative

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.device(0)
print(f'Using {device} device')

def init_weights_xav(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight, gain=1)
        m.bias.data.fill(0.0)


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
    def __init__(self, n_in, n_qs, n_hidden, n_layers, lr, weight_decay=0):
        super(Net, self).__init__()
        taus = np.arange(2 * n_qs + 1) / (2 * n_qs)
        self.taus, = to_variable(var=(taus,), cuda=True)
        self.n_taus = len(self.taus)

        self.layers = nn.ModuleList([nn.Linear(n_in, n_hidden)])
        self.layers.extend([nn.Linear(n_hidden, n_hidden) for i in range(1, n_layers)])
        self.layers.append(nn.Linear(n_hidden, self.n_taus))

        self.loss = self.kl_proj_loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
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

    def kl_proj_loss(self, outs, samples):
        thetas = outs
        loss = torch.mean(
            self.rho_tau(samples - thetas, self.taus, kappa = 0.0) 
            )
        return loss

n_q = 250
net = Net(n_in=1, n_qs= n_q, n_hidden=200, n_layers=2, lr=1e-4, weight_decay=0).cuda()



torch.manual_seed(1)

x_u = torch.unsqueeze(torch.linspace(-1,1,20000), dim=1) # shape = (1000, 1)
mu = 1
sigma = 1.5
x_n = torch.normal(1,1.5, size=(20000,1)) # shape = (1000, 1)
y_n = normal(x_n, mu, sigma)

bs = 4096
epochs = 5000

sub_idx = np.random.choice(np.arange(0, len(x_n)), size=bs, replace=True)
y_train = x_n[sub_idx]
x_train = np.zeros_like(y_train)

for i in range(epochs):
    loss = net.fit(x_train, y_train)
    if i % 200 == 0:
        print('Epoch %4d, Train loss = %6.3f' % (i, loss.cpu().data.numpy()))

x_test = np.linspace(-10, 12, 200)
y_test = normal(x_test, mean=mu, sigma=sigma)
quants_pred = net(torch.tensor(np.ones(shape=(1,1))).float().cuda()).cpu().detach().numpy()[0]
quants_cdf = np.linspace(0, 1, 2 * n_q + 1)
quants_pdf = derivative(quants_cdf, ) np.diff(quants_cdf) / np.diff(quants_pred)
# y_pdf_pred = y_cdf_pred[1:]-y_cdf_pred[:1]

plt.figure(figsize = (6, 5))
plt.style.use('default')
plt.plot(x_test, y_test, color = 'green', alpha = 0.5, label='pdf')
plt.plot(quants_pred, quants_cdf, color = 'red', alpha = 0.5, label='pdf')
plt.plot(quants_pred[1:], quants_pdf, color = 'red', alpha = 0.5, label='pdf')
# plt.scatter(x_test, y_pred, s = 10, marker = 'x', color = 'red', alpha = 0.5)
# plt.plot(np.linspace(xmin, xmax, n_testsamples))
# plt.fill_between(np.linspace(xmin, xmax, n_testsamples), means + aleatoric, means + total_unc, color = c[0], alpha = 0.3, label = r"$\sigma(y'|x')$")
# plt.fill_between(np.linspace(xmin, xmax, n_testsamples), means - total_unc, means - aleatoric, color = c[0], alpha = 0.3)
# plt.fill_between(np.linspace(xmin, xmax, n_testsamples), means - aleatoric, means + aleatoric, color = c[1], alpha = 0.4, label = r'$\mathbb{E}_{q(\mathbf{w})}[\sigma^2]^{1/2}$')
# plt.plot(np.linspace(xmin, xmax, n_testsamples), means, color = 'black', linewidth = 1, label='means')
# plt.xlim([xmin, xmax])
# plt.ylim([-5, 20])
# plt.xlabel('$x$', fontsize=5)
# plt.title('map ensemble', fontsize=40)
# plt.tick_params(labelsize=5)
# plt.xticks(np.arange(xmin, xmax, 2))
# plt.yticks(np.arange(-4, 7, 2))
# plt.gca().yaxis.grid(alpha=0.3)
# plt.gca().xaxis.grid(alpha=0.3)
# plt.savefig(f'map_ensemble{i}.pdf', bbox_inches = 'tight')

plt.show()