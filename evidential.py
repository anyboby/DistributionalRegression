import functools
import numpy as np
import matplotlib.pyplot as plt

import evidential_deep_learning as edl
import tensorflow as tf
import torch
from torch import distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

def main():
    # Create some training and testing data
    x_train, y_train = my_data(-4, 4, 10000)
    x_test, y_test = my_data(-65, 65, 1000, train=False)

    # Define our model with an evidential output
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dense(200, activation="relu"),
        edl.layers.DenseNormalGamma(1),
    ])

    # Custom loss function to handle the custom regularizer coefficient
    def EvidentialRegressionLoss(true, pred):
        return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)

    # Compile and fit the model!
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=EvidentialRegressionLoss)
    model.fit(x_train, y_train, batch_size=100, epochs=500)

    # Predict and plot using the trained model
    y_pred = model(x_test)
    plot_predictions(x_train, y_train, x_test, y_test, y_pred)

    # Done!!


#### Helper functions ####
def my_data(x_min, x_max, n, train=True):
    # generate xs
    x_clusters = torch.tensor([-25, 0, 35]).float()
    x_stds = torch.tensor([1.5,2.5, 2.5]).float()
    mix = D.Categorical(torch.ones_like(x_clusters))
    comp = D.Normal(x_clusters, x_stds)
    gmm = MixtureSameFamily(mix, comp)

    if train:
        x_samples = torch.sort(gmm.sample(sample_shape=(n,1)), dim=0)[0]
    else: 
        x_samples = torch.linspace(x_min, x_max, steps=n)[...,None].float()

    # model repose samples
    # define mean and std functions
    def f_mu_1(x, scale=1):
        res = scale * torch.sin(2 * x) * torch.cos(x / 2)
        return res
    def f_sig_1(x, scale=1e-1):
        res = scale * (.5 * torch.sin(x) + 1) + 3.5 * torch.exp( - (x-35)**2 / 2)
        return res

    def f_mu_2(x, scale=1):
        res = scale * torch.cos(x_samples) * torch.sin(x_samples / 2 + 2)
        return res
    def f_sig_2(x, scale=1e-1):                     ## add a high noise cluster for evaluation
        res = scale * (.5 * torch.cos(x) + 1) + 3.5 * torch.exp( - (x-35)**2 / 2)
        return res

    ## define mixture components. Stds in x-direct are kept small. 
    mu_x1 = f_mu_1(x_samples, scale=2.5)
    mu_x1 = torch.cat((x_samples, mu_x1), dim = -1)
    sig_x1 = f_sig_1(x_samples, scale=1.5e-1)
    sig_x1 = torch.cat((1e-5 * torch.ones_like(x_samples), sig_x1), dim = -1)

    mu_x2 = f_mu_2(x_samples, scale=2.5)
    mu_x2 = torch.cat((x_samples, mu_x2), dim = -1)
    sig_x2 = f_sig_2(x_samples, scale=3.5e-1)
    sig_x2 = torch.cat((1e-5 * torch.ones_like(x_samples), sig_x2), dim = -1)

    mu_mix = torch.cat((mu_x1, mu_x2), dim = 0)
    sig_mix = torch.cat((sig_x1, sig_x2), dim = 0)

    mix_y = D.Categorical(torch.ones(2 * n,))
    comps_y = D.Independent(D.Normal(
        loc = mu_mix, scale = sig_mix), 1)
    gmm_y = MixtureSameFamily(mix_y, comps_y)

    samples = gmm_y.sample(sample_shape=(n,))
    # samples[...,1] = torch.zeros_like(samples[...,1]) #<---- for zeros
    samples_np = samples.cpu().detach().numpy()
    
    x = samples_np[...,0:1]
    y = samples_np[...,1:2]
    # x = np.linspace(x_min, x_max, n)
    # x = np.expand_dims(x, -1).astype(np.float32)

    # sigma = 3 * np.ones_like(x) if train else np.zeros_like(x)
    # y = x**3 + np.random.normal(0, sigma).astype(np.float32)

    return x, y

def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))
    var = np.minimum(var, 1e3)[:, 0]  # for visualization

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
    plt.scatter(x_test, y_test, s=1., color='red', zorder=2, label="Test")
    plt.scatter(x_test, mu, s=1., color='#007cab', zorder=3, label="Pred")
    # plt.plot([-4, -4], [-150, 150], 'k--', alpha=0.4, zorder=0)
    # plt.plot([+4, +4], [-150, 150], 'k--', alpha=0.4, zorder=0)
    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            x_test, (mu - k * var), (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)
    plt.gca().set_ylim(-5, 5)
    plt.gca().set_xlim(-65, 65)
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()