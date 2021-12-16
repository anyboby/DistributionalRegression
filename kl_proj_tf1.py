import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Quantiles:
    def __init__(self, taus, tf_graph=None):
        self.taus = taus
        N = len(taus)

        graph = tf_graph or tf.get_default_graph()
        with graph.as_default():
            with tf.variable_scope('quantiles'):
                self.xs = tf.placeholder('float')
                self.theta = tf.get_variable('theta', shape=(N,))
                self.loss = sum(
                    tf.reduce_mean(self._rho_tau(self.xs - self.theta[i],
                                                 taus[i], kappa=0))
                    for i in range(N))
                self.train_step = tf.train.AdamOptimizer(0.05).minimize(
                    self.loss)

    @staticmethod
    def _HL(u, kappa):
        delta = tf.cast(abs(u) <= kappa, 'float')
        return delta * (u * u / 2) + (1 - delta) * (
                kappa * (abs(u) - kappa / 2))

    @staticmethod
    def _rho_tau(u, tau, kappa=1):
        delta = tf.cast(u < 0, 'float')
        if kappa == 0:
            return (tau - delta) * u
        else:
            return abs(tau - delta) * Quantiles._HL(u, kappa)

    def get_quantiles(self, samples, loops):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for _ in range(loops):
                loss, _ = sess.run([self.loss, self.train_step],
                                   {self.xs: samples})
            qs = sess.run(self.theta)
        return qs


class MixtureOfGaussians:
    def __init__(self, pis, mus, sigmas):
        self.pis = pis
        self.mus = mus
        self.sigmas = sigmas

    def draw_samples(self, n):
        samples = np.empty(n)
        for i in range(n):
            idx = np.random.multinomial(1, self.pis).argmax()
            samples[i] = np.random.normal(self.mus[idx], self.sigmas[idx])
        return samples

    def pdf(self, x):
        return np.sum(pi * np.exp(-0.5 * ((x - mu) / s) ** 2) /
                        (s * np.sqrt(2 * pi))
                      for pi, mu, s in zip(self.pis, self.mus, self.sigmas))


tf.reset_default_graph()

MoG = MixtureOfGaussians(pis=[1/3, 1/3, 1/3], mus=[-3, 0, 5], sigmas=[2, 1, 2])
xs = np.linspace(-11, 11, num=200)
ys = MoG.pdf(xs)

N = 10  # num of quantiles
taus = [i / (2 * N) for i in range(0, 2 * N + 1)]
Q = Quantiles(taus)
samples = MoG.draw_samples(10000)
qs = Q.get_quantiles(samples, loops=2000)

plt.plot(xs, ys)

for q in qs[::2]:
    plt.plot([q, q], [0, MoG.pdf(q)], 'black')
plt.plot(qs[1::2], np.zeros_like(qs[1::2]), 'or')

plt.savefig('quantiles.svg', bbox_inches='tight')
plt.show()
