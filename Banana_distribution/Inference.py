from SVGD import run_svgd
from Banana import Banana_dist
import tensorflow as tf
import numpy as np
from RMH import run_chain_RMH
from HMC import run_chain_HMC
from HessianMC import run_chain_hessian
import matplotlib.pyplot as plt


class MCMC:

    def __init__(self, target_distribution=Banana_dist()):

        self.target_distribution = target_distribution
        self.num_results = 500
        self.burnin = 100
        self.initial_chain_state = tf.constant([[-1.7, -1.7]])

    @tf.function
    def unnormalized_posterior_log_prob(self, *args):
        return self.target_distribution.joint_log_post(*args)

    def run_chain(self, method='RMH'):
        """run MCMC chain on given target distribution

        Keyword Arguments:
            method {str} -- options for MCMC methods, 'RMC','HMC', 'HessianMC'(default: {'RMH'})

        Raises:
            ValueError: [SVGD is a variational methods, use the other class VI: Variational inference]

        Returns:
            accepted,rejected {tensor} -- accepted and rejected samples in MCMC chain
        """
        self.method = method
        if self.method == 'RMH':  # Random Walk Matroplis Hasting algorithm
            scale = 0.1
            samples, kernel_results = run_chain_RMH(scale, self.num_results, self.burnin,
                                                    self.initial_chain_state, self.unnormalized_posterior_log_prob)

            samples = tf.squeeze(samples)
            accepted_ = tf.squeeze(kernel_results.is_accepted)
            samples = samples.numpy()
            accepted_ = accepted_.numpy()
            accepted = samples[np.where(accepted_ == True)]
            rejected = samples[np.where(accepted_ == False)]

            self.acceptance_rate_RMH = accepted.shape[0]/self.num_results

            return accepted, rejected

        if self.method == 'HMC':  # Hamiltonian Monte Carlo algoritem
            samples, kernel_results = run_chain_HMC(self.num_results, self.burnin,
                                                    self.initial_chain_state, self.unnormalized_posterior_log_prob)

            samples = tf.squeeze(samples)
            accepted_ = tf.squeeze(kernel_results.is_accepted)
            samples = samples.numpy()
            accepted_ = accepted_.numpy()
            accepted = samples[np.where(accepted_ == True)]
            rejected = samples[np.where(accepted_ == False)]

            self.acceptance_rate_HMC = accepted.shape[0]/self.num_results

            return accepted, rejected

        if self.method == 'HessianMC':
            accepted, rejected = run_chain_hessian(self.target_distribution.cov, self.num_results, self.burnin,
                                                   self.unnormalized_posterior_log_prob)
            accepted = np.array(accepted)
            rejected = np.array(rejected)
            return accepted, rejected

        if self.method == 'SVGD':
            raise ValueError('SVGD is not MCMC, use \'run_svgd\' ')


class VI():
    def __init__(self, target_distribution=Banana_dist()):
        self.target_distribution = target_distribution

    @tf.function
    def unnormalized_posterior_log_prob(self, *args):
        return self.target_distribution.joint_log_post(*args)

    def run_inference(self):
        self.results = run_svgd(self.unnormalized_posterior_log_prob)
        return self.results

    def plot(self):
        range_limit = [-3, 3]
        _, ax = plt.subplots(figsize=(15, 7))
        ax = plt.subplot(1, 3, 1, aspect='equal')
        plt.scatter(self.results[:, 0], self.results[:, 1])
        plt.xlim(range_limit)
        plt.ylim(range_limit)
        ax.set_title('Updated particles: $p_0(z)$')
