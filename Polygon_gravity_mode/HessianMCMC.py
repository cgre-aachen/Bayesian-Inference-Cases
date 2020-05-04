import tensorflow_probability as tfp
from scipy.stats import norm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tfd = tfp.distributions


def constant64(i):
    return(tf.constant(i, dtype=tf.float64))


class HessianMCMC():
    def __init__(self, Number_para, negative_log_posterior, Data, MAP, C_prior, number_sample, number_burnin, mu_init, beta=constant64(0.1)):
        self.Number_para = Number_para
        self.negative_log_posterior = negative_log_posterior
        self.Data = Data
        self.MAP = MAP
        self.C_prior = C_prior
        self.number_sample = number_sample
        self.number_burnin = number_burnin
        self.mu_init = mu_init
        self.beta = beta

    def Full_Hessian(self):
        Hess = tf.TensorArray(tf.float64, size=self.Number_para)
        j = 0
        for i in range(self.Number_para):
            with tf.GradientTape() as t:
                t.watch(self.MAP)
                with tf.GradientTape() as tt:
                    tt.watch(self.MAP)
                    loss = self.negative_log_posterior(self.Data, self.MAP)
                jac = tt.gradient(
                    loss, self.MAP, unconnected_gradients='zero')[i]
            hess = t.gradient(jac, self.MAP, unconnected_gradients='none')
            Hess = Hess.write(j, hess)
            j = j+1
        self.Hessian = Hess.stack()

    def Laplace_appro(self):
        self.cov_post = tf.linalg.inv(
            (tf.add(self.Hessian, tf.linalg.inv(self.C_prior))))

    @tf.function
    def matrixcompute(self, matrix1, matrix2, Cov):
        matrix1 = tf.cast(matrix1, tf.float64)
        matrix2 = tf.cast(matrix2, tf.float64)
        matrix = tf.subtract(matrix1, matrix2)
        matrix = tf.reshape(matrix, [matrix.shape[0], 1])
        matrix_T = tf.transpose(matrix)
        Cov_inv = tf.linalg.inv(Cov)
        result = tf.multiply(constant64(
            1/2), tf.matmul(tf.matmul(matrix_T, Cov_inv), matrix))
        return result

    def acceptance_gpCN(self, m_current, m_proposed):
        delta_current = tf.subtract(self.negative_log_posterior(
            self.Data, m_current), self.matrixcompute(m_current, self.MAP, self.cov_post))
        delta_proposed = tf.subtract(self.negative_log_posterior(
            self.Data, m_proposed), self.matrixcompute(m_proposed, self.MAP, self.cov_post))

        # calculate accept ratio if exp()<1
        accept_ratio = tf.exp(tf.subtract(delta_current, delta_proposed))
        acceptsample = tfd.Sample(
            tfd.Uniform(constant64(0), constant64(1)),
            sample_shape=[1, 1])
        sample = acceptsample.sample()

        if(accept_ratio > sample):
            return True
        else:
            return False

    @tf.function
    def draw_proposal(self, m_current):

        _term1 = self.MAP

        # sqrt term
        tem_1 = tf.convert_to_tensor(tf.sqrt(1-self.beta**2), dtype=tf.float64)
        # sqrt(1-beta^2)()
        _term2 = tf.multiply(tem_1, (tf.subtract(m_current, self.MAP)))

        Xi = tfd.MultivariateNormalTriL(
            loc=0,
            scale_tril=tf.linalg.cholesky(self.cov_post))

        Xi_s = tfd.Sample(Xi)
        _term3 = tf.multiply(self.beta, Xi_s.sample())

        m_proposed = tf.add(self.MAP, tf.add(_term2, _term3))

        return m_proposed

    def run_chain_hessian(self,Hess = None):

        if Hess is None:
            self.Full_Hessian()
        else: self.Hessian = Hess
        self.Laplace_appro()

        burn_in = self.number_burnin
        steps = self.number_sample
        k = 0
        accepted = []
        rejected = []
        samples = []

        m_current = self.mu_init  # init m

        for k in range(steps+burn_in):

            m_proposed = self.draw_proposal(m_current)

            if self.acceptance_gpCN(m_current, m_proposed):
                m_current = m_proposed
                if k > burn_in:
                    accepted.append(m_proposed.numpy())
                    samples.append(m_proposed.numpy())
            else:
                m_current = m_current
                rejected.append(m_proposed.numpy())
                samples.append(m_current.numpy())
        self.acceptance_rate = np.shape(accepted)[0]/self.number_sample

        return accepted, rejected, samples
