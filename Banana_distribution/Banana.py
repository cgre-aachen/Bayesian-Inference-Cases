# define the 2D banana-shape distribution

#import library
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class Banana_dist:

    def __init__(self, mu=[0., 0.], cov=[[1,  0.], [0.,  1]]):

        # initiate the model

        # prior
        self.mu = mu
        self.cov = cov

        self.c = 0  # mean of observations
        self.N = 100  # number of observation data
        self.sigma2y = 1  # standard deviation of observation data
        # generate the observation data
        np.random.seed(121)
        self.y_ = np.random.normal(loc=self.c, scale=self.sigma2y, size=self.N)
        self.D = tf.convert_to_tensor(self.y_, dtype=tf.float32)

    @tf.function
    def joint_log_post(self, theta):
        """Calculate the joint posterior of a given point

        Arguments:
            theta {[tensor:float32]} -- [2xn] tensor! eg: tf.constant([[1.,1.]])

        Returns:
            [tensor] -- value of the posterior
        """
        # define random variables prior

        D_n = tf.reshape(self.D, [self.D.shape[0], 1])
        D_n = tf.tile(D_n, [1, theta.shape[0]])

        mvn = tfd.MultivariateNormalTriL(
            loc=self.mu,
            scale_tril=tf.linalg.cholesky(self.cov))

        # define likelihood
        y = tfd.Normal(loc=tf.add(theta[:, 1], tf.pow(
            theta[:, 0], 2.)), scale=self.sigma2y)
        # return the posterior probability
        return(mvn.log_prob(tf.squeeze(theta))
               + tf.reduce_sum(y.log_prob(D_n), axis=0))


# calculate the posterior density

    def full_post(self):
        self.x_1, self.y_1 = np.mgrid[-2:2:.03, -2:2:.03]
        pos = np.empty(self.x_1.shape + (2,), dtype=np.float32)
        pos[:, :, 0] = self.x_1
        pos[:, :, 1] = self.y_1
        pos = tf.convert_to_tensor(pos)
        post = np.empty(self.x_1.shape)
        for i in range(np.arange(-2, 2, .03).shape[0]):
            for j in range(np.arange(-2, 2, .03).shape[0]):
                post[i][j] = self.joint_log_post(
                    tf.convert_to_tensor([pos[i][j]]))
        return post

    def draw_post(self, post=None, title=None):
        Min = tf.constant([[-0., -0.2]])
        Max = tf.constant([[0.01, 0.04]])
        levels = np.arange(self.joint_log_post(Min), self.joint_log_post(
            Max), (self.joint_log_post(Max) - self.joint_log_post(Min))/50)

        if post is None:
            post = self.full_post()
        plt.contourf(self.x_1, self.y_1, post, levels=levels, alpha=0.7)

        if title is not None:
            plt.title(title)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-2, 0.5)
        plt.xlabel("x1", fontsize=15)
        plt.ylabel("x2", fontsize=15)
