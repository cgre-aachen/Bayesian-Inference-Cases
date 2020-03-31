import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math as m

from GaussianProcess import GaussianProcess2Dlayer


tfd = tfp.distributions


def constant64(i):
    return tf.constant(i, dtype=tf.float64)


def divide(a, b):
    """Tensorflow save divide

    Arguments:
        a {[Tensor]} -- [description]
        b {[Tensor]} -- [description]

    Returns:
        [Tensor] -- 
    """
    return tf.math.divide_no_nan(a, b)


pi = constant64(m.pi)


class Gravity_Polygon(tf.Module):
    def __init__(self, obs_N, Range, rho, thickness, Number_para):
        super(Gravity_Polygon, self).__init__()
        self.obs_N = obs_N
        self.Range = Range
        self.Number_para = Number_para
        self.rho = constant64(rho)        # density difference   kg/m^3
        self.x_obv = tf.linspace(constant64(-70.), constant64(70.), self.obs_N)
        self.y_obv = tf.zeros(tf.shape(self.x_obv), dtype=tf.float64)
        self.number_of_fixpoints = 10
        self.depth = constant64(-50)
        self.thickness = thickness
        # set some points out of the model area to eliminate the boundary effect

        self.gp = GaussianProcess2Dlayer(
            self.Range, self.depth, self.Number_para, self.thickness)

    def A(self, x1, z1, x2, z2):
        numerator = (x2-x1)*(x1*z2-x2*z1)
        denominator = (x2-x1)**2 + (z2-z1)**2
        return divide(numerator, denominator)

    def B(self, x1, z1, x2, z2):
        '''
        x : array, x coordinate
        z : array, z coordinate
        p1, p2 : int, position

        '''
        return divide((z1-z2), (x2-x1))

    def theta_new(self, xn, zn):

        m = tf.atan(divide(zn, xn))

        m = tf.where(m < 0, m + pi, m)

        m = tf.where(m == 0, m + pi/2, m)

        return m

    def Z_new(self, x1, z1, x2, z2):

        # let's do not allow 1) points at origin
        # 2) two points in a sequence have the same x coordinate

        theta1 = self.theta_new(x1, z1)
        theta2 = self.theta_new(x2, z2)

        r1 = (tf.sqrt(x1**2.+z1**2.))
        r2 = (tf.sqrt(x2**2.+z2**2.))

        _A = self.A(x1, z1, x2, z2)
        _B = self.B(x1, z1, x2, z2)

        Z_result = _A*((theta1-theta2)+_B*tf.math.log(divide(r1, r2)))

        return Z_result

    @tf.function
    def calculate_gravity(self, x, z):

        x_obv = tf.linspace(self.Range[0], self.Range[1], self.obs_N)
        y_obv = tf.zeros(tf.shape(x_obv), dtype=tf.float64)

        tx = tf.transpose(tf.tile(x, [1, self.obs_N]))

        x_tile = tf.expand_dims(tx, axis=2)

        tz = tf.transpose(tf.tile(z, [1, self.obs_N]))

        z_tile = tf.expand_dims(tz, axis=2)

        x_obv_tile = tf.expand_dims(x_obv, axis=1)
        x_obv_tile = tf.tile(x_obv_tile, [1, x.shape[0]])
        x_obv_tile = tf.reshape(x_obv_tile, [self.obs_N, x.shape[0], 1])

        z_obv_tile = tf.expand_dims(y_obv, axis=1)
        z_obv_tile = tf.tile(z_obv_tile, [1, x.shape[0]])
        z_obv_tile = tf.reshape(z_obv_tile, [self.obs_N, x.shape[0], 1])

        _x = x_tile - x_obv_tile
        _z = z_tile - z_obv_tile

        x_ = tf.concat(
            [_x, tf.reshape(_x[:, 0], shape=[self.obs_N, 1, 1])], axis=1)
        z_ = tf.concat(
            [_z, tf.reshape(_z[:, 0], shape=[self.obs_N, 1, 1])], axis=1)

        x1 = x_[:, 0:-1]
        x2 = x_[:, 1:]
        z1 = z_[:, 0:-1]
        z2 = z_[:, 1:]

        # gravitational constant  m^3 kg ^-1 s^-2
        G = constant64(6.67 * 10**(-11))
        gravity = 2*G*self.rho * \
            tf.reduce_sum(-self.Z_new(x1, z1, x2, z2), axis=1)

        # print('tracing')
        # tf.print('executing')

        return tf.squeeze(gravity)

    def set_prior(self, mu_prior, cov_prior, cov):
        self.mu_prior = mu_prior
        self.cov_prior = cov_prior
        # likelihood covariance
        self.cov = cov

    @tf.function
    def joint_log_post(self, Data, _control_position):
        """[summary]

        Arguments:
            Data {[Tensor]} -- [description]
            _control_position {[Tensor]} -- [description]

        Returns:
            [type] -- [description]
        """
        self.Data = Data
        # define random variables prior

        mvn_prior = tfd.MultivariateNormalTriL(
            loc=self.mu_prior,
            scale_tril=tf.linalg.cholesky(self.cov_prior))
        # define likelihood

        _control_index = tf.linspace(
            self.Range[0], self.Range[1], _control_position.shape[0])

        __x, __z = self.gp.GaussianProcess(_control_index, _control_position)

        Gm_ = self.calculate_gravity(__x, __z)

        mvn_likelihood = tfd.MultivariateNormalTriL(
            loc=Gm_,
            scale_tril=tf.linalg.cholesky(self.cov))

        # return the posterior probability
        return (mvn_prior.log_prob(_control_position)
                + mvn_likelihood.log_prob(Data))

    @tf.function
    def negative_log_posterior(self, Data, _control_position):
        return -self.joint_log_post(Data, _control_position)
