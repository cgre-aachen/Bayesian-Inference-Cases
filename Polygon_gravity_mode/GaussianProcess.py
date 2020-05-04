import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


class GaussianProcess2Dlayer:

    def __init__(self, Range, depth, Number_para, thickness=None, number_of_fixpoints=10):
        """[summary]

        Arguments:
            Range {[type]} -- [description]
            depth {[type]} -- [description]
            Number_para {[type]} -- [description]
            amplitude -- Signal variance σ2

        Keyword Arguments:
            thickness {[type]} -- [description] (default: {None})
            number_of_fixpoints {int} -- [description] (default: {10})
        """
        self.number_of_fixpoints = number_of_fixpoints
        self.Range = Range
        self.depth = depth
        self.Number_para = Number_para
        self.thickness = thickness
        self.fix_point1 = tf.transpose(tf.stack([tf.linspace(
            self.Range[0] - 200, self.Range[0] - 10, self.number_of_fixpoints), self.depth * tf.ones(self.number_of_fixpoints, dtype=tf.float64)]))
        self.fix_point2 = tf.transpose(tf.stack([tf.linspace(
            self.Range[1] + 10, self.Range[1] + 200, self.number_of_fixpoints), self.depth * tf.ones(self.number_of_fixpoints, dtype=tf.float64)]))

        self.amplitude = tfp.util.TransformedVariable(
            2, tfb.Exp(), dtype=tf.float64, name='amplitude')
        self.length_scale = tfp.util.TransformedVariable(
            50, tfb.Exp(), dtype=tf.float64, name='length_scale')
        self.kernel = psd_kernels.ExponentiatedQuadratic(
            self.amplitude, self.length_scale)

        self.observation_noise_variance = tfp.util.TransformedVariable(
            np.exp(-.1), tfb.Exp(), dtype=tf.float64, name='observation_noise_variance')
        # x- index used to construct GP model

    def GaussianProcess(self, _control_index, _control_position, resolution=5):
        '''
        Arguments:
            kernel: trained GP kernal
            k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2))

        return:
            GP interpolated model index and model psition

        '''
        # define how many points interpolated between control points
        self.resolution = resolution

        points = tf.stack([_control_index, _control_position], axis=-1)

        points = tf.concat(
            [tf.concat([self.fix_point1, points], axis=0), self.fix_point2], axis=0)

        observation_index_points = tf.reshape(
            points[:, 0], [self.Number_para + 2 * self.number_of_fixpoints, 1])

        # x-index where we want to interpolate
        model_index = tf.expand_dims(tf.linspace(
            observation_index_points[0, 0], observation_index_points[-1, 0], self.resolution * self.Number_para + 4), axis=1)

        gprm = tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=model_index,
            observation_index_points=observation_index_points,
            observations=points[:, 1],
            observation_noise_variance=self.observation_noise_variance)

        model_position = gprm.mean()

        # model_complete contains the extended polygon used to do gravity calculation
        model_position_complete = tf.reshape([tf.concat([model_position - self.thickness,
                                                         tf.reverse(model_position, axis=[-1])], axis=-1)],
                                             shape=[model_position.shape[0] * 2, 1])

        model_index_complete = tf.concat(
            [model_index, tf.reverse(model_index, axis=[0])], axis=0)

        return model_index_complete, model_position_complete
