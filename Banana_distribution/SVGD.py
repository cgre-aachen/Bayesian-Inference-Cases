import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

dtype = tf.float32
num_particles = 250
num_latent = 2
lr = 0.003
alpha = .9
fudge_factor = 1e-6
num_iter = 10000
range_limit = [-3, 3]
npoints_plot = 50

z_np = tf.convert_to_tensor(-np.random.randn(num_particles,
                                             num_latent)*2.0, dtype=tf.float32)


def svgd_kernel(X0):
    XY = tf.matmul(X0, tf.transpose(X0))
    X2_ = tf.reduce_sum(tf.square(X0), axis=1)

    x2 = tf.reshape(X2_, shape=(tf.shape(X0)[0], 1))

    X2e = tf.tile(x2, [1, tf.shape(X0)[0]])
    ## (x1 -x2)^2 + (y1 -y2)^2
    H = tf.subtract(tf.add(X2e, tf.transpose(X2e)), 2 * XY)

    V = tf.reshape(H, [-1, 1])

    # median distance
    def get_median(v):
        v = tf.reshape(v, [-1])
        m = v.get_shape()[0]//2
        return tf.nn.top_k(v, m).values[m-1]
    h = get_median(V)
    h = tf.sqrt(
        0.5 * h / tf.math.log(tf.cast(tf.shape(X0)[0], tf.float32) + 1.0))

    # compute the rbf kernel
    Kxy = tf.exp(-H / h ** 2 / 2.0)

    dxkxy = tf.negative(tf.matmul(Kxy, X0))
    sumkxy = tf.expand_dims(tf.reduce_sum(Kxy, axis=1), 1)
    dxkxy = tf.add(dxkxy, tf.multiply(X0, sumkxy)) / (h ** 2)

    return (Kxy, dxkxy)


def get_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0]//2
    return tf.nn.top_k(v, m).values[m-1]


@tf.function
def gradient(z_np, unnormalized_posterior_log_prob):
    with tf.GradientTape() as t:
        t.watch(z_np)
        f = unnormalized_posterior_log_prob(z_np)
    log_p_grad = tf.squeeze(t.gradient(f, z_np))
    return log_p_grad


def run_svgd(unnormalized_posterior_log_prob):
    global z_np
    for _ in range(num_iter):
        log_p_grad = gradient(z_np, unnormalized_posterior_log_prob)
        kernel_matrix, kernel_gradients = svgd_kernel(z_np)
        grad_theta = (tf.matmul(kernel_matrix, log_p_grad) +
                      kernel_gradients)/num_particles
        z_np = z_np+lr*grad_theta
    return z_np
