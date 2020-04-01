import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# find MAP point
@tf.function()
def gradient_decent(unnormalized_posterior_log_prob, steps=10000, learning_rate=0.001):

    mu = tf.constant([[-1., -1.]])

    for _ in tf.range(steps):
        with tf.GradientTape() as t:
            t.watch(mu)
            loss = tf.negative(unnormalized_posterior_log_prob(mu))
            dlossdmu = t.gradient(loss, mu)
            mu = mu - learning_rate*dlossdmu
    return mu


# Hessian
@tf.function
def Full_Hessian(MAP, unnormalized_posterior_log_prob):
    Hess = tf.TensorArray(tf.float32, size=2)
    j = 0
    for i in range(2):
        with tf.GradientTape() as t:
            t.watch(MAP)
            with tf.GradientTape() as tt:
                tt.watch(MAP)
                loss = -unnormalized_posterior_log_prob(MAP)
            jac = tt.gradient(loss, MAP, unconnected_gradients='zero')[0][i]
        hess = t.gradient(jac, MAP, unconnected_gradients='none')
        Hess = Hess.write(j, hess)
        j = j+1
    hessian = tf.squeeze(Hess.stack())
    return hessian


@tf.function
def matrixcompute(matrix1, matrix2, Cov):
    matrix1 = tf.squeeze(matrix1)
    matrix2 = tf.squeeze(matrix2)
    matrix = tf.subtract(matrix1, matrix2)
    matrix = tf.reshape(matrix, [matrix.shape[0], 1])
    matrix_T = tf.transpose(matrix)
    Cov_inv = tf.linalg.inv(Cov)
    result = tf.multiply(tf.constant(
        1/2), tf.matmul(tf.matmul(matrix_T, Cov_inv), matrix))
    return result


@tf.function
def negative_log_post(unnormalized_posterior_log_prob, vars):
    return tf.negative(unnormalized_posterior_log_prob(vars))


@tf.function
def acceptance_gpCN(unnormalized_posterior_log_prob, m_current, m_proposed, MAP, C_post):

    delta_current = tf.subtract(negative_log_post(
        unnormalized_posterior_log_prob, m_current), matrixcompute(m_current, MAP, C_post))
    delta_proposed = tf.subtract(negative_log_post(
        unnormalized_posterior_log_prob, m_proposed), matrixcompute(m_proposed, MAP, C_post))

    # calculate accept ratio if exp()<1
    accept_ratio = tf.exp(tf.subtract(delta_current, delta_proposed))
    acceptsample = tfd.Sample(
        tfd.Uniform(0., 1.),
        sample_shape=[1, 1])
    sample = acceptsample.sample()

    if(accept_ratio > sample):
        return True
    else:
        return False


@tf.function
def draw_proposal(m_current, MAP, C_post):

    beta = tf.constant(0.25)
    _term1 = MAP

    # sqrt term
    tem_1 = tf.convert_to_tensor(tf.sqrt(1-beta**2), dtype=tf.float32)
    # sqrt(1-beta^2)()
    _term2 = tf.multiply(tem_1, (tf.subtract(m_current, MAP)))

    Xi = tfd.MultivariateNormalTriL(
        loc=0,
        scale_tril=tf.linalg.cholesky(C_post))

    Xi_s = tfd.Sample(Xi)
    _term3 = tf.multiply(beta, Xi_s.sample())

    m_proposed = tf.add(MAP, tf.add(_term2, _term3))

    return m_proposed


def Laplace_appro(H, C_prior):
    return tf.linalg.inv((tf.add(H, tf.linalg.inv(C_prior))))


def run_chain_hessian(cov, num_results, burnin, unnormalized_posterior_log_prob):
    MAP = gradient_decent(unnormalized_posterior_log_prob)
    Hessian_matrix = Full_Hessian(MAP, unnormalized_posterior_log_prob)

    C_post = Laplace_appro(Hessian_matrix, cov)

    burn_in = burnin
    steps = num_results
    k = 0
    accepted = []
    rejected = []

    m_current = MAP  # init m

    for k in range(steps+burn_in):

        m_proposed = draw_proposal(m_current, MAP, C_post)

        if acceptance_gpCN(unnormalized_posterior_log_prob, m_current, m_proposed, MAP, C_post):
            m_current = m_proposed
            if k > burn_in:
                accepted.append(m_proposed.numpy()[0])
        else:
            m_current = m_current
            rejected.append(m_proposed.numpy()[0])

    return accepted, rejected
