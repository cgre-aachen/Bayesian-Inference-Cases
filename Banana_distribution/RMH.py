import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# adjust the step size of Random walk Matroplis Hasting MCMC


def gauss_new_state_fn(scale, dtype):
    gauss = tfd.Normal(loc=dtype(0), scale=dtype(scale))

    def _fn(state_parts, seed):
        next_state_parts = []
        seed_stream = tfp.util.SeedStream(seed, salt='RandomNormal')
        for sp in state_parts:
            next_state_parts.append(sp + gauss.sample(
                sample_shape=sp.shape, seed=seed_stream()))
        return next_state_parts
    return _fn


@tf.function
def run_chain_RMH(scale, num_results, burnin, initial_chain_state, unnormalized_posterior_log_prob):
    dtype = np.float32
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=burnin,
        current_state=initial_chain_state,
        kernel=tfp.mcmc.RandomWalkMetropolis(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            new_state_fn=gauss_new_state_fn(scale=scale, dtype=dtype),
            seed=42))  # For determinism.

    return samples, kernel_results
