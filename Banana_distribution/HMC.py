import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

@tf.function
def run_chain_HMC(num_results,burnin,initial_chain_state,unnormalized_posterior_log_prob):
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=initial_chain_state,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            step_size = 0.02,
            num_leapfrog_steps = 3),
        num_burnin_steps=burnin,
        num_steps_between_results=1,  # Thinning.
        parallel_iterations=1)
    return samples,kernel_results