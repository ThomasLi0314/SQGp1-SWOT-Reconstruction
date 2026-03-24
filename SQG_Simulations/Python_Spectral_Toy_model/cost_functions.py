import jax.numpy as jnp
from physics_functions import forward_ssh

def cost_function_fmin(phi0_s, f, kx, ky, mu, inv_mu, Bu, epsilon, K2, inv_K2, eta_s_hat_true):
    """
    Scalar cost for fminunc-style optimization (operates in physical space).
    Weighted spectral error: sum(|eta_hat_guess - eta_hat_true|^2 * (1 + K^2))
    """
    phi0_s_hat_guess = jnp.fft.fft2(phi0_s)
    eta_s_hat_guess = forward_ssh(phi0_s_hat_guess, f, kx, ky, mu, inv_mu, K2, inv_K2, Bu, epsilon)

    diff_hat = jnp.abs(eta_s_hat_guess - eta_s_hat_true)
    err_spec = 1.0 + K2
    cost = jnp.sum(diff_hat**2 * err_spec)
    return jnp.real(cost)


def cost_function_lsq(phi0_s_hat_guess, f, kx, ky, mu, inv_mu, Bu, epsilon, K2, inv_K2, eta_s_hat_true):
    """Residual vector for least-squares optimization (operates in spectral space)."""
    eta_s_hat_guess = forward_ssh(phi0_s_hat_guess, f, kx, ky, mu, inv_mu, K2, inv_K2, Bu, epsilon)
    return jnp.abs(eta_s_hat_guess - eta_s_hat_true)
