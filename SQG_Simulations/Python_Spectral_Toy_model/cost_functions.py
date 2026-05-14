import jax.numpy as jnp
from physics_functions import forward_ssh

def cost_function_fmin(phi0_s, kx, ky, mu, inv_mu, Bu, Ro_sys, K2, inv_K2, eta_s_hat_true):
    """
    Scalar cost for fminunc-style optimization (operates in physical space).
    Unweighted spectral L2 error: sum(|eta_hat_guess - eta_hat_true|^2).
    """
    phi0_s_hat_guess = jnp.fft.fft2(phi0_s)
    eta_s_hat_guess = forward_ssh(phi0_s_hat_guess, kx, ky, mu, inv_mu, K2, inv_K2, Bu, Ro_sys)

    diff_hat = jnp.abs(eta_s_hat_guess - eta_s_hat_true)
    # err_spec = 1.0 + inv_K2
    # cost = jnp.sum(diff_hat**2 * err_spec)
    cost = jnp.sum(diff_hat ** 2)
    return jnp.real(cost) / diff_hat.size


def cost_function_fmin_gls(phi0_s, kx, ky, mu, inv_mu, Bu, Ro_sys, K2, inv_K2,
                           eta_s_hat_true, weight, filter_lp):
    """GLS spectral cost with LP-projection at BOTH ends of the forward map.

        L = (1/N) * sum_{k} |LP{eta_hat_guess(k)} - eta_hat_true(k)|^2 * weight(k)

    Why LP both ends:
      The SQG+1 forward map  eta = phi0 + eps * p1(phi0)  is nonlinear, so
      products of derivatives of LP{phi0} produce content in [k_c, 2 k_c].
      The target is strictly band-limited to [0, k_c], so without an output
      LP-filter the residual is dominated by the unfittable nonlinear-spread
      content above k_c -- the optimizer can't beat it (and the GLS weight
      downweights it, so it doesn't try).  Projecting eta_pred onto the
      same band as the target turns the data-fidelity term into a clean
      LP -> LP map and the loss measures the actual model error within the
      observable band.

    Parameters
    ----------
    weight : (Nx, Ny) array
        Per-mode spectral weight 1/S_n(k). For unweighted L2, pass ones.
        For Lorentzian / Butterworth-style noise model:
            S_n(k) = 1 + (K/K_c)^p   ->   weight = 1 / S_n
    filter_lp : (Nx, Ny) array
        Low-pass mask. Applied to phi0_hat at the top of forward_ssh
        (band-limited unknown) AND to eta_hat_guess at the bottom
        (band-limited prediction).
    """
    phi0_s_hat_guess = jnp.fft.fft2(phi0_s)
    eta_s_hat_guess = forward_ssh(
        phi0_s_hat_guess, kx, ky, mu, inv_mu, K2, inv_K2, Bu, Ro_sys,
        filter_lp=filter_lp,
    )
    # LP-project the prediction so the residual is measured in the same
    # band as the (LP-filtered) target.
    eta_s_hat_guess = eta_s_hat_guess * filter_lp
    diff_hat = eta_s_hat_guess - eta_s_hat_true
    cost = jnp.sum(jnp.abs(diff_hat) ** 2 * weight)
    return jnp.real(cost) / diff_hat.size