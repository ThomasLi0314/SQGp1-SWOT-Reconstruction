"""Spectral skill-score helpers for SQG+1 inversion diagnostics.

Functions
---------
azimuthal_sum(field_2d, K, dk)
    Per-ring sum (binned by integer wavenumber).
energy_spectral_skill(phi0_opt_hat, phi0_s_hat, ...)
    Ross et al. 2023 eq. 10 skill score on the SQG+1 surface KE.
scalar_spectral_skill(field_pred, field_true, K, dk, Nx, Ny)
    Same skill score for a generic 2D scalar (e.g. surface vorticity).
"""
import numpy as np
import jax.numpy as jnp

from physics_functions import calculate_surface_u


def azimuthal_sum(field_2d, K, dk):
    """Per-ring sum of a real-valued 2D spectral field.

    Use on |x_hat|**2 to get WPSD(x) (per-ring summed power).
    """
    K_np = np.array(K)
    field_np = np.array(np.real(field_2d))
    k_int = np.round(K_np / dk).astype(int)
    k_max = int(k_int.max())
    spectrum = np.zeros(k_max + 1)
    np.add.at(spectrum, k_int.ravel(), field_np.ravel())
    return np.arange(k_max + 1), spectrum


def energy_spectral_skill(phi0_opt_hat, phi0_s_hat, mu, inv_mu, kx, ky,
                          K, K2, inv_K2, Ro_sys, Bu, Nx, Ny, dk):
    """WPSD skill score on the SQG+1 surface KE (Ross et al. 2023, eq. 10).

        WPSD_S(k) = 1 - WPSD(u_pred - u_true) / WPSD(u_true)

    Both u_pred and u_true go through the same toy operator, so a perfect
    recovery (phi0_opt_hat == phi0_s_hat) gives WPSD_S == 1.

    Returns
    -------
    k_axis, E_true, E_diff, WPSD_S : 1D arrays.
    """
    u_true, v_true = calculate_surface_u(phi0_s_hat, mu, inv_mu,
                                         kx, ky, K2, inv_K2, Ro_sys, Bu)
    u_opt,  v_opt  = calculate_surface_u(phi0_opt_hat, mu, inv_mu,
                                         kx, ky, K2, inv_K2, Ro_sys, Bu)

    u_true_hat = jnp.fft.fft2(u_true)
    v_true_hat = jnp.fft.fft2(v_true)
    u_opt_hat  = jnp.fft.fft2(u_opt)
    v_opt_hat  = jnp.fft.fft2(v_opt)

    norm = (Nx * Ny) ** 2
    e_true_2d = 0.5 * (jnp.abs(u_true_hat) ** 2 + jnp.abs(v_true_hat) ** 2) / norm
    e_diff_2d = 0.5 * (jnp.abs(u_opt_hat - u_true_hat) ** 2
                       + jnp.abs(v_opt_hat - v_true_hat) ** 2) / norm

    k_axis, E_true = azimuthal_sum(e_true_2d, K, dk)
    _,      E_diff = azimuthal_sum(e_diff_2d, K, dk)

    with np.errstate(invalid='ignore', divide='ignore'):
        rel_E = np.where(E_true > 1e-30, E_diff / E_true, np.nan)
    return k_axis, E_true, E_diff, 1.0 - rel_E


def scalar_spectral_skill(field_pred, field_true, K, dk, Nx, Ny):
    """WPSD skill score on a generic real-valued 2D scalar field.

        WPSD_S(k) = 1 - WPSD(field_pred - field_true) / WPSD(field_true)

    Use this for surface vorticity / SSH / buoyancy comparisons where the
    'truth' is given directly (not derived from an inversion variable).

    Returns
    -------
    k_axis, S_true, S_diff, WPSD_S : 1D arrays.
    """
    pred_hat = jnp.fft.fft2(field_pred)
    true_hat = jnp.fft.fft2(field_true)
    norm = (Nx * Ny) ** 2
    s_true_2d = jnp.abs(true_hat) ** 2 / norm
    s_diff_2d = jnp.abs(pred_hat - true_hat) ** 2 / norm

    k_axis, S_true = azimuthal_sum(s_true_2d, K, dk)
    _,      S_diff = azimuthal_sum(s_diff_2d, K, dk)

    with np.errstate(invalid='ignore', divide='ignore'):
        rel_S = np.where(S_true > 1e-30, S_diff / S_true, np.nan)
    return k_axis, S_true, S_diff, 1.0 - rel_S
