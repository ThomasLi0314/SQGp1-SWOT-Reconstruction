"""
perturbation_setup_C2.py
------------------------
Group-C2 sweep: band-limited high-k white noise with several K_cutoff
threshold values. All variants share the standard normalization
    rms(perturb) = Ro * rms(eta_tar)
so cases differ only in the spectral location of the noise band.

Convention
----------
A mode at wavevector k is *kept* if  K(k) > K_cutoff,  i.e. only modes
strictly above the cutoff carry energy. Larger K_cutoff means noise lives
at higher wavenumber and farther from the SQG signal band (peak at K0=9.5
in Shafer units). DC is always zeroed.

Usage
-----
    from perturbation_setup_C1 import build_C1_perturbations, print_rms_table

    cases = build_C2_perturbations(
        eta_tar, K, Ro, rng_key,
        K_cutoffs=[5.0, 7.5, 10.0, 12.5, 15.0, 20.0],
    )
    # cases: {name: (perturbation, eta_obs)} where eta_obs = eta_tar + perturbation
"""

import sys
import os
import jax
import jax.numpy as jnp

# Reuse the normalization helper from the v2 perturbation module
sys.path.insert(0, os.path.dirname(__file__))
from perturbation_setup_v3 import normalize_to_rms, _rms  # noqa: E402


# Default sweep — six cutoffs spanning below to well above the SQG peak K0=9.5
DEFAULT_K_CUTOFFS = [5.0, 6.0, 7.0,   8.0, 9.0, 10.0, 11.0, 12.0]
# DEFAULT_K_CUTOFFS = [10.0]


# # Single C2 perturbation

# def perturb_C2_cutoff(eta_tar, K, Ro, rng_key, K_cutoff):
#     """
#     White noise retained only for K > K_cutoff.

#     Parameters
#     ----------
#     eta_tar  : (Nx, Ny) baseline field used for RMS normalization
#     K        : (Nx, Ny) wavenumber magnitude grid (same units as K_cutoff)
#     Ro       : Rossby number — sets perturbation amplitude
#     rng_key  : jax PRNGKey
#     K_cutoff : float threshold; modes with K <= K_cutoff are zeroed. Only keep the high k mode (small scale)
#     """
#     Nx, Ny = eta_tar.shape
#     noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
#     noise_hat = jnp.where(K < K_cutoff, noise_hat, 0.0)
#     noise_hat = noise_hat.at[0, 0].set(0.0)   # zero DC
#     perturb = jnp.real(jnp.fft.ifft2(noise_hat))
#     return normalize_to_rms(perturb, eta_tar, Ro)


# Builder

def build_C1_perturbations(eta_tar, K, Ro, rng_key, K_cutoffs=None):
    """
    Build C1 cases sharing a single white-noise realization,
    differing only by the K_cutoff applied to the same spectrum.
    """
    if K_cutoffs is None:
        K_cutoffs = list(DEFAULT_K_CUTOFFS)

    Nx, Ny = eta_tar.shape

    # ── Generate ONE white-noise field; reuse across all cutoffs ──
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    noise_hat = noise_hat.at[0, 0].set(0.0)   # zero DC

    perturbations = {}
    for kc in K_cutoffs:
        # Filter the SHARED noise spectrum, then go back to physical space
        filtered_hat = jnp.where(K >= kc, noise_hat, 0.0)
        perturb      = jnp.real(jnp.fft.ifft2(filtered_hat))
        perturb      = normalize_to_rms(perturb, eta_tar, Ro)
        perturbations[f"C2_kcut_{kc:g}"] = perturb

    return {name: (p, eta_tar + p) for name, p in perturbations.items()}



def print_rms_table(cases, eta_tar, Ro):
    """RMS verification table — every case should show ratio = 1.0000."""
    signal_rms = float(_rms(eta_tar))
    target_rms = Ro * signal_rms
    print(f"  signal rms(eta_tar)   = {signal_rms:.6f}")
    print(f"  target rms(perturb)   = Ro * rms(eta_tar) = {target_rms:.6f}\n")
    print(f"  {'Case':<25} {'rms(perturb)':>14}  {'ratio to target':>16}")
    print("  " + "-" * 60)
    for name, (perturb, _) in cases.items():
        p_rms = float(_rms(jnp.real(perturb)))
        print(f"  {name:<25} {p_rms:>14.6f}  {p_rms / target_rms:>16.4f}")
