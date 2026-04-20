"""
perturbation_setup_group_C.py
-----------------------------
Extended Group C perturbations for the SQG+1 high-k noise benchmark.

Two families:
  * C2_kcut_{K}    — white noise retained only for |k| > K_cutoff
                     (sweeps over a list of cutoff wavenumbers)
  * C3_slope_{a}   — random-phase noise with a K^(-alpha) energy spectrum
                     (sweeps over a list of spectral slopes)

All perturbations are normalized via `normalize_to_rms` (imported from the
parent `perturbation_setup` module) so that rms(perturb) = Ro * rms(phi0_s).

Wavenumbers here are interpreted in the same units as the solver's `K` grid —
i.e. Shafer's non-dimensional convention where Lx = 2π and integer k is the
physical wavenumber.
"""

import sys
import os

import jax
import jax.numpy as jnp

# Reuse the normalization helper from the main perturbation module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from perturbation_setup import normalize_to_rms, _rms  # noqa: E402


# Default sweep values — override by passing explicit lists to the builder
DEFAULT_C2_CUTOFFS = [5.0, 7.5, 8.5, 9.5, 11.0, 12.5]
DEFAULT_C3_SLOPES  = [-3, -5 / 3, -2]


# C2: band-limited high-k white noise

def perturb_C2_cutoff(phi0_s, K, Ro, rng_key, K_cutoff):
    """
    White noise retained only for |k| > K_cutoff.

    Parameters
    ----------
    K_cutoff : float
        Wavenumber threshold (same units as `K`). Modes with K <= K_cutoff
        are zeroed; the rest keep i.i.d. Gaussian spectral amplitudes.
    """
    Nx, Ny = phi0_s.shape
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    noise_hat = jnp.where(K > K_cutoff, noise_hat, 0.0)
    noise_hat = noise_hat.at[0, 0].set(0.0)  # zero DC for safety
    perturb = jnp.real(jnp.fft.ifft2(noise_hat))
    return normalize_to_rms(perturb, phi0_s, Ro)


# C3: power-law energy spectrum

def perturb_C3_slope(phi0_s, K, Ro, rng_key, slope):
    """
    Random-phase noise with a K^(-slope) energy spectrum.

    The spectral amplitude is scaled so that |noise_hat(k)|^2 ∝ K^(-slope),
    then passed through `normalize_to_rms`. A larger `slope` concentrates
    energy at low k (redder spectrum); slope = 0 recovers flat white noise;
    slope = 2 matches the original C3 case.

    Parameters
    ----------
    slope : float
        Energy spectrum power-law exponent (positive means red).
    """
    Nx, Ny = phi0_s.shape
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    # amplitude ∝ K^(-slope/2), so |amp|^2 ∝ K^(-slope)
    amp = jnp.where(K > 0, K ** (-slope / 2.0), 0.0)
    noise_hat = noise_hat * amp
    noise_hat = noise_hat.at[0, 0].set(0.0)
    perturb = jnp.real(jnp.fft.ifft2(noise_hat))
    return normalize_to_rms(perturb, phi0_s, Ro)


# Builder

def build_group_C_perturbations(ssh_qg, K, Ro, rng_key,
                                 c2_cutoffs=None, c3_slopes=None,
                                 perturb_streamfunction=False):
    """
    Build a dict of Group-C perturbation cases for the benchmark.

    Parameters
    ----------
    ssh_qg : (Nx, Ny) array
        QG surface SSH / streamfunction used as the unperturbed baseline.
    K : (Nx, Ny) array
        Wavenumber magnitude grid (same units expected by the solver).
    Ro : float
        Rossby number — sets perturbation amplitude via normalize_to_rms.
    rng_key : jax PRNGKey
        Split internally into one key per case for reproducibility.
    c2_cutoffs : list[float] or None
        K_cutoff values for the C2 sweep (default: DEFAULT_C2_CUTOFFS).
    c3_slopes : list[float] or None
        Slope exponents for the C3 sweep (default: DEFAULT_C3_SLOPES).
    perturb_streamfunction : bool
        If True, target = ssh_qg + p is interpreted as the perturbed
        streamfunction ψ_true. If False (default), target = ssh_qg + p
        is the perturbed SSH observation η_true. Both are in physical space.

    Returns
    -------
    dict  {case_name: (perturbation, target)}
        Same format as build_all_perturbations in perturbation_setup.py.
    """
    if c2_cutoffs is None:
        c2_cutoffs = list(DEFAULT_C2_CUTOFFS)
    if c3_slopes is None:
        c3_slopes = list(DEFAULT_C3_SLOPES)

    n_cases = len(c2_cutoffs) + len(c3_slopes)
    keys = jax.random.split(rng_key, n_cases)

    perturbations = {}
    ki = 0

    for kc in c2_cutoffs:
        name = f"C2_kcut_{kc:g}"
        perturbations[name] = perturb_C2_cutoff(ssh_qg, K, Ro, keys[ki], kc)
        ki += 1

    for s in c3_slopes:
        name = f"C3_slope_{s:g}"
        perturbations[name] = perturb_C3_slope(ssh_qg, K, Ro, keys[ki], s)
        ki += 1

    if perturb_streamfunction:
        return {name: (p, ssh_qg + p) for name, p in perturbations.items()}
    else:
        return {name: (p, ssh_qg + p) for name, p in perturbations.items()}


def print_rms_table(cases, phi0_s, Ro):
    """RMS verification table (mirrors the one in perturbation_setup.py)."""
    signal_rms = float(_rms(phi0_s))
    target_rms = Ro * signal_rms
    print(f"  signal rms(phi0_s) = {signal_rms:.6f}")
    print(f"  target rms(perturb) = Ro * rms(phi0_s) = {target_rms:.6f}\n")
    print(f"  {'Case':<25} {'rms(perturb)':>14}  {'ratio to target':>16}")
    print("  " + "-" * 60)
    for name, (perturb, _) in cases.items():
        p_rms = float(_rms(jnp.real(perturb)))
        print(f"  {name:<25} {p_rms:>14.6f}  {p_rms / target_rms:>16.4f}")
