import sys
import os
import jax
import jax.numpy as jnp

# Reuse the normalization helper from the v2 perturbation module
sys.path.insert(0, os.path.dirname(__file__))
from perturbation_setup_v3 import normalize_to_rms, _rms  # noqa: E402


# Default sweep — the three slopes requested for this study
DEFAULT_SLOPES = [-3.0, -5.0 / 3.0, -2.0]


# Builder

def build_C3_perturbations(eta_tar, K, Ro, rng_key, slopes=None):
    if slopes is None:
        slopes = list(DEFAULT_SLOPES)

    Nx, Ny = eta_tar.shape

    # ── Generate ONE white-noise field; reuse across all slopes ──
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    noise_hat = noise_hat.at[0, 0].set(0.0)   # zero DC

    perturbations = {}
    for s in slopes:
        # amplitude ∝ K^(slope/2), so |amp|^2 ∝ K^(slope)
        amp = jnp.where(K > 0, K ** (s / 2.0), 0.0)
        shaped_hat = noise_hat * amp
        shaped_hat = shaped_hat.at[0, 0].set(0.0)
        perturb = jnp.real(jnp.fft.ifft2(shaped_hat))
        perturb = normalize_to_rms(perturb, eta_tar, Ro)
        perturbations[f"C3_slope_{s:g}"] = perturb

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
