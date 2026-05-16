"""
perturbation_setup_v4.py
------------------------
Generates O(epsilon) perturbations to the SQG streamfunction (SSH) for
benchmarking the SQG+1 inversion under different kinds of ageostrophic signal.

All perturbations are normalized so that:
    rms(perturbation) = epsilon * rms(eta_tar)
which equalises the initial cost function value across cases, ensuring the
benchmark isolates the *structure* of the perturbation rather than its
amplitude.

Currently active perturbations
------------------------------
- A1 "nothing"          : zero perturbation (control / sanity check).
- E3 "swot_aliased"     : SWOT-like noise reproducing the Du et al. (2026)
                          processing artefact — high-resolution (250-m) white
                          noise convolved with a 17-point 2D Hamming filter,
                          then subsampled (aliased) onto the 2-km posting
                          grid. This generates a red along-track spectrum
                          purely as a side effect of filtering+aliasing,
                          even though the underlying noise is white.

Usage
-----
    from perturbation_setup_v4 import build_all_perturbations, print_rms_table

    cases = build_all_perturbations(
        eta_tar, eta_tar_hat, kx, ky, K, K2, inv_K2,
        mu, inv_mu, Bu, epsilon, X, Y, Lx, Ly, rng_key
    )
    # cases: {name: (perturbation, eta_obs)}
    #        both in physical space, eta_obs = eta_tar + perturbation
    print_rms_table(cases, eta_tar, epsilon)
"""

import jax
import jax.numpy as jnp


# ── Normalization helper ──────────────────────────────────────────────────────

def _rms(field):
    return jnp.sqrt(jnp.mean(field ** 2))


def normalize_to_rms(perturb, eta_tar, epsilon):
    """
    Rescale `perturb` so that rms(perturb) = epsilon * rms(eta_tar).
    Also removes the spatial mean to avoid a DC offset in eta_obs.
    """
    perturb = perturb - jnp.mean(perturb)
    target = epsilon * _rms(eta_tar)
    current = _rms(perturb)
    return jnp.where(current > 0, perturb * (target / current), perturb)


# ── Group A ───────────────────────────────────────────────────────────────────

def perturb_A1(eta_tar, eta_tar_hat, kx, ky, epsilon):
    """
    A.1  Zero perturbation (control case).
    """
    perturb = jnp.zeros_like(eta_tar)
    return normalize_to_rms(perturb, eta_tar, epsilon)


# ── Group E: SWOT-like instrument & geometry errors ──────────────────────────

def perturb_E3_swot_aliased(eta_tar, epsilon, rng_key, M=8, window_size=17):
    """
    E.3  SWOT 2-km-posting aliased-white-noise artefact.

    Reproduces the processing pipeline that Du et al. (2026) identified as
    the cause of the "red" along-track spectrum seen in SWOT 2-km data:

        1. White Gaussian noise on the native 250-m grid (M = 8x finer
           than the 2-km posting grid).
        2. 2-D 17-point Hamming smoothing on the 250-m field.
        3. Subsampling (= aliasing) every M-th sample onto the 2-km grid.

    Even though the input is white, steps (2)+(3) bend the apparent spectrum
    on the coarse grid into a power-law shape, *purely as a filtering /
    aliasing artefact* — no real geophysical structure required.

    All operations are JAX-compatible: the 2-D convolution is implemented
    spectrally as a multiplication of FFTs, which is also exact for the
    doubly-periodic toy domain.

    Parameters
    ----------
    eta_tar     : (Nx, Ny) array — the 2-km posting target field; sets the
                  output shape and the rms scale used for normalization.
    epsilon     : scalar — sets target rms = epsilon * rms(eta_tar).
    rng_key     : jax.random PRNGKey.
    M           : int (default 8). Upsampling factor 2 km / 250 m = 8.
    window_size : int (default 17). 1-D Hamming length; the 2-D filter
                  is the outer product of two such windows.

    Returns
    -------
    perturb : (Nx, Ny) array, mean-zero, rms = epsilon * rms(eta_tar).
    """
    Nx, Ny = eta_tar.shape
    Nx_hi, Ny_hi = Nx * M, Ny * M

    # 1. High-resolution white Gaussian noise (250-m grid).
    noise_hi = jax.random.normal(rng_key, (Nx_hi, Ny_hi))

    # 2. 2-D Hamming filter 
    h1d = jnp.hamming(window_size) # Build in hamming window function
    filter_2d = jnp.outer(h1d, h1d)
    filter_2d = filter_2d / jnp.sum(filter_2d)

    # 3. Embed the filter in a (Nx_hi, Ny_hi) zero array and roll so its
    #    centre sits at index (0, 0). This makes the FFT-based convolution
    #    zero-phase, i.e. it smooths without spatially shifting the field.
    half = window_size // 2
    filter_padded = jnp.zeros((Nx_hi, Ny_hi))
    filter_padded = filter_padded.at[:window_size, :window_size].set(filter_2d)
    filter_padded = jnp.roll(filter_padded, shift=(-half, -half), axis=(0, 1))

    # 4. Periodic (circular) convolution via FFT.
    filter_hat = jnp.fft.fft2(filter_padded)
    noise_hat = jnp.fft.fft2(noise_hi)
    filtered_hi = jnp.real(jnp.fft.ifft2(noise_hat * filter_hat))

    # 5. Subsample every M-th point in each direction → 2-km grid (aliasing).
    perturb = filtered_hi[::M, ::M]

    # 6. Normalize to the benchmark amplitude convention.
    return normalize_to_rms(perturb, eta_tar, epsilon)


# ── Driver ───────────────────────────────────────────────────────────────────

def build_all_perturbations(eta_tar, eta_tar_hat, kx, ky, K, K2, inv_K2,
                            mu, inv_mu, Bu, epsilon, X, Y, Lx, Ly, rng_key):
    swot_key, = jax.random.split(rng_key, 1)

    perturbations = {
        "A1_nothing":      perturb_A1(
                               eta_tar, eta_tar_hat, kx, ky, epsilon),
        "E3_swot_aliased": perturb_E3_swot_aliased(
                               eta_tar, epsilon, swot_key),
    }
    # eta_obs = eta_tar + perturbation, both in physical space.
    return {
        name: (p, eta_tar + p)
        for name, p in perturbations.items()
    }


def print_rms_table(cases, eta_tar, epsilon):
    """
    Print a verification table showing all perturbations have the same RMS.
    """
    signal_rms = float(_rms(eta_tar))
    target_rms = epsilon * signal_rms
    print(f"  signal rms(eta_tar)  = {signal_rms:.6f}")
    print(f"  target rms(perturb) = epsilon * rms(eta_tar) = {target_rms:.6f}\n")
    print(f"  {'Case':<25} {'rms(perturb)':>14}  {'ratio to target':>16}")
    print("  " + "-" * 60)
    for name, (perturb, _) in cases.items():
        p_rms = float(_rms(jnp.real(perturb)))
        print(f"  {name:<25} {p_rms:>14.6f}  {p_rms / target_rms:>16.4f}")
