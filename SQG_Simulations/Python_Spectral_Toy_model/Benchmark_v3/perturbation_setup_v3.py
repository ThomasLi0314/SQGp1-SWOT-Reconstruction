"""
perturbation_setup.py
---------------------
Generates 10 O(epsilon) perturbations to the SQG streamfunction (SSH) for
benchmarking the SQG+1 inversion under different kinds of ageostrophic signal.

All perturbations are normalized so that:
    rms(perturbation) = epsilon * rms(phi0_s)
which equalises the initial cost function value across cases, ensuring the
benchmark isolates the *structure* of the perturbation rather than its amplitude.


Usage
-----
    from perturbation_setup import build_all_perturbations, print_rms_table

    cases = build_all_perturbations(
        phi0_s, phi0_s_hat, kx, ky, K, K2, inv_K2,
        mu, inv_mu, Bu, epsilon, X, Y, Lx, Ly, rng_key
    )
    # cases: {name: (perturbation, psi_true)}  — both in physical space, psi_true = phi0_s + perturbation
    print_rms_table(cases, phi0_s, epsilon)
"""

import sys
import os
import jax
import jax.numpy as jnp

# Allow importing physics_functions from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from physics_functions import cyclogeo_term, vorticity_term


# ── Normalization helper ───────────────────────────────────────────────────────

def _rms(field):
    return jnp.sqrt(jnp.mean(field ** 2))


def normalize_to_rms(perturb, phi0_s, epsilon):
    """
    Rescale `perturb` so that rms(perturb) = epsilon * rms(phi0_s).
    Also removes the spatial mean to avoid a DC offset in eta_true.
    """
    perturb = perturb - jnp.mean(perturb)
    target = epsilon * _rms(phi0_s)
    current = _rms(perturb)
    return jnp.where(current > 0, perturb * (target / current), perturb)


# Group A

def perturb_A1(eta_tar, eta_tar_hat, kx, ky, epsilon):
    """
    A nothing perturbation for testing
    """
    perturb = jnp.zeros_like(eta_tar)
    return normalize_to_rms(perturb, eta_tar, epsilon)


def perturb_A2(eta_tar, eta_tar_hat, kx, ky, epsilon):
    """
    A.2  Frontogenesis-correlated perturbation.

    Proportional to |∇ψ|², which is large at fronts and strain regions.
    Spatially correlated with the SQG field but not identical to it.

        perturbation ∝ (∂ψ/∂x)² + (∂ψ/∂y)²
    """
    psi_x = jnp.real(jnp.fft.ifft2(eta_tar_hat * 1j * kx))
    psi_y = jnp.real(jnp.fft.ifft2(eta_tar_hat * 1j * ky))
    perturb = psi_x ** 2 + psi_y ** 2
    return normalize_to_rms(perturb, eta_tar, epsilon)


def perturb_A3(eta_tar, eta_tar_hat, K2, epsilon):
    """
    A.3  Vorticity-aligned perturbation.

    Proportional to the SQG surface vorticity ζ = −∇²ψ.
    In spectral space: ζ̂ = −K² ψ̂.
    Strong inside vortex cores, zero at saddle points.
    """
    zeta_hat = -K2 * eta_tar_hat
    zeta = jnp.real(jnp.fft.ifft2(zeta_hat))
    return normalize_to_rms(zeta, eta_tar, epsilon)


# ── Group B: Inertia-gravity waves ────────────────────────────────────────────

def perturb_B1(eta_tar, X, Y, epsilon, k0=10.0, theta=jnp.pi / 4):
    """
    B.1 Plane wave
    k0    : wavenumber default 10
    theta : propagation direction (default π/4)
    """
    perturb = jnp.cos(k0 * (jnp.cos(theta) * X + jnp.sin(theta) * Y))
    return normalize_to_rms(perturb, eta_tar, epsilon)


def perturb_B2(eta_tar, X, Y, epsilon, rng_key):
    """
    B.2 Supper position of plan wave

    Wavenumbers span from mesoscale to submesoscale (5 → 30).
    Directions are uniformly distributed over [0, π).
    Phases are randomised for each call.

    More physically realistic than B1 — resembles the continuum IGW field.
    """
    wavenumbers = [5.0, 6.5, 7.0, 8.5, 10.0, 11.5, 13.0]
    N = len(wavenumbers)
    thetas  = [n * jnp.pi / N for n in range(N)]
    phases  = jax.random.uniform(rng_key, (N,), minval=0.0, maxval=2.0 * jnp.pi)

    perturb = jnp.zeros_like(X)
    for k0, theta, phase in zip(wavenumbers, thetas, phases):
        perturb = perturb + jnp.cos(k0 * (jnp.cos(theta) * X + jnp.sin(theta) * Y) + phase)
    perturb = perturb / N
    return normalize_to_rms(perturb, eta_tar, epsilon)

# ── Group C: Noise ────────────────────────────────────────────────────────────

def perturb_C1(eta_tar, K, epsilon, rng_key, K_cut, s_slope=-2.0):
    """
    C.1 Noise shaped by |noise_hat| ∝ K^s_slope, keeping only K > K_cut
    (high-pass: low-k part removed).

    s_slope : float
        Slope of the amplitude spectrum. s_slope = -2 reproduces the
        K^-2 energy spectrum used previously; s_slope = 0 gives flat
        (white) noise; positive values give blue noise.
    """
    Nx, Ny = eta_tar.shape
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    K_safe = jnp.where(K > 0, K, 1.0)
    amp = jnp.where(K > 0, K_safe ** (s_slope / 2), 0.0)
    noise_hat = noise_hat * amp
    noise_hat = noise_hat.at[0, 0].set(0.0)
    # Keep only the high-k part (cut low k)
    noise_hat = jnp.where(K > K_cut, noise_hat, 0.0)
    perturb = jnp.real(jnp.fft.ifft2(noise_hat))
    return normalize_to_rms(perturb, eta_tar, epsilon)

def perturb_C2(eta_tar, K, epsilon, rng_key, K_cut, s_slope=-2.0):
    """
    C.2 Noise shaped by |noise_hat| ∝ K^s_slope, keeping only K < K_cut
    (low-pass: high-k part removed).

    s_slope : float
        Slope of the amplitude spectrum. s_slope = -2 reproduces the
        K^-2 energy spectrum used previously; s_slope = 0 gives flat
        (white) noise; positive values give blue noise.
    """
    Nx, Ny = eta_tar.shape
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    K_safe = jnp.where(K > 0, K, 1.0)
    amp = jnp.where(K > 0, K_safe ** (s_slope / 2), 0.0)
    noise_hat = noise_hat * amp
    noise_hat = noise_hat.at[0, 0].set(0.0)
    # Keep only the low-k part (cut high k)
    noise_hat = jnp.where(K < K_cut, noise_hat, 0.0)
    perturb = jnp.real(jnp.fft.ifft2(noise_hat))
    return normalize_to_rms(perturb, eta_tar, epsilon)

def perturb_C3(eta_tar, K, epsilon, rng_key, s_slope = -2.0):
    """
    C.3 noise without the cutoff
    """
    Nx, Ny = eta_tar.shape
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    K_safe = jnp.where(K > 0, K, 1.0)
    amp = jnp.where(K > 0, K_safe ** s_slope, 0.0)
    noise_hat = noise_hat * amp
    noise_hat = noise_hat.at[0, 0].set(0.0)
    perturb = jnp.real(jnp.fft.ifft2(noise_hat))
    return normalize_to_rms(perturb, eta_tar, epsilon)

# ── Group E: SWOT-like instrument & geometry errors ──────────────────────────

def perturb_E1_karin(eta_tar, kx, ky, epsilon, rng_key, k_corr=10.0):
    """
    E.1  Anisotropic KaRIn-like instrument noise.

    White noise shaped by a 2D anisotropic PSD that decorrelates faster in the
    cross-track direction than along-track. By convention here, x is the
    along-track direction and y is the cross-track direction:

        |noise_hat(kx, ky)|^2  ∝  1 / (1 + (ky / k_corr)^2)

    so high cross-track wavenumbers are suppressed (the noise is smoother in y
    than in x), while along-track wavenumbers are unfiltered. This produces a
    spectrally anisotropic noise field — the qualitative signature of the
    KaRIn interferometer baseline error pattern in SWOT data.

    Parameters
    ----------
    k_corr : float
        Cross-track decorrelation wavenumber. Larger k_corr → noise is
        nearly isotropic; smaller k_corr → noise is highly elongated
        in the along-track direction.
    """
    Nx, Ny = eta_tar.shape
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    amp = 1.0 / jnp.sqrt(1.0 + (ky / k_corr) ** 2)
    noise_hat = noise_hat * amp
    noise_hat = noise_hat.at[0, 0].set(0.0)   # zero DC
    perturb = jnp.real(jnp.fft.ifft2(noise_hat))
    return normalize_to_rms(perturb, eta_tar, epsilon)


def perturb_E2_roll(eta_tar, X, Y, Lx, Ly, epsilon, rng_key,
                    n_along=3, n_cross=1):
    """
    E.2  SWOT roll-error proxy.

    Mimics residual antenna roll error: a slowly-varying along-track function
    α(x) multiplied by a near-domain-scale cross-track structure (the "ramp"
    across the swath). In a doubly-periodic toy box we replace the literal
    ramp with the lowest cross-track cosine to keep periodicity:

        eta(x, y) = α(x) · cos(2π · n_cross · y / Ly)

    where α(x) is built from `n_along` low-frequency along-track wavenumbers
    with random amplitudes and phases. The result is concentrated at low
    along-track wavenumber (k_x ≤ n_along) and a single cross-track
    wavenumber (k_y = n_cross) — the spectral fingerprint of a roll error.

    Parameters
    ----------
    n_along : int
        Number of along-track wavenumbers to combine in α(x). Default 3 keeps
        α slowly varying at scales comparable to the domain.
    n_cross : int
        Cross-track wavenumber (in integer units, since Lx = Ly = 2π).
        Default 1 = one wavelength across the cross-track direction, the
        closest periodic analogue of a linear ramp.
    """
    Nx, Ny = eta_tar.shape
    keys = jax.random.split(rng_key, 2)

    amps   = jax.random.normal(keys[0], (n_along,))
    phases = jax.random.uniform(keys[1], (n_along,),
                                minval=0.0, maxval=2.0 * jnp.pi)

    # along-track 1D field α(x), built from low-k cosines
    x_1d = jnp.linspace(0.0, Lx, Nx, endpoint=False)
    alpha_1d = jnp.zeros(Nx)
    for n in range(1, n_along + 1):
        alpha_1d = alpha_1d + amps[n - 1] * jnp.cos(
            2.0 * jnp.pi * n * x_1d / Lx + phases[n - 1]
        )

    # cross-track structure (function of y only, broadcast to (Nx, Ny))
    y_pattern = jnp.cos(2.0 * jnp.pi * n_cross * Y / Ly)

    perturb = alpha_1d[:, None] * y_pattern
    return normalize_to_rms(perturb, eta_tar, epsilon)


def build_all_perturbations(eta_tar, eta_tar_hat, kx, ky, K, K2, inv_K2,
                             mu, inv_mu, Bu, epsilon, X, Y, Lx, Ly, rng_key):
    # Split into three keys:
    igw_key, noise_key, roll_key = jax.random.split(rng_key, 3)

    perturbations = {
        "A1_nothing":         perturb_A1(
                                  eta_tar, eta_tar_hat, kx, ky, epsilon),
        # "A2_frontogenesis":   perturb_A2(
        #                           eta_tar, eta_tar_hat, kx, ky, epsilon),
        # "A3_vorticity":       perturb_A3(
        #                           eta_tar, eta_tar_hat, K2, epsilon),
        # "B1_mono_igw":        perturb_B1(
        #                           eta_tar, X, Y, epsilon),
        "B2_igw_packet":      perturb_B2(
                                  eta_tar, X, Y, epsilon, igw_key),
        # Group C
        "C1_high_k_noise":    perturb_C1(
                                  eta_tar, K, epsilon, noise_key, K_cut=7.0, s_slope=-3.0),
        "C2_low_k_noise":     perturb_C2(
                                  eta_tar, K, epsilon, noise_key, K_cut=7.0, s_slope=-3.0),
        "C3_no_cutoff_noise": perturb_C3(
                                  eta_tar, K, epsilon, noise_key, s_slope=-3.0),
        # ── Group E: SWOT-like errors (opt-in) ─────────────────────────
        # "E1_karin":           perturb_E1_karin(
        #                           eta_tar, kx, ky, epsilon, noise_key, k_corr=10.0),
        # "E2_roll":            perturb_E2_roll(
        #                           eta_tar, X, Y, Lx, Ly, epsilon, roll_key,
        #                           n_along=3, n_cross=1),
    }
    # Target is the QG surface sea surface height plus the perturbation,
    # both in physical space: eta_obs = eta_tar + p
    return {
        name: (p, eta_tar + p)
        for name, p in perturbations.items()
    }

def print_rms_table(cases, phi0_s, epsilon):
    """
    Print a verification table showing that all perturbations have the same RMS.

    Parameters
    ----------
    cases   : dict returned by build_all_perturbations
    phi0_s  : (Nx, Ny) base streamfunction
    epsilon      : epsilonssby number used for normalization
    """
    signal_rms = float(_rms(phi0_s))
    target_rms = epsilon * signal_rms
    print(f"  signal rms(phi0_s) = {signal_rms:.6f}")
    print(f"  target rms(perturb) = epsilon * rms(phi0_s) = {target_rms:.6f}\n")
    print(f"  {'Case':<25} {'rms(perturb)':>14}  {'ratio to target':>16}")
    print("  " + "-" * 60)
    for name, (perturb, _) in cases.items():
        p_rms = float(_rms(jnp.real(perturb)))  # real() handles both physical and spectral targets
        print(f"  {name:<25} {p_rms:>14.6f}  {p_rms / target_rms:>16.4f}")
