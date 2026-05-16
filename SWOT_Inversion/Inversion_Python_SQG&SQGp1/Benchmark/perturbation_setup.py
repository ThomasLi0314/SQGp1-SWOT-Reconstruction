"""
perturbation_setup.py
---------------------
Generates 10 O(Ro) perturbations to the SQG streamfunction (SSH) for
benchmarking the SQG+1 inversion under different kinds of ageostrophic signal.

All perturbations are normalized so that:
    rms(perturbation) = Ro * rms(phi0_s)
which equalises the initial cost function value across cases, ensuring the
benchmark isolates the *structure* of the perturbation rather than its amplitude.


Usage
-----
    from perturbation_setup import build_all_perturbations, print_rms_table

    cases = build_all_perturbations(
        phi0_s, phi0_s_hat, kx, ky, K, K2, inv_K2,
        mu, inv_mu, Bu, Ro, X, Y, Lx, Ly, rng_key
    )
    # cases: {name: (perturbation, psi_true)}  — both in physical space, psi_true = phi0_s + perturbation
    print_rms_table(cases, phi0_s, Ro)
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


def normalize_to_rms(perturb, phi0_s, Ro):
    """
    Rescale `perturb` so that rms(perturb) = Ro * rms(phi0_s).
    Also removes the spatial mean to avoid a DC offset in eta_true.
    """
    perturb = perturb - jnp.mean(perturb)
    target = Ro * _rms(phi0_s)
    current = _rms(perturb)
    return jnp.where(current > 0, perturb * (target / current), perturb)


# Group A

def perturb_A1(phi0_s, phi0_s_hat, kx, ky, K2, inv_K2, mu, inv_mu, Bu, Ro):
    """
    A.1  Cyclogeostrophic + vorticity correction.

    This is the exact O(ε) term in the SQG+1 forward model:
        p1_hat = -(vort + cyc) * inv_K2
        psi_true = phi0_s + Ro * p1_s

    Using Ro directly (no rms normalization) keeps this self-consistent with
    forward_ssh, so the inversion can reach loss = 0. This is the best-case
    baseline — the perturbation is exactly in the range of forward_ssh.
    """
    cyc  = cyclogeo_term(phi0_s_hat, kx, ky)
    vort = vorticity_term(phi0_s_hat, mu, inv_mu, kx, ky, K2, Bu)
    p1_s_hat = -(vort + cyc) * inv_K2
    p1_s = jnp.real(jnp.fft.ifft2(p1_s_hat))
    return Ro * p1_s


def perturb_A2(phi0_s, phi0_s_hat, kx, ky, Ro):
    """
    A.2  Frontogenesis-correlated perturbation.

    Proportional to |∇ψ|², which is large at fronts and strain regions.
    Spatially correlated with the SQG field but not identical to it.

        perturbation ∝ (∂ψ/∂x)² + (∂ψ/∂y)²
    """
    psi_x = jnp.real(jnp.fft.ifft2(phi0_s_hat * 1j * kx))
    psi_y = jnp.real(jnp.fft.ifft2(phi0_s_hat * 1j * ky))
    perturb = psi_x ** 2 + psi_y ** 2
    return normalize_to_rms(perturb, phi0_s, Ro)


def perturb_A3(phi0_s, phi0_s_hat, K2, Ro):
    """
    A.3  Vorticity-aligned perturbation.

    Proportional to the SQG surface vorticity ζ = −∇²ψ.
    In spectral space: ζ̂ = −K² ψ̂.
    Strong inside vortex cores, zero at saddle points.
    """
    zeta_hat = -K2 * phi0_s_hat
    zeta = jnp.real(jnp.fft.ifft2(zeta_hat))
    return normalize_to_rms(zeta, phi0_s, Ro)


# ── Group B: Inertia-gravity waves ────────────────────────────────────────────

def perturb_B1(phi0_s, X, Y, Ro, k0=10.0, theta=jnp.pi / 4):
    """
    B.1 Plane wave
    k0    : wavenumber default 10
    theta : propagation direction (default π/4)
    """
    perturb = jnp.cos(k0 * (jnp.cos(theta) * X + jnp.sin(theta) * Y))
    return normalize_to_rms(perturb, phi0_s, Ro)


def perturb_B2(phi0_s, X, Y, Ro, rng_key):
    """
    B.2 Supper position of plan wave

    Wavenumbers span from mesoscale to submesoscale (5 → 30).
    Directions are uniformly distributed over [0, π).
    Phases are randomised for each call.

    More physically realistic than B1 — resembles the continuum IGW field.
    """
    wavenumbers = [5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]
    N = len(wavenumbers)
    thetas  = [n * jnp.pi / N for n in range(N)]
    phases  = jax.random.uniform(rng_key, (N,), minval=0.0, maxval=2.0 * jnp.pi)

    perturb = jnp.zeros_like(X)
    for k0, theta, phase in zip(wavenumbers, thetas, phases):
        perturb = perturb + jnp.cos(k0 * (jnp.cos(theta) * X + jnp.sin(theta) * Y) + phase)
    perturb = perturb / N
    return normalize_to_rms(perturb, phi0_s, Ro)


def perturb_B3(phi0_s, X, Y, Ro, k0=2.0, theta=jnp.pi / 6):
    """
    B.3  Near-inertial wave at low wavenumber (large spatial scale).

    k0=2 means the wave fits twice across the domain — a near-domain-scale
    unbalanced oscillation. Interesting because the forward model is least
    constrained at low k (few spectral modes), so errors here are large-scale.

    Parameters
    ----------
    k0    : wavenumber (default 2, much smaller than K0=9.5)
    theta : propagation direction (default π/6)
    """
    perturb = jnp.cos(k0 * (jnp.cos(theta) * X + jnp.sin(theta) * Y))
    return normalize_to_rms(perturb, phi0_s, Ro)


# ── Group C: Noise ────────────────────────────────────────────────────────────

def perturb_C1(phi0_s, K, Ro, rng_key, K0=9.5):
    """
    C.1 White noise only at low k
    """
    Nx, Ny = phi0_s.shape
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    noise_hat = jnp.where(K > K0 / 2, noise_hat, 0.0)
    perturb = jnp.real(jnp.fft.ifft2(noise_hat))
    return normalize_to_rms(perturb, phi0_s, Ro)


def perturb_C2(phi0_s, K, Ro, rng_key, K0=9.5):
    """
    C.2  White noise only at high k
    """
    Nx, Ny = phi0_s.shape
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    noise_hat = jnp.where(K < K0 / 2.0, noise_hat, 0.0)
    noise_hat = noise_hat.at[0, 0].set(0.0)   # zero DC
    perturb = jnp.real(jnp.fft.ifft2(noise_hat))
    return normalize_to_rms(perturb, phi0_s, Ro)


def perturb_C3(phi0_s, K, Ro, rng_key):
    """
    C.3  White noise with K⁻² energy spectrum.
    """
    Nx, Ny = phi0_s.shape
    noise_hat = jnp.fft.fft2(jax.random.normal(rng_key, (Nx, Ny)))
    inv_K_safe = jnp.where(K > 0, 1.0 / K, 0.0)
    noise_hat = noise_hat * inv_K_safe
    noise_hat = noise_hat.at[0, 0].set(0.0)   # zero DC
    perturb = jnp.real(jnp.fft.ifft2(noise_hat))
    return normalize_to_rms(perturb, phi0_s, Ro)


# ── Group D: Coherent structures ──────────────────────────────────────────────

def perturb_D1_gaussian_dipole(phi0_s, X, Y, Lx, Ly, Ro):
    """
    D.1  Gaussian vortex dipole.

    Two opposite-sign Gaussian blobs placed at (Lx/3, Ly/2) and (2Lx/3, Ly/2),
    each with width σ = Lx/20. Mimics a submesoscale dipole or an isolated
    frontal eddy pair. Has broadband spectral content (both mesoscale and
    submesoscale energy) but is spatially localised.
    """
    sigma = Lx / 20.0
    cx1, cy1 = Lx / 3.0,       Ly / 2.0
    cx2, cy2 = 2.0 * Lx / 3.0, Ly / 2.0
    blob1 = jnp.exp(-((X - cx1) ** 2 + (Y - cy1) ** 2) / (2.0 * sigma ** 2))
    blob2 = jnp.exp(-((X - cx2) ** 2 + (Y - cy2) ** 2) / (2.0 * sigma ** 2))
    perturb = blob1 - blob2
    return normalize_to_rms(perturb, phi0_s, Ro)

def build_all_perturbations(ssh_qg, ssh_qg_hat, kx, ky, K, K2, inv_K2,
                             mu, inv_mu, Bu, Ro, X, Y, Lx, Ly, rng_key,
                             perturb_streamfunction=True):
    """
    Build all 10 perturbations.

    Parameters
    ----------
    perturb_streamfunction : bool
        True  — perturb the streamfunction ψ.
                Returns {name: (p, psi_true)} where psi_true = phi0_s + p.
                The benchmark calls forward_ssh(psi_true) as the inversion target.
                A1 is exactly invertible (loss → 0). Best for testing optimizer speed.

        False — perturb the SSH observation η.
                Returns {name: (p, eta_true)} where
                eta_true = ssh_qg + p  (both in physical space).
                The perturbation is NOT in the range of forward_ssh, so the
                inversion cannot reach loss = 0. Best for testing inversion skill
                against different ageostrophic signal types.

    Returns
    -------
    dict  {case_name: (perturbation, target)}
        perturbation : (Nx, Ny)   the O(Ro) field p in physical space
        target       : (Nx, Ny)   psi_true  if perturb_streamfunction=True
                       (Nx, Ny)   eta_true = ssh_qg + p  if perturb_streamfunction=False
    """
    keys = jax.random.split(rng_key, 4)

    perturbations = {
        # "A1_cyclogeo":        perturb_A1(
        #                           ssh_qg, ssh_qg_hat, kx, ky, K2, inv_K2, mu, inv_mu, Bu, Ro),
        "A2_frontogenesis":   perturb_A2(
                                  ssh_qg, ssh_qg_hat, kx, ky, Ro),
        # "A3_vorticity":       perturb_A3(
        #                           ssh_qg, ssh_qg_hat, K2, Ro),
        # "B1_mono_igw":        perturb_B1(
        #                           ssh_qg, X, Y, Ro),
        "B2_igw_packet":      perturb_B2(
                                  ssh_qg, X, Y, Ro, keys[0]),
        # "B3_near_inertial":   perturb_B3(
        #                           ssh_qg, X, Y, Ro),
        # "C1_low_k_noise":     perturb_C1(
        #                           ssh_qg, K, Ro, keys[1]),
        "C2_high_k_noise":     perturb_C2(
                                  ssh_qg, K, Ro, keys[2]),
        "C3_k2_spectrum":     perturb_C3(
                                  ssh_qg, K, Ro, keys[3]),
        # "D1_gaussian_dipole": perturb_D1_gaussian_dipole(
        #                           ssh_qg, X, Y, Lx, Ly, Ro),
    }

    if perturb_streamfunction:
        # target is the perturbed streamfunction in physical space
        # benchmark: eta_s_hat_target = forward_ssh(fft2(target), ...)
        return {
            name: (p, phi0_s + p)
            for name, p in perturbations.items()
        }
    else:
        # Target is the QG surface sea surface height plus the perturbation,
        # both in physical space: eta_true = ssh_qg + p
        return {
            name: (p, ssh_qg + p)
            for name, p in perturbations.items()
        }

def print_rms_table(cases, phi0_s, Ro):
    """
    Print a verification table showing that all perturbations have the same RMS.

    Parameters
    ----------
    cases   : dict returned by build_all_perturbations
    phi0_s  : (Nx, Ny) base streamfunction
    Ro      : Rossby number used for normalization
    """
    signal_rms = float(_rms(phi0_s))
    target_rms = Ro * signal_rms
    print(f"  signal rms(phi0_s) = {signal_rms:.6f}")
    print(f"  target rms(perturb) = Ro * rms(phi0_s) = {target_rms:.6f}\n")
    print(f"  {'Case':<25} {'rms(perturb)':>14}  {'ratio to target':>16}")
    print("  " + "-" * 60)
    for name, (perturb, _) in cases.items():
        p_rms = float(_rms(jnp.real(perturb)))  # real() handles both physical and spectral targets
        print(f"  {name:<25} {p_rms:>14.6f}  {p_rms / target_rms:>16.4f}")
