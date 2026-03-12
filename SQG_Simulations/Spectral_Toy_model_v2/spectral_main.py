"""
Spectral Toy Model v2 — Python + JAX Implementation
Ported from MATLAB: spectral_main.m + 6 helper .m files.

All operations are 2D spectral (surface only).
Uses JAX for JIT compilation, GPU acceleration, and auto-differentiation.
"""

import jax
import jax.numpy as jnp
from jax import jit
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import time

# Enable 64-bit precision (critical for matching MATLAB numerics)
jax.config.update("jax_enable_x64", True)


# ──────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────

@jit
def cyclogeo_term(phi0_s_hat, kx, ky):
    """
    Cyclogeostrophic correction: 2 * J(Phi_x, Phi_y) in spectral space.
    J = Phi_xx * Phi_yy - Phi_xy^2
    """
    phi0_s_xx = jnp.fft.ifft2(phi0_s_hat * (-1.0) * kx**2)
    phi0_s_yy = jnp.fft.ifft2(phi0_s_hat * (-1.0) * ky**2)
    phi0_s_xy = jnp.fft.ifft2(phi0_s_hat * (-1.0) * ky * kx)

    J_Phi_s = phi0_s_xx * phi0_s_yy - phi0_s_xy**2
    J_Phi_s_hat = 2.0 * jnp.fft.fft2(J_Phi_s)
    return J_Phi_s_hat


@jit
def vorticity_term(phi0_s_hat, mu, inv_mu, kx, ky, K2, Bu):
    """
    Surface vorticity term: zeta_s_hat = (I1 + I2 + ... + I7) / Bu
    """
    phi0_s_x   = jnp.fft.ifft2(phi0_s_hat * 1j * kx)
    phi0_s_y   = jnp.fft.ifft2(phi0_s_hat * 1j * ky)
    phi0_s_zzx = jnp.fft.ifft2(phi0_s_hat * mu * mu * 1j * kx)
    phi0_s_zzy = jnp.fft.ifft2(phi0_s_hat * mu * mu * 1j * ky)
    phi0_s_zz  = jnp.fft.ifft2(phi0_s_hat * mu * mu)
    phi0_s_lap = jnp.fft.ifft2(phi0_s_hat * (-1.0) * K2)
    phi0_s_zx  = jnp.fft.ifft2(phi0_s_hat * mu * 1j * kx)
    phi0_s_zy  = jnp.fft.ifft2(phi0_s_hat * mu * 1j * ky)
    phi0_s_z   = jnp.fft.ifft2(phi0_s_hat * mu)
    phi0_s_lap_z = jnp.fft.ifft2(phi0_s_hat * (-1.0) * K2 * mu)

    # I1: nabla Phi_z . nabla Phi_zz
    I_1 = jnp.fft.fft2(phi0_s_x * phi0_s_zzx + phi0_s_y * phi0_s_zzy)

    # I2: nabla^2 Phi * Phi_zz
    I_2 = jnp.fft.fft2(phi0_s_lap * phi0_s_zz)

    # I3: 2 ||nabla Phi_z||^2
    I_3 = jnp.fft.fft2(2.0 * (phi0_s_zx**2 + phi0_s_zy**2))

    # I4: 2 Phi_z nabla^2 Phi_z
    I_4 = jnp.fft.fft2(2.0 * phi0_s_z * phi0_s_lap_z)

    # I5: K2^2 / mu * Phi_z * Phi_zz
    I_5 = jnp.fft.fft2(phi0_s_z * phi0_s_zz) * K2**2 * inv_mu

    # I6: i*ky*mu * Phi_y * Phi_z
    I_6 = jnp.fft.fft2(phi0_s_y * phi0_s_z) * 1j * ky * mu

    # I7: i*kx*mu * Phi_x * Phi_z
    I_7 = jnp.fft.fft2(phi0_s_x * phi0_s_z) * 1j * kx * mu

    zeta_s_hat = (I_1 + I_2 + I_3 + I_4 + I_5 + I_6 + I_7) / Bu
    return zeta_s_hat


@jit
def calculate_surface_u(phi0_s_hat, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu):
    """
    Compute surface horizontal velocities u, v from phi0_s_hat.
    u = -Phi_y - epsilon * (Phi1_y + F1_z) / Bu
    v =  Phi_x + epsilon * (Phi1_x - G1_z) / Bu
    """
    # Phi1 terms
    phi0_s_z  = jnp.real(jnp.fft.ifft2(phi0_s_hat * mu))
    Phi_1_term1 = jnp.fft.fft2(0.5 * phi0_s_z**2)
    phi0_s_zz = jnp.real(jnp.fft.ifft2(phi0_s_hat * mu * mu))
    Phi_1_term2 = -jnp.fft.fft2(phi0_s_z * phi0_s_zz) * inv_mu

    Phi1_s_hat_y = (Phi_1_term1 + Phi_1_term2) * 1j * ky
    Phi1_s_hat_x = (Phi_1_term1 + Phi_1_term2) * 1j * kx

    # F1 terms
    phi0_s_y  = jnp.real(jnp.fft.ifft2(phi0_s_hat * 1j * ky))
    phi0_s_yz = jnp.real(jnp.fft.ifft2(phi0_s_hat * 1j * ky * mu))
    phi0_s_z  = jnp.real(jnp.fft.ifft2(phi0_s_hat * mu))
    phi0_s_zz = jnp.real(jnp.fft.ifft2(phi0_s_hat * mu * mu))

    F1_term1 = jnp.fft.fft2(phi0_s_y * phi0_s_zz + phi0_s_yz * phi0_s_z)
    F1_term2 = -jnp.fft.fft2(phi0_s_y * phi0_s_z) * mu
    F1_s_hat_z = F1_term1 + F1_term2

    # G1 terms
    phi0_s_x  = jnp.real(jnp.fft.ifft2(phi0_s_hat * 1j * kx))
    phi0_s_xz = jnp.real(jnp.fft.ifft2(phi0_s_hat * 1j * kx * mu))

    G1_term1 = jnp.fft.fft2(phi0_s_x * phi0_s_zz + phi0_s_xz * phi0_s_z)
    G1_term2 = -jnp.fft.fft2(phi0_s_x * phi0_s_z) * mu
    G1_s_hat_z = G1_term1 + G1_term2

    # Sum up
    phi0_s_hat_y = phi0_s_hat * 1j * ky
    phi0_s_hat_x = phi0_s_hat * 1j * kx

    u_s_hat = -phi0_s_hat_y - epsilon * (Phi1_s_hat_y + F1_s_hat_z) / Bu
    v_s_hat =  phi0_s_hat_x + epsilon * (Phi1_s_hat_x - G1_s_hat_z) / Bu

    u_surface = jnp.real(jnp.fft.ifft2(u_s_hat))
    v_surface = jnp.real(jnp.fft.ifft2(v_s_hat))
    return u_surface, v_surface


def forward_ssh(phi0_s_hat, f, kx, ky, mu, inv_mu, K2, inv_K2, Bu, epsilon):
    """Full forward model: phi0_s_hat -> eta_s_hat (SSH in spectral space)."""
    cyc = cyclogeo_term(phi0_s_hat, kx, ky)
    vort = vorticity_term(phi0_s_hat, mu, inv_mu, kx, ky, K2, Bu)
    p1_s_hat = -(f * vort + cyc) * inv_K2
    eta_s_hat = f * phi0_s_hat + p1_s_hat * epsilon
    return eta_s_hat


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


# ──────────────────────────────────────────────────────────
# SSH Setup (Initial Condition Cases)
# ──────────────────────────────────────────────────────────

def ssh_setup(case_num, X, Y, K, Nx, Ny, key=None):
    """
    Generate initial surface potential phi0_s for various benchmark cases.
    
    Parameters
    ----------
    case_num : int (1–7)
    X, Y : 2D coordinate grids (from meshgrid with indexing='ij')
    K : wavenumber magnitude grid
    Nx, Ny : grid dimensions
    key : JAX random key (needed for cases 4 and 6)
    
    Returns
    -------
    phi0_s : 2D array of surface potential (physical space)
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    if case_num == 1:
        # Case 1: Sum
        phi0_s = jnp.cos(X) + jnp.cos(Y)

    elif case_num == 2:
        # Case 2: Product
        phi0_s = jnp.sin(X) * jnp.cos(Y)

    elif case_num == 3:
        # Case 3: Submesoscale perturbation
        phi0_meso = jnp.sin(X) * jnp.cos(Y)
        phi0_submeso = 0.1 * (jnp.sin(10*X) * jnp.cos(12*Y) + jnp.cos(8*X) * jnp.sin(15*Y))
        phi0_s = phi0_meso + phi0_submeso

    elif case_num == 4:
        # Case 4: Random field with k^(-11/3) spectrum
        spectral_slope = -11.0 / 3.0
        white_noise = jax.random.normal(key, (Nx, Ny))
        white_noise_hat = jnp.fft.fft2(white_noise)
        amplitude = jnp.where(K > 0, K**(spectral_slope / 2.0), 0.0)
        phi0_s_hat_random = white_noise_hat * amplitude
        phi0_s = jnp.real(jnp.fft.ifft2(phi0_s_hat_random))
        phi0_s = phi0_s / jnp.max(jnp.abs(phi0_s))

    elif case_num == 5:
        # Case 5: Gaussian
        x0, y0 = jnp.pi, jnp.pi
        R_eddy = 0.5
        phi0_s = jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * R_eddy**2))

    elif case_num == 6:
        # Case 6: Gaussian with white noise
        x0, y0 = jnp.pi, jnp.pi
        R_eddy = 0.5
        phi0_s = jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * R_eddy**2))
        white_noise = 0.01 * jax.random.normal(key, (Nx, Ny))
        phi0_s = phi0_s + white_noise

    elif case_num == 7:
        # Case 7: Different scales of modes
        mode1 = 1.0 * jnp.cos(1*X + 1*Y)
        mode2 = 0.5 * jnp.sin(3*X - 2*Y)
        mode3 = 0.2 * jnp.cos(7*X + 5*Y)
        mode4 = 0.05 * jnp.sin(12*X) * jnp.cos(10*Y)
        phi0_s = mode1 + mode2 + mode3 + mode4
        phi0_s = phi0_s / jnp.max(jnp.abs(phi0_s))

    else:
        raise ValueError(f"Unknown case_num: {case_num}. Use 1–7.")

    return phi0_s


# ──────────────────────────────────────────────────────────
# Main Script
# ──────────────────────────────────────────────────────────

def main():
    # ── Parameters ──
    Nx = 512
    Ny = 512
    Lx = 2.0 * jnp.pi
    Ly = 2.0 * jnp.pi
    Ro = 0.01
    Bu = 1.0
    f = 1.0
    epsilon = Ro

    # ── Grid Setup (initialize.m) ──
    dx = Lx / Nx
    dy = Ly / Ny
    x = jnp.arange(Nx) * dx
    y = jnp.arange(Ny) * dy
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    # Spectral grid
    dk = 2.0 * jnp.pi / Nx
    dl = 2.0 * jnp.pi / Ny
    k_zonal = jnp.concatenate([jnp.arange(Nx // 2), jnp.arange(-Nx // 2, 0)]) * dk
    l_meridional = jnp.concatenate([jnp.arange(Ny // 2), jnp.arange(-Ny // 2, 0)]) * dl
    kx, ky = jnp.meshgrid(k_zonal, l_meridional, indexing='ij')

    K2 = kx**2 + ky**2
    K = jnp.sqrt(K2)

    # Safe inverses (avoid division by zero at k=0)
    inv_K = jnp.where(K > 0, 1.0 / K, 0.0)
    inv_K2 = jnp.where(K2 > 0, 1.0 / K2, 0.0)

    mu = jnp.sqrt(Bu) * K
    inv_mu = jnp.where(mu > 0, 1.0 / mu, 0.0)

    # ── SSH Setup ──
    case_num = 4  # Change this to try different cases (1–7)
    key = jax.random.PRNGKey(42)
    phi0_s = ssh_setup(case_num, X, Y, K, Nx, Ny, key)
    phi0_s_hat = jnp.fft.fft2(phi0_s)

    # ── Forward Part: Generate True SSH ──
    eta_s_hat_true = forward_ssh(phi0_s_hat, f, kx, ky, mu, inv_mu, K2, inv_K2, Bu, epsilon)
    print("True SSH data generated")

    # ── Inversion Part ──
    max_phi0_s = jnp.max(phi0_s)
    key2 = jax.random.PRNGKey(0)
    phi0_s_guess = phi0_s + 0.1 * max_phi0_s * jax.random.normal(key2, (Nx, Ny))

    # ===================== Optimization Settings =====================
    num_iterations = 3000

    # Build JIT-compiled cost and gradient (fminunc style)
    @jit
    def loss_fn(phi0_s_flat):
        phi0_s_2d = phi0_s_flat.reshape(Nx, Ny)
        return cost_function_fmin(
            phi0_s_2d, f, kx, ky, mu, inv_mu, Bu, epsilon, K2, inv_K2, eta_s_hat_true
        )

    grad_fn = jit(jax.grad(loss_fn))

    # Use L-BFGS (matches MATLAB's fminunc quasi-newton)
    solver = jaxopt.LBFGS(
        fun=loss_fn,
        maxiter=num_iterations,
        tol=1e-8,
    )

    print(f"Running optimization: LBFGS (auto-diff gradients, {Nx}x{Ny})")
    print("JIT compiling (first call may be slow)...")
    t0 = time.time()

    phi0_flat_guess = phi0_s_guess.ravel()

    # ── Option A: One-liner (no per-iteration logging) ──
    # params, state = solver.run(phi0_flat_guess)

    # ── Option B: Manual loop with early stopping + logging ──
    state = solver.init_state(phi0_flat_guess)
    params = phi0_flat_guess

    for i in range(num_iterations):
        params, state = solver.update(params, state)
        loss_val = float(loss_fn(params))
        g_norm = float(jnp.linalg.norm(grad_fn(params)))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Iter {i+1:4d} | Loss = {loss_val:.6e} | |grad| = {g_norm:.6e}")
        if loss_val < 1e-12:
            print(f"  Converged at iter {i+1} (Loss = {loss_val:.6e})")
            break

    elapsed = time.time() - t0
    print(f"Optimization Complete. Elapsed: {elapsed:.2f}s")

    phi0_s_opt = params.reshape(Nx, Ny)
    phi0_s_hat_opt = jnp.fft.fft2(phi0_s_opt)

    # ── Validation: Surface Velocities ──
    u_surface_opt, v_surface_opt = calculate_surface_u(
        phi0_s_hat_opt, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu
    )
    u_surface_true, v_surface_true = calculate_surface_u(
        phi0_s_hat, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu
    )

    # Surface vorticity: zeta = dv/dx - du/dy
    u_opt_hat = jnp.fft.fft2(u_surface_opt)
    v_opt_hat = jnp.fft.fft2(v_surface_opt)
    u_true_hat = jnp.fft.fft2(u_surface_true)
    v_true_hat = jnp.fft.fft2(v_surface_true)

    zeta_opt  = jnp.real(jnp.fft.ifft2(1j * kx * v_opt_hat  - 1j * ky * u_opt_hat))
    zeta_true = jnp.real(jnp.fft.ifft2(1j * kx * v_true_hat - 1j * ky * u_true_hat))

    # ── Plotting ──
    x_np, y_np = np.array(x), np.array(y)
    method_label = f"LBFGS (jax auto-diff, {Nx}x{Ny})"

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    # Row 1: Surface zonal velocity u
    plot_data = [
        (axes[0, 0], u_surface_opt,  "Optimized $u_{surface}$"),
        (axes[0, 1], u_surface_true, "True $u_{surface}$"),
        (axes[0, 2], u_surface_opt - u_surface_true, "$u$ Difference"),
    ]
    for ax, data, title in plot_data:
        im = ax.imshow(np.array(data).T, origin='lower', extent=[x_np[0], x_np[-1], y_np[0], y_np[-1]], aspect='equal')
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # Row 2: Surface vorticity zeta
    plot_data = [
        (axes[1, 0], zeta_opt,  r"Optimized $\zeta_{surface}$"),
        (axes[1, 1], zeta_true, r"True $\zeta_{surface}$"),
        (axes[1, 2], zeta_opt - zeta_true, r"$\zeta$ Difference"),
    ]
    for ax, data, title in plot_data:
        im = ax.imshow(np.array(data).T, origin='lower', extent=[x_np[0], x_np[-1], y_np[0], y_np[-1]], aspect='equal')
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.suptitle(f"Surface Fields — {method_label}  ({elapsed:.1f}s)")
    plt.tight_layout()
    plt.savefig("surface_comparison.png", dpi=150)
    plt.show()
    print("Plot saved to surface_comparison.png")


if __name__ == "__main__":
    main()
