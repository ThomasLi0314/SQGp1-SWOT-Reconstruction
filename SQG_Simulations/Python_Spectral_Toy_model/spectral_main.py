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
import os
import time
import tkinter as tk
from tkinter import filedialog
import scipy.io as sio

# Input subfiles. 
from physics_functions import calculate_surface_u, forward_ssh
from cost_functions import cost_function_fmin
from ssh_setup import ssh_setup
from plotting import plot_surface_fields

# Enable 64-bit precision (critical for matching MATLAB numerics)
jax.config.update("jax_enable_x64", True)


# ──────────────────────────────────────────────────────────
# Main Script
# ──────────────────────────────────────────────────────────

def main():
    import sys
    if "--default" in sys.argv:
        file_path = ""
    else:
        print("Select an SSH data file (.mat)... Press Cancel to use default SSH_setup")
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select an SSH data file (.mat)",
            filetypes=[("MATLAB Data Files", "*.mat"), ("All Files", "*.*")]
        )
    
    use_custom_ssh = False
    phi0_s_custom = None
    
    if file_path:
        mat_contents = sio.loadmat(file_path)
        # Find the relevant array
        ssh_data = None
        for key in ['bout', 'phi0_s', 'ssh_data', 'b2spec']:
            if key in mat_contents:
                ssh_data = mat_contents[key]
                if ssh_data.ndim >= 2: # ensure it's a spatial array
                    break
        else:
            keys = [k for k in mat_contents.keys() if not k.startswith('__')]
            if keys:
                for k in keys:
                    if mat_contents[k].ndim >= 2:
                        ssh_data = mat_contents[k]
                        break
                if ssh_data is None:
                    raise ValueError("No valid 2D matrix found in the selected .mat file.")
            else:
                raise ValueError("No variables found in the selected .mat file.")

        if ssh_data.ndim == 3:
            phi0_s_custom = ssh_data[:, :, -1]
            print(f"Loaded 3D data from {file_path}, using the last time period.")
        elif ssh_data.ndim == 2:
            phi0_s_custom = ssh_data
            print(f"Loaded 2D data from {file_path}.")
        phi0_s_custom = phi0_s_custom / jnp.max(jnp.abs(phi0_s_custom))
        
        Nx, Ny = phi0_s_custom.shape
        print(f"Grid size adjusted automatically to Nx={Nx}, Ny={Ny}.")
        use_custom_ssh = True
        data_name = os.path.splitext(os.path.basename(file_path))[0]
    else:
        # ── Parameters ──
        Nx = 512
        Ny = 512
        case_num = 4  # Change this to try different cases (1–7)
        print(f"Using default SSH setup (Nx=512, Ny=512, Case {case_num}).")
        data_name = f"Default_Case_{case_num}"

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
    if use_custom_ssh:
        phi0_s = jnp.array(phi0_s_custom)
        phi0_s_hat = jnp.fft.fft2(phi0_s)
    else:
        # Use the case_num defined above
        key = jax.random.PRNGKey(42)
        phi0_s = ssh_setup(case_num, X, Y, K, Nx, Ny, key)
        phi0_s_hat = jnp.fft.fft2(phi0_s)

    # ── Forward Part: Generate True SSH ──
    eta_s_hat_true = forward_ssh(phi0_s_hat, f, kx, ky, mu, inv_mu, K2, inv_K2, Bu, epsilon)
    print("True SSH data generated")

    # ── Inversion Part ──
    max_phi0_s = jnp.max(phi0_s)
    key2 = jax.random.PRNGKey(0)
    phi0_s_guess = phi0_s + 0.001 * max_phi0_s * jax.random.normal(key2, (Nx, Ny))

    # ===================== Optimization Settings =====================
    num_iterations = 30000

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
    plot_surface_fields(x, y, u_surface_opt, u_surface_true, zeta_opt, zeta_true, Nx, Ny, elapsed, data_name=data_name)


if __name__ == "__main__":
    main()
