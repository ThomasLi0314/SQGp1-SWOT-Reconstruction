import os
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from physics_functions import calculate_surface_u

def compute_isotropic_spectrum(u, v, K_2d):
    """
    Computes the 1D isotropic kinetic energy spectrum from 2D velocity fields.
    """
    Nx, Ny = u.shape
    # Compute 2D FFT and normalize
    u_hat = np.fft.fft2(np.array(u)) / (Nx * Ny)
    v_hat = np.fft.fft2(np.array(v)) / (Nx * Ny)
    
    # 2D Kinetic Energy
    E_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)
    
    K_flat = np.array(K_2d).flatten()
    E_flat = E_2d.flatten()
    
    # Create radial bins for wavenumber K
    bins = np.linspace(0, K_flat.max(), min(Nx, Ny)//2 + 1)
    K_1d = 0.5 * (bins[:-1] + bins[1:])
    
    # Sum energy in each radial bin
    E_1d, _ = np.histogram(K_flat, bins=bins, weights=E_flat)
    return K_1d, E_1d



def plot_surface_fields(phi0_s_opt, phi0_s_true, u_qg, v_qg, x, y,
                        kx, ky, mu, inv_mu, K2, inv_K2, epsilon, Bu,
                        Nx, Ny, elapsed, run_dir):
    """
    Computes surface velocities and vorticities from the recovered and true
    surface potentials, then plots a 2x3 comparison panel.

    Parameters
    ----------
    phi0_s_opt  : Optimized (recovered) surface potential  (Nx x Ny)
    phi0_s_true : True surface potential                    (Nx x Ny)
    x, y        : 1-D coordinate arrays
    kx, ky      : 2-D wavenumber grids
    mu, inv_mu  : sqrt(Bu)*K and its safe inverse
    K2, inv_K2  : |k|^2 and its safe inverse
    epsilon, Bu : physical parameters
    Nx, Ny      : grid dimensions
    elapsed     : optimization wall-clock time (seconds)
    run_dir     : path to the run folder where the plot is saved
    """
    # ── Compute surface velocities from potentials ──
    phi0_s_hat_opt  = jnp.fft.fft2(phi0_s_opt)
    phi0_s_hat_true = jnp.fft.fft2(phi0_s_true)

    u_surface_opt, v_surface_opt = calculate_surface_u(
        phi0_s_hat_opt, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu
    )
    u_surface_true, v_surface_true = calculate_surface_u(
        phi0_s_hat_true, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu
    )

    # ── Compute surface vorticity: zeta = dv/dx - du/dy ──
    u_opt_hat  = jnp.fft.fft2(u_surface_opt)
    v_opt_hat  = jnp.fft.fft2(v_surface_opt)
    u_true_hat = jnp.fft.fft2(u_surface_true)
    v_true_hat = jnp.fft.fft2(v_surface_true)

    zeta_opt  = jnp.real(jnp.fft.ifft2(1j * kx * v_opt_hat  - 1j * ky * u_opt_hat))
    zeta_true = jnp.real(jnp.fft.ifft2(1j * kx * v_true_hat - 1j * ky * u_true_hat))

    # ── Plot ──
    x_np, y_np = np.array(x), np.array(y)
    method_label = f"LBFGS (jax auto-diff, {Nx}x{Ny})"

    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figs = []

    def plot_row(data_list, suptitle, filename_suffix):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, (data, title) in zip(axes, data_list):
            data_np = np.array(data).T
            vlim = max(abs(np.nanmin(data_np)), abs(np.nanmax(data_np)))
            im = ax.imshow(data_np, origin='lower',
                           extent=[x_np[0], x_np[-1], y_np[0], y_np[-1]],
                           aspect='equal', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("x"); ax.set_ylabel("y")
        
        plt.suptitle(suptitle, fontsize=13)
        plt.tight_layout()
        
        save_path = os.path.join(run_dir, f"{filename_suffix}_grid_{Nx}x{Ny}_date_{date_str}.png")
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        figs.append(fig)

    # 1. Surface zonal velocity u
    plot_row([
        (u_surface_opt,  "Optimized SQG+1 $u_{surface}$"),
        (u_surface_true, "True SQG+1 $u_{surface}$"),
        (u_surface_opt - u_surface_true, "$u$ Difference")
    ], f"Surface Zonal Velocity — {method_label}  ({elapsed:.1f}s)", "surface_u_comparison")

    # 2. Surface vorticity zeta
    plot_row([
        (zeta_opt,  r"Optimized SQG+1 $\zeta_{surface}$"),
        (zeta_true, r"True SQG+1 $\zeta_{surface}$"),
        (zeta_opt - zeta_true, r"$\zeta$ Difference")
    ], f"Surface Vorticity — {method_label}  ({elapsed:.1f}s)", "surface_zeta_comparison")

    # 3. SQG+1 vs Pure QG Zonal Velocity
    plot_row([
        (u_surface_true, r"True SQG+1 $u$"),
        (u_qg, r"Pure QG $u_{QG}$"),
        (u_surface_true - u_qg, r"Ageostrophic Correction ($u_{SQG+1} - u_{QG}$)")
    ], f"QG vs SQG+1 Zonal Velocity — {method_label}  ({elapsed:.1f}s)", "qg_vs_sqg_comparison")

    # 4. Isotropic Kinetic Energy Spectrum
    K_1d, E1d_opt = compute_isotropic_spectrum(u_surface_opt, v_surface_opt, jnp.sqrt(K2))
    _, E1d_true   = compute_isotropic_spectrum(u_surface_true, v_surface_true, jnp.sqrt(K2))
    _, E1d_qg     = compute_isotropic_spectrum(u_qg, v_qg, jnp.sqrt(K2))
    
    fig_ke, ax_ke = plt.subplots(figsize=(7, 5))
    ax_ke.loglog(K_1d, E1d_true, 'k-', linewidth=2, label='True SQG+1')
    ax_ke.loglog(K_1d, E1d_opt, 'r--', linewidth=2, label='Optimized SQG+1')
    ax_ke.loglog(K_1d, E1d_qg, 'b:', linewidth=2, label='Pure QG')
    
    # Only keep wavenumbers greater than 0 for log-log plot to avoid log(0) warning
    valid_idx = K_1d > 0
    ax_ke.set_xlim(left=K_1d[valid_idx].min(), right=K_1d.max())
    
    ax_ke.set_xlabel('Wavenumber $K$')
    ax_ke.set_ylabel('Kinetic Energy Spectral Density')
    ax_ke.set_title(f"Isotropic Kinetic Energy Spectrum\n{method_label}")
    ax_ke.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_ke.legend()
    plt.tight_layout()
    
    save_ke_path = os.path.join(run_dir, f"ke_spectrum_grid_{Nx}x{Ny}_date_{date_str}.png")
    fig_ke.savefig(save_ke_path, dpi=150)
    print(f"Plot saved to {save_ke_path}")
    figs.append(fig_ke)

    import sys
    if "--default" not in sys.argv:
        plt.show()  # This will show all 3 open figures simultaneously
    else:
        for fig in figs:
            plt.close(fig)
