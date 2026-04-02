import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from physics_functions import calculate_surface_u

# ── Global toggle: set to True to save plots to disk ──
SAVE_OUTPUT = True


# ─────────────────────────────────────────────────────────
#  Utility: isotropic energy spectrum
# ─────────────────────────────────────────────────────────
def compute_isotropic_spectrum(u, v, K_2d):
    """
    Computes the 1D isotropic kinetic energy spectrum from 2D velocity fields.
    """
    Nx, Ny = u.shape
    u_hat = np.fft.fft2(np.array(u)) / (Nx * Ny)
    v_hat = np.fft.fft2(np.array(v)) / (Nx * Ny)

    E_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    K_flat = np.array(K_2d).flatten()
    E_flat = E_2d.flatten()

    bins = np.linspace(0, K_flat.max(), min(Nx, Ny)//2 + 1)
    K_1d = 0.5 * (bins[:-1] + bins[1:])

    E_1d, _ = np.histogram(K_flat, bins=bins, weights=E_flat)
    return K_1d, E_1d


# ─────────────────────────────────────────────────────────
#  Core helper: draw a 1×3 comparison row
# ─────────────────────────────────────────────────────────
def plot_row(data_list, suptitle, filename_suffix, ctx):
    """
    Plots a single 1×3 comparison panel.

    Parameters
    ----------
    data_list       : list of (2D-array, title_str) triples
    suptitle        : figure super-title
    filename_suffix : used when saving the PNG
    ctx             : context dict returned by prepare_surface_fields
    """
    x_np    = ctx["x_np"]
    y_np    = ctx["y_np"]
    Nx      = ctx["Nx"]
    Ny      = ctx["Ny"]
    run_dir = ctx["run_dir"]
    date_str = ctx["date_str"]

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

    if SAVE_OUTPUT and run_dir is not None:
        save_path = os.path.join(run_dir,
                                 f"{filename_suffix}_grid_{Nx}x{Ny}_date_{date_str}.png")
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")

    return fig


# ─────────────────────────────────────────────────────────
#  SETUP: compute all derived fields, return a context dict
# ─────────────────────────────────────────────────────────
def prepare_surface_fields(phi0_s_opt, phi0_s_true, phi0_s_ori, u_qg, v_qg, zeta_qg, x, y,
                           kx, ky, mu, inv_mu, K2, inv_K2, epsilon, Bu,
                           Nx, Ny, elapsed, run_dir):
    """
    Computes surface velocities and vorticities from the recovered and true
    surface potentials, and bundles everything into a context dict.

    Returns
    -------
    ctx : dict
        Contains all computed fields plus plotting metadata.
        Keys: u_surface_opt, v_surface_opt, u_surface_true, v_surface_true,
              zeta_opt, zeta_true, u_qg, v_qg, zeta_qg,
              x_np, y_np, method_label, date_str,
              Nx, Ny, elapsed, run_dir, K2
    """
    # ── Compute surface velocities from potentials ──
    phi0_s_hat_opt  = jnp.fft.fft2(phi0_s_opt)
    phi0_s_hat_true = jnp.fft.fft2(phi0_s_true)
    phi0_s_hat_ori = jnp.fft.fft2(phi0_s_ori)

    u_surface_opt, v_surface_opt = calculate_surface_u(
        phi0_s_hat_opt, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu
    )
    u_surface_true, v_surface_true = calculate_surface_u(
        phi0_s_hat_true, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu
    )
    u_surface_ori, v_surface_ori = calculate_surface_u(
        phi0_s_hat_ori, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu
    )

    # ── Compute surface vorticity: zeta = dv/dx - du/dy ──
    u_opt_hat  = jnp.fft.fft2(u_surface_opt)
    v_opt_hat  = jnp.fft.fft2(v_surface_opt)
    u_true_hat = jnp.fft.fft2(u_surface_true)
    v_true_hat = jnp.fft.fft2(v_surface_true)
    u_ori_hat = jnp.fft.fft2(u_surface_ori)
    v_ori_hat = jnp.fft.fft2(v_surface_ori)

    zeta_opt  = jnp.real(jnp.fft.ifft2(1j * kx * v_opt_hat  - 1j * ky * u_opt_hat))
    zeta_true = jnp.real(jnp.fft.ifft2(1j * kx * v_true_hat - 1j * ky * u_true_hat))
    zeta_ori = jnp.real(jnp.fft.ifft2(1j * kx * v_ori_hat - 1j * ky * u_ori_hat))

    # ── Bundle into context dict ──
    ctx = dict(
        u_surface_opt=u_surface_opt,   v_surface_opt=v_surface_opt,
        u_surface_true=u_surface_true, v_surface_true=v_surface_true,
        u_surface_ori=u_surface_ori, v_surface_ori=v_surface_ori,
        zeta_opt=zeta_opt,             zeta_true=zeta_true,
        zeta_ori=zeta_ori,
        u_qg=u_qg,                    v_qg=v_qg,
        zeta_qg=zeta_qg,
        x_np=np.array(x),             y_np=np.array(y),
        method_label=f"LBFGS (jax {Nx}x{Ny})",
        date_str=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        Nx=Nx, Ny=Ny, elapsed=elapsed,
        run_dir=run_dir, K2=K2,
    )
    return ctx


# ─────────────────────────────────────────────────────────
#  Individual plot functions (call after prepare_surface_fields)
# ─────────────────────────────────────────────────────────
def plot_surface_u(ctx):
    """1. Surface zonal velocity u: optimized vs true vs difference."""
    return plot_row([
        (ctx["u_surface_opt"],  "Optimized SQG+1 $u_{surface}$"),
        (ctx["u_surface_true"], "True SQG+1 $u_{surface}$"),
        (ctx["u_surface_opt"] - ctx["u_surface_true"], "$u$ Difference"),
    ],
        f"Surface Zonal Velocity — {ctx['method_label']}  ({ctx['elapsed']:.1f}s)",
        "surface_u_comparison", ctx)


def plot_surface_zeta(ctx):
    """2. Surface vorticity zeta: optimized vs true vs difference."""
    return plot_row([
        (ctx["zeta_opt"],  r"Optimized SQG+1 $\zeta_{surface}$"),
        (ctx["zeta_true"], r"True SQG+1 $\zeta_{surface}$"),
        (ctx["zeta_opt"] - ctx["zeta_true"], r"$\zeta$ Difference"),
    ],
        f"Surface Vorticity — {ctx['method_label']}  ({ctx['elapsed']:.1f}s)",
        "surface_zeta_comparison", ctx)


def plot_qg_vs_sqg_u(ctx):
    """3. SQG+1 vs Pure QG zonal velocity."""
    return plot_row([
        (ctx["u_surface_true"], r"True SQG+1 $u$"),
        (ctx["u_qg"],           r"Pure QG $u_{QG}$"),
        (ctx["u_surface_true"] - ctx["u_qg"],
         r"Ageostrophic Correction ($u_{SQG+1} - u_{QG}$)"),
    ],
        f"QG vs SQG+1 Zonal Velocity — {ctx['method_label']}  ({ctx['elapsed']:.1f}s)",
        "qg_vs_sqg_comparison", ctx)


def plot_qg_vs_sqg_zeta(ctx):
    """4. SQG+1 vs Pure QG vorticity zeta."""
    return plot_row([
        (ctx["zeta_opt"],  r"Optimized SQG+1 $\zeta_{surface}$"),
        (ctx["zeta_qg"],   r"Pure QG $\zeta_{QG}$"),
        (ctx["zeta_opt"] - ctx["zeta_qg"],
         r"Ageostrophic Correction ($\zeta_{SQG+1} - \zeta_{QG}$)"),
    ],
        f"Surface Vorticity — {ctx['method_label']}  ({ctx['elapsed']:.1f}s)",
        "qg_vs_sqg_vorticity_comparison", ctx)

def plot_ori_vs_true_zeta(ctx):
    " True Vorticity vs Original Vorticity (Unperturbed)"
    return plot_row([
        (ctx["zeta_ori"], r"Original SQG+1 $\zeta_{surface}$"),
        (ctx["zeta_true"], r"True SQG+1 $\zeta_{surface}$"),
        (ctx["zeta_ori"] - ctx["zeta_true"], r"$\zeta$ Difference"),
    ],
        f"Surface Vorticity — {ctx['method_label']}  ({ctx['elapsed']:.1f}s)",
        "ori_vs_true_vorticity_comparison", ctx)



# ─────────────────────────────────────────────────────────
#  Convenience: plot all at once (preserves old behaviour)
# ─────────────────────────────────────────────────────────
def plot_surface_fields(phi0_s_opt, phi0_s_true, phi0_s_ori, u_qg, v_qg, zeta_qg, x, y,
                        kx, ky, mu, inv_mu, K2, inv_K2, epsilon, Bu,
                        Nx, Ny, elapsed, run_dir):
    """
    Legacy all-in-one wrapper.  Calls prepare → individual plots → show/close.
    """
    ctx = prepare_surface_fields(
        phi0_s_opt, phi0_s_true, phi0_s_ori, u_qg, v_qg, zeta_qg, x, y,
        kx, ky, mu, inv_mu, K2, inv_K2, epsilon, Bu,
        Nx, Ny, elapsed, run_dir,
    )

    figs = [
        plot_surface_u(ctx),
        plot_surface_zeta(ctx),
        plot_qg_vs_sqg_u(ctx),
        plot_qg_vs_sqg_zeta(ctx),
    ]

    import sys
    if "--default" not in sys.argv:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)
