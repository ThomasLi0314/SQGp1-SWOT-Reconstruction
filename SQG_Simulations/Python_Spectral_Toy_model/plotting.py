import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_surface_fields(x, y, u_surface_opt, u_surface_true, zeta_opt, zeta_true, Nx, Ny, elapsed, data_name="Default"):
    """
    Plots the optimized vs true surface horizontal velocities and vorticities.
    Uses a diverging RdBu_r colormap (red = high, blue = low) with symmetric limits.
    """
    x_np, y_np = np.array(x), np.array(y)
    method_label = f"LBFGS (jax auto-diff, {Nx}x{Ny})"

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Row 1: Surface zonal velocity u
    plot_data = [
        (axes[0, 0], u_surface_opt,  "Optimized $u_{surface}$"),
        (axes[0, 1], u_surface_true, "True $u_{surface}$"),
        (axes[0, 2], u_surface_opt - u_surface_true, "$u$ Difference"),
    ]
    for ax, data, title in plot_data:
        data_np = np.array(data).T
        vlim = max(abs(np.nanmin(data_np)), abs(np.nanmax(data_np)))
        im = ax.imshow(data_np, origin='lower',
                       extent=[x_np[0], x_np[-1], y_np[0], y_np[-1]],
                       aspect='equal', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # Row 2: Surface vorticity zeta
    plot_data = [
        (axes[1, 0], zeta_opt,  r"Optimized $\zeta_{surface}$"),
        (axes[1, 1], zeta_true, r"True $\zeta_{surface}$"),
        (axes[1, 2], zeta_opt - zeta_true, r"$\zeta$ Difference"),
    ]
    for ax, data, title in plot_data:
        data_np = np.array(data).T
        vlim = max(abs(np.nanmin(data_np)), abs(np.nanmax(data_np)))
        im = ax.imshow(data_np, origin='lower',
                       extent=[x_np[0], x_np[-1], y_np[0], y_np[-1]],
                       aspect='equal', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    plt.suptitle(f"Surface Fields — {method_label}  ({elapsed:.1f}s)", fontsize=13)
    plt.tight_layout()
    
    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_save_dir = r"D:\Documents\College\Research\Oceangrophy\Shafer_Project\Output"
    save_dir = os.path.join(base_save_dir, data_name)
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"surface_comparison_grid_{Nx}x{Ny}_date_{date_str}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    
    import sys
    if "--default" not in sys.argv:
        plt.show()
    else:
        plt.close(fig)
