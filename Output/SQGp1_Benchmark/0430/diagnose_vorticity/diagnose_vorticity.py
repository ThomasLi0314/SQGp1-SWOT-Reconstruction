"""Why is SSH reconstruction good (~1% rel err) but vorticity bad (~173%)?

Loads the saved workspace from Benchmark_v4 and runs a battery of spectral
diagnostics.  Does NOT modify the notebook.

Run from this folder:
    python diagnose_vorticity.py
"""
from pathlib import Path
import sys

import numpy as np
import dill
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Path wiring so we can pull in the same physics + spectrum helpers used by
# the notebook.
# ---------------------------------------------------------------------------
SQG_ROOT = Path(__file__).resolve().parents[4] / \
    "SQG_Simulations" / "Python_Spectral_Toy_model"
sys.path.insert(0, str(SQG_ROOT))
sys.path.insert(0, str(SQG_ROOT / "Benchmark_v4"))

from physics_functions import calculate_surface_u, forward_ssh
from spectrum_diagnostics import azimuthal_sum

import scipy.io as sio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE       = Path(__file__).resolve().parent              # .../0430/diagnose_vorticity
PKL        = HERE.parent / "workspace.pkl"                # .../0430/workspace.pkl
MAT        = SQG_ROOT / "Benchmark_v4" / "data" / "sqgp1_for_thomas.mat"
OUT        = HERE
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rms(x):
    return float(jnp.sqrt(jnp.mean(jnp.asarray(x) ** 2)))


def radial_psd(field_hat, K, dk, Nx, Ny):
    """Per-ring summed |field_hat|^2 / (Nx Ny)^2  (matches WPSD convention)."""
    p2d = jnp.abs(field_hat) ** 2 / (Nx * Ny) ** 2
    return azimuthal_sum(p2d, K, dk)


def compute_zeta_from_phi(phi_hat, mu, inv_mu, kx, ky, K2, inv_K2, Ro_sys, Bu):
    u, v = calculate_surface_u(phi_hat, mu, inv_mu, kx, ky,
                               K2, inv_K2, Ro_sys, Bu)
    zeta = jnp.real(jnp.fft.ifft2(
        1j * kx * jnp.fft.fft2(v) - 1j * ky * jnp.fft.fft2(u)
    ))
    return u, v, zeta


def apply_lp_filter(field_phys, filter_lp, Nx, Ny):
    f_hat = jnp.fft.fft2(field_phys) * filter_lp
    f_hat = f_hat.at[Nx // 2, :].set(0.0).at[:, Ny // 2].set(0.0)
    return jnp.real(jnp.fft.ifft2(f_hat))


# ---------------------------------------------------------------------------
# Load the saved workspace
# ---------------------------------------------------------------------------
print(f"Loading workspace: {PKL}")
with open(PKL, "rb") as f:
    ws = dill.load(f)

# Pull what we need
Nx, Ny      = ws["Nx"], ws["Ny"]
Lx, Ly      = ws["Lx"], ws["Ly"]
kx, ky      = ws["kx"], ws["ky"]
K, K2       = ws["K"], ws["K2"]
inv_K2      = ws["inv_K2"]
mu, inv_mu  = ws["mu"], ws["inv_mu"]
Ro_sys      = ws["Ro_sys"]
epsilon     = ws["epsilon"]
Bu          = ws["Bu"]
phi0_opt    = ws["phi0_opt"]
phi0_opt_hat = ws["phi0_opt_hat"]
eta_tar     = ws["eta_tar"]                 # unfiltered target
phi0_s      = ws["phi0_s"]                  # = psi0n (unfiltered truth streamfunction)

dk = 1.0
print(f"  Nx,Ny = {Nx},{Ny}    Ro = {Ro_sys}    epsilon = {epsilon}")

# The workspace doesn't carry psi0_lpfilter / eta_tar_lpfilter directly, so
# rebuild them with the same filter the notebook used (kappa_c = 100).
kappa_c  = 100
dx_lores = float(Lx) / float(Nx)
exponent  = -23.6 * (K - kappa_c) ** 4 * dx_lores ** 4
filter_lp = jnp.where(K < kappa_c, 1.0, jnp.exp(exponent))

psi0_lpfilter      = apply_lp_filter(phi0_s, filter_lp, Nx, Ny)
psi0_lpfilter_hat  = jnp.fft.fft2(psi0_lpfilter)
eta_tar_lpfilter   = apply_lp_filter(eta_tar, filter_lp, Nx, Ny)
eta_tar_lpfilter_hat = jnp.fft.fft2(eta_tar_lpfilter)

# Need the .mat truth zetas (not in the pickle)
print(f"Loading .mat:        {MAT}")
mat = sio.loadmat(MAT)
zeta0n = jnp.asarray(mat["zeta0n"], dtype=jnp.float64)
zeta1n = jnp.asarray(mat["zeta1n"], dtype=jnp.float64)
zeta_tar_truth = zeta0n + epsilon * zeta1n

# Recovered SSH and vorticity from phi0_opt
eta_opt_hat = forward_ssh(phi0_opt_hat, kx, ky, mu, inv_mu, K2, inv_K2, Bu, Ro_sys)
eta_opt     = jnp.real(jnp.fft.ifft2(eta_opt_hat))
_, _, zeta_opt = compute_zeta_from_phi(
    phi0_opt_hat, mu, inv_mu, kx, ky, K2, inv_K2, Ro_sys, Bu)

# Vorticity from the LP-filtered initial guess (baseline floor)
_, _, zeta_init = compute_zeta_from_phi(
    psi0_lpfilter_hat, mu, inv_mu, kx, ky, K2, inv_K2, Ro_sys, Bu)

# ---------------------------------------------------------------------------
# Bulk-number summary
# ---------------------------------------------------------------------------
ssh_res        = eta_opt   - eta_tar_lpfilter
zeta_res       = zeta_opt  - zeta_tar_truth
zeta_res_init  = zeta_init - zeta_tar_truth

# Filtered truth: what fraction of zeta_tar even survives the LP cutoff?
zeta_tar_lp    = apply_lp_filter(zeta_tar_truth, filter_lp, Nx, Ny)
zeta_res_lp    = zeta_opt - zeta_tar_lp                  # opt vs FILTERED truth
zeta_lost_to_lp = zeta_tar_truth - zeta_tar_lp           # what filter throws away

print("\n" + "=" * 72)
print("BULK NUMBERS")
print("=" * 72)
print(f"  rms eta_tar_lpfilter             = {rms(eta_tar_lpfilter):.4e}")
print(f"  rms ssh residual (opt vs lp tar) = {rms(ssh_res):.4e}    "
      f"rel = {rms(ssh_res)/rms(eta_tar_lpfilter):.4%}")
print()
print(f"  rms zeta_tar (UNfiltered truth)  = {rms(zeta_tar_truth):.4e}")
print(f"  rms zeta_tar_lp (LP truth)       = {rms(zeta_tar_lp):.4e}    "
      f"rel-of-truth = {rms(zeta_tar_lp)/rms(zeta_tar_truth):.4%}")
print(f"  rms (zeta_tar - zeta_tar_lp)     = {rms(zeta_lost_to_lp):.4e}    "
      f"<-- pure LP-truncation error floor")
print(f"  rms zeta_opt                     = {rms(zeta_opt):.4e}")
print()
print(f"  rms zeta_res (opt vs UNFILT)     = {rms(zeta_res):.4e}    "
      f"rel = {rms(zeta_res)/rms(zeta_tar_truth):.4%}")
print(f"  rms zeta_res (opt vs LP truth)   = {rms(zeta_res_lp):.4e}    "
      f"rel-vs-LP = {rms(zeta_res_lp)/rms(zeta_tar_lp):.4%}")
print(f"  rms zeta_init residual baseline  = {rms(zeta_res_init):.4e}    "
      f"rel = {rms(zeta_res_init)/rms(zeta_tar_truth):.4%}")

# ---------------------------------------------------------------------------
# Hypothesis 1: phi0_opt has spurious high-k content the inversion didn't
# care about (because the loss is dominated by k <= kappa_c).
# ---------------------------------------------------------------------------
k_axis, S_phi0_opt   = radial_psd(phi0_opt_hat,      K, dk, Nx, Ny)
_,      S_phi0_lp    = radial_psd(psi0_lpfilter_hat, K, dk, Nx, Ny)
_,      S_phi0_truth = radial_psd(jnp.fft.fft2(phi0_s), K, dk, Nx, Ny)

phi_diff_hat = phi0_opt_hat - psi0_lpfilter_hat
_, S_phi_diff = radial_psd(phi_diff_hat, K, dk, Nx, Ny)

# SSH residual spectrum
ssh_res_hat = eta_opt_hat - eta_tar_lpfilter_hat
_, S_ssh_res = radial_psd(ssh_res_hat, K, dk, Nx, Ny)
_, S_eta_lp  = radial_psd(eta_tar_lpfilter_hat, K, dk, Nx, Ny)

# Vorticity residual / target spectra (scalar fields)
_, S_zres   = radial_psd(jnp.fft.fft2(zeta_res),     K, dk, Nx, Ny)
_, S_zres_lp = radial_psd(jnp.fft.fft2(zeta_res_lp), K, dk, Nx, Ny)
_, S_ztar   = radial_psd(jnp.fft.fft2(zeta_tar_truth), K, dk, Nx, Ny)
_, S_ztar_lp = radial_psd(jnp.fft.fft2(zeta_tar_lp),   K, dk, Nx, Ny)
_, S_zopt   = radial_psd(jnp.fft.fft2(zeta_opt),     K, dk, Nx, Ny)

# Skill score on vorticity vs UNFILTERED truth
with np.errstate(invalid="ignore", divide="ignore"):
    WPSD_skill_unf = 1.0 - np.where(S_ztar > 1e-30, S_zres / S_ztar, np.nan)
    WPSD_skill_lp  = 1.0 - np.where(S_ztar_lp > 1e-30, S_zres_lp / S_ztar_lp, np.nan)

# Cumulative vorticity error broken into k <= kc and k > kc
mask_lo = k_axis <= kappa_c
mask_hi = k_axis  > kappa_c
e_lo = float(np.sum(S_zres[mask_lo]))
e_hi = float(np.sum(S_zres[mask_hi]))
t_lo = float(np.sum(S_ztar[mask_lo]))
t_hi = float(np.sum(S_ztar[mask_hi]))
print()
print("=" * 72)
print(f"VORTICITY ERROR SPLIT AT kappa_c = {kappa_c}")
print("=" * 72)
print(f"  zeta_res^2 below kc  : {e_lo:.4e}   ({e_lo/(e_lo+e_hi):.2%} of total)")
print(f"  zeta_res^2 above kc  : {e_hi:.4e}   ({e_hi/(e_lo+e_hi):.2%} of total)")
print(f"  zeta_tar^2 below kc  : {t_lo:.4e}   ({t_lo/(t_lo+t_hi):.2%} of total)")
print(f"  zeta_tar^2 above kc  : {t_hi:.4e}   ({t_hi/(t_lo+t_hi):.2%} of total)")

# ---------------------------------------------------------------------------
# Plot 1: streamfunction spectra (truth, lp truth, opt, opt - lp truth)
# ---------------------------------------------------------------------------
mask = k_axis > 0
k_plot_max = Nx // 2

fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
ax.loglog(k_axis[mask], S_phi0_truth[mask], lw=1.4, color="k",
          label=r"$|\hat\psi_0|^2$  truth")
ax.loglog(k_axis[mask], S_phi0_lp[mask],    lw=1.2, color="C0", ls="--",
          label=r"$|\hat\psi_{0,\mathrm{lp}}|^2$  (LP{truth})")
ax.loglog(k_axis[mask], S_phi0_opt[mask],   lw=1.2, color="C3",
          label=r"$|\hat\Phi^0_{opt}|^2$  recovered")
ax.loglog(k_axis[mask], S_phi_diff[mask],   lw=1.0, color="C3", ls=":",
          label=r"$|\hat\Phi^0_{opt}-\hat\psi_{0,\mathrm{lp}}|^2$")
ax.axvline(kappa_c, color="k", lw=0.7, ls=":", label=rf"$\kappa_c={kappa_c}$")
ax.set_xlabel("wavenumber  $k$")
ax.set_ylabel("radial PSD")
ax.set_title("Streamfunction spectra: truth, LP-truth, recovered")
ax.set_xlim(1, k_plot_max)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
fig.savefig(OUT / "01_streamfunction_spectra.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: SSH residual vs vorticity residual spectra, overlaid with K^4
# ---------------------------------------------------------------------------
# At leading QG order, zeta_hat ≈ -K^2 * phi0_hat, so the residual PSD of
# zeta should be roughly K^4 times the residual PSD of phi0.  And since
# (eta_hat ≈ phi0_hat) for the linear part, K^4 * S_ssh_res is a back-of-
# envelope predictor for S_zres.
K_axis_eff = k_axis.astype(float)
K_axis_eff[0] = 1.0
predictor_K4 = (K_axis_eff ** 4) * S_ssh_res     # K^4 * SSH residual PSD

fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
ax.loglog(k_axis[mask], S_eta_lp[mask],   lw=1.4, color="C0",
          label=r"$|\hat\eta_{tar,\mathrm{lp}}|^2$  (target)")
ax.loglog(k_axis[mask], S_ssh_res[mask],  lw=1.2, color="C0", ls="--",
          label=r"SSH residual  $|\hat\eta_{opt}-\hat\eta_{tar,\mathrm{lp}}|^2$")
ax.loglog(k_axis[mask], S_ztar[mask],     lw=1.4, color="C3",
          label=r"$|\hat\zeta_{tar}|^2$  (UNfiltered truth)")
ax.loglog(k_axis[mask], S_zres[mask],     lw=1.2, color="C3", ls="--",
          label=r"vorticity residual  $|\hat\zeta_{opt}-\hat\zeta_{tar}|^2$")
ax.loglog(k_axis[mask], predictor_K4[mask], lw=1.0, color="grey", ls=":",
          label=r"$K^4\,\cdot\,$SSH-residual PSD  (predictor)")
ax.axvline(kappa_c, color="k", lw=0.6, ls=":")
ax.set_xlabel("wavenumber  $k$")
ax.set_ylabel("radial PSD")
ax.set_title("SSH and vorticity residual spectra (compare K⁴ amplification)")
ax.set_xlim(1, k_plot_max)
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="lower left")
fig.savefig(OUT / "02_residual_spectra_K4.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3: WPSD skill score on vorticity, vs UNfiltered AND LP-filtered truth
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
ax.semilogx(k_axis, WPSD_skill_unf, lw=1.3, color="C3",
            label=r"vs UNfiltered truth")
ax.semilogx(k_axis, WPSD_skill_lp,  lw=1.3, color="C0",
            label=r"vs LP-filtered truth")
ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
ax.axvline(kappa_c, color="k", lw=0.6, ls=":")
ax.set_ylim(-1.05, 1.05)
ax.set_xlim(1, k_plot_max)
ax.set_xlabel("wavenumber  $k$")
ax.set_ylabel(r"$\mathrm{WPSD}_S(k) = 1 - \mathrm{WPSD}(\zeta_{opt}-\zeta_*)/\mathrm{WPSD}(\zeta_*)$")
ax.set_title("Vorticity skill score: filtered vs unfiltered truth")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
fig.savefig(OUT / "03_vorticity_skill.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 4: maps — zeta_opt vs zeta_tar_lp (LP truth), residual
# ---------------------------------------------------------------------------
extent = [0, float(Lx), 0, float(Ly)]
pct = 99
vmax_z   = float(jnp.percentile(jnp.abs(zeta_tar_truth), pct))
vmax_err = float(jnp.percentile(jnp.abs(zeta_res_lp), pct))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
panels = [
    (axes[0, 0], zeta_tar_truth, vmax_z,
     r"$\zeta_{tar}$  (UNfiltered truth)"),
    (axes[0, 1], zeta_tar_lp, vmax_z,
     r"$\zeta_{tar,\mathrm{lp}}$  (LP-filtered truth)"),
    (axes[0, 2], zeta_tar_truth - zeta_tar_lp, vmax_z,
     r"$\zeta_{tar}-\zeta_{tar,\mathrm{lp}}$  (lost to LP)"),
    (axes[1, 0], zeta_opt, vmax_z,
     r"$\zeta_{opt}$  (from $\Phi^0_{opt}$)"),
    (axes[1, 1], zeta_res_lp, vmax_err,
     rf"$\zeta_{{opt}}-\zeta_{{tar,\mathrm{{lp}}}}$"
     f"\nrms = {rms(zeta_res_lp):.3e},"
     f"  rel-of-LP = {rms(zeta_res_lp)/rms(zeta_tar_lp):.2%}"),
    (axes[1, 2], zeta_res, vmax_err,
     rf"$\zeta_{{opt}}-\zeta_{{tar}}$  (vs UNFILT)"
     f"\nrms = {rms(zeta_res):.3e},"
     f"  rel = {rms(zeta_res)/rms(zeta_tar_truth):.2%}"),
]
for ax, field, vm, title in panels:
    im = ax.imshow(field, origin="lower", cmap="RdBu_r",
                   vmin=-vm, vmax=vm, extent=extent)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Vorticity: filtered vs unfiltered comparison", fontsize=12)
fig.savefig(OUT / "04_vorticity_maps.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 5: phi0_opt - psi0_lpfilter map and 1D zonal-mean spectrum-vs-k
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
panels = [
    (axes[0], psi0_lpfilter, r"$\psi_{0,\mathrm{lp}}$ (LP truth)"),
    (axes[1], phi0_opt,      r"$\Phi^0_{opt}$ (recovered)"),
    (axes[2], phi0_opt - psi0_lpfilter,
     rf"$\Phi^0_{{opt}}-\psi_{{0,\mathrm{{lp}}}}$"
     f"\nrms = {rms(phi0_opt - psi0_lpfilter):.3e}"),
]
vmax = float(jnp.max(jnp.abs(psi0_lpfilter)))
for i, (ax, field, title) in enumerate(panels):
    vm = vmax if i < 2 else float(jnp.percentile(jnp.abs(panels[2][1]), 99))
    im = ax.imshow(field, origin="lower", cmap="RdBu_r",
                   vmin=-vm, vmax=vm, extent=extent)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.savefig(OUT / "05_streamfunction_maps.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Print a verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("INTERPRETATION")
print("=" * 72)

# How much of zeta_tar's energy lives above kappa_c?
frac_above = t_hi / (t_lo + t_hi)
print(f"  fraction of zeta_tar power above kappa_c = {frac_above:.2%}")
print(f"  -> the LP filter throws away this much of the truth's vorticity power.")

# Compare residual rms when measured against LP truth vs UNfiltered truth
print(f"\n  rel zeta error vs UNFILT truth = "
      f"{rms(zeta_res)/rms(zeta_tar_truth):.2%}")
print(f"  rel zeta error vs LP    truth = "
      f"{rms(zeta_res_lp)/rms(zeta_tar_lp):.2%}")
print(f"  -> if dropping to the LP truth doesn't help, the problem is NOT")
print(f"     simply LP truncation; it is K^2 amplification of the SSH residual.")

print(f"\nWrote diagnostics to: {OUT}")
print("Plots:")
for p in sorted(OUT.glob("*.png")):
    print(f"   {p.name}")
