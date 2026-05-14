"""Diagnose why the GLS+LP-projected inversion gives a BAD unweighted-SSH
reconstruction (rel-err ≈ 127%) even though the loss converges.

Hypothesis: the SQG+1 forward map's nonlinear term  eps * p1(LP{phi0})  produces
content in [kappa_c, 2*kappa_c] that the strictly-band-limited target
(eta_tar_lpfilter) does not contain. The GLS weight downweights that band, so
the optimizer doesn't try to fit it. But it DOES show up in real-space when
we plot eta_opt = forward_ssh(phi0_opt) without any post-filter.

Strategy:
  1. Reproduce a short (~600 iter) GLS+LP inversion from the .mat data.
  2. Examine spectra of phi0_opt, eta_opt, residual.
  3. Show that LP-filtering eta_opt also drops the residual dramatically.
  4. Show the GLS-weighted residual is small even when unweighted is huge.
"""
from pathlib import Path
import sys
import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import jaxopt

HERE     = Path(__file__).resolve().parent
SQG_ROOT = HERE.parents[3] / "SQG_Simulations" / "Python_Spectral_Toy_model"
sys.path.insert(0, str(SQG_ROOT))
sys.path.insert(0, str(SQG_ROOT / "Benchmark_v4"))

from physics_functions import calculate_surface_u, forward_ssh
from cost_functions import cost_function_fmin_gls
from spectrum_diagnostics import azimuthal_sum

OUT = HERE
MAT = SQG_ROOT / "Benchmark_v4" / "data" / "sqgp1_for_thomas.mat"

# ---------------------------------------------------------------------------
# Load and replicate the notebook setup
# ---------------------------------------------------------------------------
mat = sio.loadmat(MAT)
psi0n = jnp.asarray(mat["psi0n"], dtype=jnp.float64)
p1n   = jnp.asarray(mat["p1n"],   dtype=jnp.float64)
zeta0n = jnp.asarray(mat["zeta0n"], dtype=jnp.float64)
zeta1n = jnp.asarray(mat["zeta1n"], dtype=jnp.float64)
Ro_sys = float(mat["Ro"].squeeze()); epsilon = Ro_sys
Bu = 1.0
Nx, Ny = psi0n.shape
print(f"Nx,Ny = {Nx},{Ny}   Ro = {Ro_sys}")

Lx = Ly = 2.0 * jnp.pi
dx = Lx / Nx
dk = 1.0
k_zonal = jnp.concatenate([jnp.arange(Nx//2), jnp.arange(-Nx//2, 0)]) * dk
l_merid = jnp.concatenate([jnp.arange(Ny//2), jnp.arange(-Ny//2, 0)]) * dk
k_zonal = k_zonal.at[Nx//2].set(0.0)
l_merid = l_merid.at[Ny//2].set(0.0)
kx, ky = jnp.meshgrid(k_zonal, l_merid, indexing='ij')
K2 = kx**2 + ky**2
K = jnp.sqrt(K2)
inv_K2 = jnp.where(K2 > 0, 1.0/K2, 0.0)
mu = jnp.sqrt(Bu) * K
inv_mu = jnp.where(mu > 0, 1.0/mu, 0.0)

eta_tar_full = psi0n + Ro_sys * p1n

# ---------------------------------------------------------------------------
# LP filter
# ---------------------------------------------------------------------------
kappa_c = 100
dx_lores = float(Lx) / float(Nx)
filter_lp = jnp.where(K < kappa_c,
                     1.0,
                     jnp.exp(-23.6 * (K - kappa_c)**4 * dx_lores**4))

def apply_lp(field_phys):
    f_hat = jnp.fft.fft2(field_phys) * filter_lp
    f_hat = f_hat.at[Nx//2, :].set(0.0).at[:, Ny//2].set(0.0)
    return jnp.real(jnp.fft.ifft2(f_hat))

psi0_lpfilter   = apply_lp(psi0n)
eta_tar_lpfilter = apply_lp(eta_tar_full)
print(f"rms eta_tar_lpfilter = {float(jnp.sqrt(jnp.mean(eta_tar_lpfilter**2))):.4e}")
print(f"rms psi0_lpfilter    = {float(jnp.sqrt(jnp.mean(psi0_lpfilter**2))):.4e}")

# ---------------------------------------------------------------------------
# GLS weight
# ---------------------------------------------------------------------------
gls_p   = 4
gls_K_c = float(kappa_c)
S_n     = 1.0 + (K / gls_K_c) ** gls_p
weight  = 1.0 / S_n

# ---------------------------------------------------------------------------
# Run a SHORT inversion (CPU; ~600 iters)
# ---------------------------------------------------------------------------
@jit
def loss_fn(phi0_flat, target):
    phi0_2d = phi0_flat.reshape(Nx, Ny)
    return cost_function_fmin_gls(
        phi0_2d, kx, ky, mu, inv_mu, Bu, Ro_sys, K2, inv_K2, target,
        weight, filter_lp,
    )
grad_fn = jit(jax.grad(loss_fn))

target_hat   = jnp.fft.fft2(eta_tar_lpfilter)
phi0_flat    = psi0_lpfilter.ravel()

NUM_ITERS = 1500
solver = jaxopt.LBFGS(fun=loss_fn, maxiter=NUM_ITERS, tol=1e-8)
state  = solver.init_state(phi0_flat, target_hat)
params = phi0_flat

print("\nRunning GLS+LP inversion ...")
t0 = time.time()
loss_hist = []
for it in range(NUM_ITERS):
    params, state = solver.update(params, state, target_hat)
    if (it + 1) % 100 == 0 or it == 0:
        l = float(loss_fn(params, target_hat))
        g = float(jnp.linalg.norm(grad_fn(params, target_hat)))
        loss_hist.append((it + 1, l, g))
        print(f"  iter {it+1:4d}  loss={l:.4e}  |grad|={g:.4e}")
print(f"  elapsed = {time.time()-t0:.1f} s")

phi0_opt_raw = params.reshape(Nx, Ny)
phi0_opt_lp_hat = jnp.fft.fft2(phi0_opt_raw) * filter_lp
phi0_opt_lp_hat = phi0_opt_lp_hat.at[Nx//2, :].set(0.0).at[:, Ny//2].set(0.0)
phi0_opt_lp = jnp.real(jnp.fft.ifft2(phi0_opt_lp_hat))

# ---------------------------------------------------------------------------
# Examine eta_opt: with vs without filter_lp inside forward_ssh, AND post-LP
# ---------------------------------------------------------------------------
def fwd(phi_phys, with_filter):
    phi_hat = jnp.fft.fft2(phi_phys)
    eta_hat = forward_ssh(phi_hat, kx, ky, mu, inv_mu, K2, inv_K2, Bu, Ro_sys,
                          filter_lp=(filter_lp if with_filter else None))
    return jnp.real(jnp.fft.ifft2(eta_hat)), eta_hat

# 1) The forward as the LOSS sees it (with filter_lp)
eta_opt_loss, eta_opt_loss_hat = fwd(phi0_opt_raw, with_filter=True)

# 2) The forward as the POST-ANALYSIS plot computes it (no filter_lp,
#    on the LP-projected phi0_opt)
eta_opt_plot, eta_opt_plot_hat = fwd(phi0_opt_lp, with_filter=False)

# 3) Post-LP-filter the eta from (2) — strip the nonlinear high-k overshoot
eta_opt_postlp = apply_lp(eta_opt_plot)
eta_opt_postlp_hat = jnp.fft.fft2(eta_opt_postlp)

# Initial-guess forward
eta_init, eta_init_hat = fwd(psi0_lpfilter, with_filter=False)

def rms(x): return float(jnp.sqrt(jnp.mean(jnp.asarray(x)**2)))
def relerr(res, ref): return rms(res) / rms(ref)

residuals = {
    "init       (no filter)":         eta_init - eta_tar_lpfilter,
    "opt loss   (filter_lp on phi0)": eta_opt_loss - eta_tar_lpfilter,
    "opt plot   (no filter, LP phi0)": eta_opt_plot - eta_tar_lpfilter,
    "opt post-LP (LP{eta_opt})":      eta_opt_postlp - eta_tar_lpfilter,
}
print("\n" + "=" * 72)
print(f"SSH residual rms / rel-err  vs eta_tar_lpfilter (rms={rms(eta_tar_lpfilter):.4e})")
print("=" * 72)
for name, r in residuals.items():
    print(f"  {name:35s}  rms={rms(r):.4e}   rel={relerr(r, eta_tar_lpfilter):.4%}")

# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------
norm = (Nx * Ny) ** 2
def psd1d(field_hat):
    p2d = jnp.abs(field_hat) ** 2 / norm
    k_axis, p1d = azimuthal_sum(p2d, K, dk)
    return k_axis, p1d

k_axis, S_eta_tar  = psd1d(target_hat)
_,      S_eta_loss = psd1d(eta_opt_loss_hat)
_,      S_eta_plot = psd1d(eta_opt_plot_hat)
_,      S_eta_postlp = psd1d(eta_opt_postlp_hat)

# Residual spectra
_, S_res_loss   = psd1d(eta_opt_loss_hat   - target_hat)
_, S_res_plot   = psd1d(eta_opt_plot_hat   - target_hat)
_, S_res_postlp = psd1d(eta_opt_postlp_hat - target_hat)

# phi0 spectrum
_, S_phi0_lp = psd1d(phi0_opt_lp_hat)
_, S_psi0_lp = psd1d(jnp.fft.fft2(psi0_lpfilter))

# ---------------------------------------------------------------------------
# Vorticity check
# ---------------------------------------------------------------------------
def zeta_from(phi_hat):
    u, v = calculate_surface_u(phi_hat, mu, inv_mu, kx, ky, K2, inv_K2, Ro_sys, Bu)
    return jnp.real(jnp.fft.ifft2(1j*kx*jnp.fft.fft2(v) - 1j*ky*jnp.fft.fft2(u)))

zeta_tar_full = zeta0n + epsilon * zeta1n
zeta_tar_lp   = apply_lp(zeta_tar_full)
zeta_opt      = zeta_from(phi0_opt_lp_hat)

print("\n" + "=" * 72)
print("VORTICITY")
print("=" * 72)
print(f"  rms zeta_tar (UNFILT) = {rms(zeta_tar_full):.4e}")
print(f"  rms zeta_tar_lp       = {rms(zeta_tar_lp):.4e}")
print(f"  rms zeta_opt          = {rms(zeta_opt):.4e}")
print(f"  rel err vs UNFILT     = {relerr(zeta_opt - zeta_tar_full, zeta_tar_full):.4%}")
print(f"  rel err vs LP truth   = {relerr(zeta_opt - zeta_tar_lp,   zeta_tar_lp  ):.4%}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
mask = k_axis > 0
k_plot_max = Nx // 2

# Plot 1: eta and residual spectra
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
ax.loglog(k_axis[mask], S_eta_tar[mask],   lw=1.5, color="k",
          label=r"$|\hat\eta_{tar,\mathrm{lp}}|^2$  target")
ax.loglog(k_axis[mask], S_eta_loss[mask],  lw=1.2, color="C0",
          label=r"$|\hat\eta_{opt}|^2$  loss-internal (filter_lp on phi0)")
ax.loglog(k_axis[mask], S_eta_plot[mask],  lw=1.2, color="C3", ls="--",
          label=r"$|\hat\eta_{opt}|^2$  post-analysis (no filter)")
ax.loglog(k_axis[mask], S_eta_postlp[mask], lw=1.0, color="C2",
          label=r"$|\hat\eta_{opt,\mathrm{post-LP}}|^2$  (post-LP'd)")
ax.loglog(k_axis[mask], S_res_plot[mask],  lw=1.0, color="C3", ls=":",
          label=r"residual (post-analysis)")
ax.loglog(k_axis[mask], S_res_postlp[mask], lw=1.0, color="C2", ls=":",
          label=r"residual (post-LP'd)")
ax.axvline(kappa_c,   color="grey", lw=0.6, ls=":", label=r"$\kappa_c$")
ax.axvline(2*kappa_c, color="grey", lw=0.6, ls="-.", label=r"$2\kappa_c$")
ax.set_xlim(1, k_plot_max); ax.set_ylim(1e-30, 1e-3)
ax.set_xlabel("wavenumber  $k$"); ax.set_ylabel("radial PSD")
ax.set_title("SSH spectra: target, opt as loss sees it, opt as plot sees it, post-LP")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, loc="lower left")
fig.savefig(OUT / "01_eta_spectra.png", dpi=150)
plt.close(fig)

# Plot 2: phi0 spectrum
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
ax.loglog(k_axis[mask], S_psi0_lp[mask], lw=1.4, color="k",
          label=r"$|\hat\psi_{0,\mathrm{lp}}|^2$  init guess")
ax.loglog(k_axis[mask], S_phi0_lp[mask], lw=1.2, color="C3",
          label=r"$|\hat\Phi^0_{opt}|^2$  recovered (LP-projected)")
ax.axvline(kappa_c, color="grey", lw=0.6, ls=":", label=r"$\kappa_c$")
ax.set_xlim(1, k_plot_max)
ax.set_xlabel("wavenumber  $k$"); ax.set_ylabel("radial PSD")
ax.set_title("Streamfunction spectra")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9, loc="lower left")
fig.savefig(OUT / "02_phi0_spectrum.png", dpi=150)
plt.close(fig)

# Plot 3: maps — eta_opt with vs without post-LP
extent = [0, float(Lx), 0, float(Ly)]
vmax = float(jnp.max(jnp.abs(eta_tar_lpfilter)))
fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
panels = [
    (axes[0,0], eta_tar_lpfilter, vmax, r"target $\eta_{tar,\mathrm{lp}}$"),
    (axes[0,1], eta_opt_plot,     vmax, r"$\eta_{opt}$ (no post-LP)"),
    (axes[0,2], eta_opt_postlp,   vmax, r"$\eta_{opt}$ (post-LP applied)"),
    (axes[1,0], eta_opt_loss - eta_tar_lpfilter,
        float(jnp.percentile(jnp.abs(eta_opt_loss - eta_tar_lpfilter), 99)),
        f"residual (loss-internal)\nrms={rms(eta_opt_loss-eta_tar_lpfilter):.3e}"),
    (axes[1,1], eta_opt_plot - eta_tar_lpfilter,
        float(jnp.percentile(jnp.abs(eta_opt_plot - eta_tar_lpfilter), 99)),
        f"residual (plot, no LP)\nrms={rms(eta_opt_plot-eta_tar_lpfilter):.3e}"),
    (axes[1,2], eta_opt_postlp - eta_tar_lpfilter,
        float(jnp.percentile(jnp.abs(eta_opt_postlp - eta_tar_lpfilter), 99)),
        f"residual (post-LP applied)\nrms={rms(eta_opt_postlp-eta_tar_lpfilter):.3e}"),
]
for ax, field, vm, title in panels:
    im = ax.imshow(field, origin="lower", cmap="RdBu_r",
                   vmin=-vm, vmax=vm, extent=extent)
    ax.set_title(title, fontsize=10); ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(
    "eta_opt: stripes from nonlinear-spread content disappear after LP-filtering",
    fontsize=11)
fig.savefig(OUT / "03_maps.png", dpi=150)
plt.close(fig)

print(f"\nWrote diagnostics to {OUT}")
for p in sorted(OUT.glob("*.png")):
    print(f"   {p.name}")
