"""
Reproduce the A1_nothing initial-loss issue EXACTLY as the notebook does,
including the dealias_nyquist step applied only to eta_tar_hat.
"""
import os
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from physics_functions import forward_ssh

Ro_sys  = 1e-1

mat_path = (
    "d:/Documents/College/Research/Oceangrophy/Shafer_Project/"
    "SQG_Simulations/Shafer Simulation output/"
    "20260327_232322_Shafer_Simulation_random_IC/"
    "20260327_232322_Shafer_Simulation_random_IC.mat"
)
mat = sio.loadmat(mat_path)
b_s = mat["bout"][:, :, -1]
b_s_normalized = b_s / jnp.max(jnp.abs(b_s))
Nx, Ny = b_s_normalized.shape
print(f"Grid: {Nx}x{Ny}")

Lx = 2.0 * jnp.pi; Ly = 2.0 * jnp.pi; Bu = 1.0
k_zonal     = jnp.concatenate([jnp.arange(Nx//2), jnp.arange(-Nx//2, 0)])
l_meridional = jnp.concatenate([jnp.arange(Ny//2), jnp.arange(-Ny//2, 0)])
k_zonal      = k_zonal.at[Nx//2].set(0.0)
l_meridional = l_meridional.at[Ny//2].set(0.0)
kx, ky = jnp.meshgrid(k_zonal, l_meridional, indexing='ij')
K2 = kx**2 + ky**2
K  = jnp.sqrt(K2)
inv_K2 = jnp.where(K2 > 0, 1.0 / K2, 0.0)
mu = jnp.sqrt(Bu) * K
inv_mu = jnp.where(mu > 0, 1.0 / mu, 0.0)

def dealias_nyquist(field_hat, Nx, Ny):
    field_hat = field_hat.at[Nx // 2, :].set(0.0)
    field_hat = field_hat.at[:, Ny // 2].set(0.0)
    return field_hat

# Notebook cell 404aa036 exactly
b_s_hat = jnp.fft.fft2(b_s_normalized)
b_s_hat = dealias_nyquist(b_s_hat, Nx, Ny)
phi0_s_hat = b_s_hat * inv_mu
phi0_s_hat = dealias_nyquist(phi0_s_hat, Nx, Ny)
phi0_s = jnp.real(jnp.fft.ifft2(phi0_s_hat))
ssh_qg = phi0_s
ssh_qg_hat = phi0_s_hat

# Target: forward_ssh THEN dealias
eta_tar_hat_raw   = forward_ssh(ssh_qg_hat, kx, ky, mu, inv_mu, K2, inv_K2, Bu, Ro_sys)
eta_tar_hat       = dealias_nyquist(eta_tar_hat_raw, Nx, Ny)

# Guess forward — what the cost function evaluates
phi0_s_hat_guess = jnp.fft.fft2(ssh_qg)
eta_guess_hat    = forward_ssh(phi0_s_hat_guess, kx, ky, mu, inv_mu, K2, inv_K2, Bu, Ro_sys)
# NO dealias here — mimics the cost function

# Key test: is forward_ssh injecting Nyquist content?
print()
print("── Nyquist content after forward_ssh (input had none) ──")
print(f"  ||eta_tar_hat_raw  Nyquist row||  = {float(jnp.linalg.norm(eta_tar_hat_raw[Nx//2, :])):.4e}")
print(f"  ||eta_tar_hat_raw  Nyquist col||  = {float(jnp.linalg.norm(eta_tar_hat_raw[:, Ny//2])):.4e}")
print(f"  ||eta_guess_hat    Nyquist row||  = {float(jnp.linalg.norm(eta_guess_hat[Nx//2, :])):.4e}")
print(f"  ||eta_guess_hat    Nyquist col||  = {float(jnp.linalg.norm(eta_guess_hat[:, Ny//2])):.4e}")

diff = eta_guess_hat - eta_tar_hat
print()
print("── Decompose the mismatch ──")
print(f"  ||eta_guess_hat - eta_tar_hat||         = {float(jnp.linalg.norm(diff)):.4e}")
# How much of the mismatch lives exclusively at Nyquist?
nyq_mask = jnp.zeros_like(diff, dtype=bool).at[Nx//2, :].set(True).at[:, Ny//2].set(True)
diff_nyq = jnp.where(nyq_mask, diff, 0.0)
diff_not = jnp.where(nyq_mask, 0.0, diff)
print(f"  Nyquist part of mismatch                = {float(jnp.linalg.norm(diff_nyq)):.4e}")
print(f"  Non-Nyquist part of mismatch            = {float(jnp.linalg.norm(diff_not)):.4e}")

loss_val = float(jnp.sum(jnp.abs(diff)**2) / diff.size)
print()
print(f"  Initial loss (A1, as notebook computes) = {loss_val:.4e}")

# And the fix: dealias the guess too
eta_guess_hat_deal = dealias_nyquist(eta_guess_hat, Nx, Ny)
diff_fixed = eta_guess_hat_deal - eta_tar_hat
loss_fixed = float(jnp.sum(jnp.abs(diff_fixed)**2) / diff_fixed.size)
print(f"  Initial loss (with guess dealiased too) = {loss_fixed:.4e}")
