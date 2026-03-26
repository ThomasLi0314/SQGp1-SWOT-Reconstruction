import jax
import jax.numpy as jnp
from jax import jit

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
    Phi_1_term1 = jnp.fft.fft2(0.5 * phi0_s_z**2 / Bu)
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
    F1_s_hat_z = (F1_term1 + F1_term2) / Bu

    # G1 terms
    phi0_s_x  = jnp.real(jnp.fft.ifft2(phi0_s_hat * 1j * kx))
    phi0_s_xz = jnp.real(jnp.fft.ifft2(phi0_s_hat * 1j * kx * mu))

    G1_term1 = jnp.fft.fft2(phi0_s_x * phi0_s_zz + phi0_s_xz * phi0_s_z)
    G1_term2 = -jnp.fft.fft2(phi0_s_x * phi0_s_z) * mu
    G1_s_hat_z = (G1_term1 + G1_term2) / Bu

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
