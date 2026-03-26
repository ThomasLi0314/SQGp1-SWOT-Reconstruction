import jax
import jax.numpy as jnp

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
