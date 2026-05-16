import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnums=(2,))
def lowpass(f, Kc, sharp=True):
    """
    Isotropic spectral low-pass filter (Python port of qg_plus1/lowpass.m).

    Mirrors the MATLAB style: forward FFT -> build radial wavenumber K
    -> mask (K < Kc) -> inverse FFT.  The MATLAB version works on the
    half-plane real-FFT representation via g2k/k2g; here we use the full
    2D FFT (jnp.fft.fft2) consistent with the rest of Benchmark_v4.

    Parameters
    ----------
    f : (Nx, Ny) real array
        Physical-space field to be filtered.
    Kc : float
        Cutoff wavenumber (same units as the integer wavenumbers built
        from dk = dl = 1, matching the notebook's grid setup).
    sharp : bool, optional
        If True (default), use a hard spectral mask K < Kc, exactly like
        lowpass.m.  If False, kill only the Nyquist rows/columns (useful
        as a dealiasing pass).

    Returns
    -------
    flp : (Nx, Ny) real array
        Low-pass-filtered field.
    """
    Nx, Ny = f.shape

    # Wavenumber grid: same convention as Field_Verification.ipynb
    # k = [0, 1, ..., N/2-1, -N/2, -N/2+1, ..., -1]
    kx = jnp.concatenate([jnp.arange(Nx // 2), jnp.arange(-Nx // 2, 0)])
    ky = jnp.concatenate([jnp.arange(Ny // 2), jnp.arange(-Ny // 2, 0)])
    kx_, ky_ = jnp.meshgrid(kx, ky, indexing='ij')
    K_ = jnp.sqrt(kx_ ** 2 + ky_ ** 2)

    fk = jnp.fft.fft2(f)

    if sharp:
        mask = (K_ < Kc).astype(fk.dtype)
        flpk = mask * fk
    else:
        flpk = fk

    # Kill Nyquist rows/columns (matches dealias_nyquist in the notebook)
    flpk = flpk.at[Nx // 2, :].set(0.0)
    flpk = flpk.at[:, Ny // 2].set(0.0)

    flp = jnp.real(jnp.fft.ifft2(flpk))
    return flp
