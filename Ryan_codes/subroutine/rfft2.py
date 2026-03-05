import jax.numpy as jnp
import numpy as np

def rfft2(inp):
    # return jnp.fft.rfft2(inp)/(inp.size/2)
    # return jnp.fft.rfft2(inp)
    # return jnp.fft.rfft2(inp,norm='ortho')/(inp.shape[0]/2)
    return jnp.fft.rfft2(inp,norm='forward')

def irfft2(inp):
    # return jnp.fft.irfft2(inp)*(inp.size)
    # return jnp.fft.irfft2(inp)
    # return jnp.fft.irfft2(inp,norm='ortho')*(inp.shape[0]/2)
    return jnp.fft.irfft2(inp,norm='forward')