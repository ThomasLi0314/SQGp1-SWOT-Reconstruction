# import jax.numpy as jnp
import numpy as np
from rfft2 import rfft2, irfft2

# dx is in m
def kxky(phi0s__,dx):
    kx = np.fft.rfftfreq(phi0s__.shape[0],d=dx)*(2*np.pi)
    ky = np.fft.fftfreq (phi0s__.shape[0],d=dx)*(2*np.pi)

    return kx,ky

################################
def U_SQGp1__func(phi0s__,dx,eps):
    kx,ky = kxky(phi0s__,dx)

    PHI0_y__ = (phi0s__.T*1j*ky).T

    if eps != 0:
        PHIp1__eval = PHIp1__(phi0s__)
        PHIy_p1__ = (PHIp1__eval.T*1j*ky).T
        FZ_p1__eval = FZ_p1__(phi0s__)
        # add low pass!
        return -PHI0_y__+1*eps*(-PHIy_p1__-FZ_p1__eval)
    
    return -PHI0_y__


def V_SQGp1__func(phi0s__,dx,eps):
    kx,ky = kxky(phi0s__,dx)

    PHI0_x__ = phi0s__*1j*kx
    
    if eps != 0:
        PHIp1__eval = PHIp1__(phi0s__)
        PHIx_p1__ = PHIp1__eval*1j*kx
        GZ_p1__eval = GZ_p1__(phi0s__)
        # add low pass!
        return PHI0_x__+1*eps*(PHIx_p1__-GZ_p1__eval)
    
    return PHI0_x__

################################

def zeta_fuv__func(phi0s__,dx,eps):
    kx,ky = kxky(phi0s__,dx)

    Vx__ =  V_SQGp1__func(phi0s__,dx,eps) *1j*kx
    Uy__ = (U_SQGp1__func(phi0s__,dx,eps) .T*1j*ky).T
    
    return Vx__-Uy__


def strain_fuv(phi0s__):
    Ux_p1 = irfft2( U_0p1__(phi0s__)*1j*kx  *lowpass_pres**1)
    Vx_p1 = irfft2( V_0p1__(phi0s__)*1j*kx  *lowpass_pres**1)
    Uy_p1 = irfft2( (U_0p1__(phi0s__).T*1j*ky).T  *lowpass_pres**1)
    Vy_p1 = irfft2( (V_0p1__(phi0s__).T*1j*ky).T  *lowpass_pres**1)
    
    strain_fuvp1 = np.sqrt(  (Ux_p1-Vy_p1)**2+(Vx_p1+Uy_p1)**2  )
    
    strain_fuv__ = rfft2(strain_fuvp1)

    return strain_fuv__


################################
def PHIp1__(phi0s__):
    # phi0s__ *= phi0s__*lowpass_pres

    phi0s_z = irfft2(phi0s__*Ka)
    term_1 = rfft2(1/2*phi0s_z**2)

    phi0s_zz = irfft2(phi0s__*Ka*Ka)
    term_2 = -rfft2(phi0s_z*phi0s_zz)/Kaa

    return term_1+term_2

def FZ_p1__(phi0s__):
    # phi0s__ *= phi0s__*lowpass_pres

    phi0s_y = irfft2((phi0s__.T*1j*ky).T)
    phi0s_zz = irfft2(phi0s__*Ka*Ka)
    phi0s_yz = irfft2((phi0s__.T*1j*ky*Ka.T).T)
    phi0s_z = irfft2(phi0s__*Ka)
    term_1 = rfft2(phi0s_y*phi0s_zz+phi0s_yz*phi0s_z)
    
    term_2 = -(rfft2(phi0s_z*phi0s_y).T*Ka.T).T

    return (term_1+term_2)

def GZ_p1__(phi0s__):
    # phi0s__ *= phi0s__*lowpass_pres

    phi0s_x = irfft2(phi0s__*1j*kx)
    phi0s_zz = irfft2(phi0s__*Ka*Ka)
    phi0s_xz = irfft2(phi0s__*1j*kx*Ka)
    phi0s_z = irfft2(phi0s__*Ka)
    term_1 = -rfft2(phi0s_x*phi0s_zz+phi0s_xz*phi0s_z)
    
    term_2 = +(rfft2(phi0s_z*phi0s_x).T*Ka.T).T

    return (term_1+term_2)