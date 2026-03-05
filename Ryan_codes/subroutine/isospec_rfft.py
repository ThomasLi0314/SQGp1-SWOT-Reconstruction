import numpy as np

def isospec_rfft(q_mag2d):
    N = np.min(np.array(q_mag2d.shape))
    q_spec = np.zeros(int(N))
    kx = np.fft.rfftfreq((q_mag2d.shape[1]-1)*2,d=1)
    ky = np.fft.fftfreq(q_mag2d.shape[0],d=1)
    kx2, ky2 = np.meshgrid(kx/kx[1],ky/ky[1])
    
    for ki in range(0,int(N)):
        mask = (np.floor( (np.sqrt(kx2**2+ky2**2)+1/2) )) == ki
        mask = mask/2; 
        q_spec[ki] = (mask*q_mag2d).sum()
    
    q_spec[0] /= 2
    
    return q_spec