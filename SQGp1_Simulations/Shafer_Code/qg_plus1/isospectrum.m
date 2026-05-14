function [f2s] = isospectrum(f);

% [f2s] = isospectrum(f) 
%     Azimuthally integrate the Fourier transform of f^2  
%     f2k = \int_0^{2\pi} fk*con(fk) K d\theta
%

fk = g2k(f);
f2k = fk.*conj(fk);

[nkx,nky] = size(fk);
kmax = nky-1;

[kx_,ky_] = ndgrid([0:kmax -kmax-1:-1],0:kmax);
K_ = sqrt(kx_.^2+ky_.^2);
f2s = zeros([kmax 1]);

for K = 1:kmax
    mask = (floor(K_+.5)-K)==0;
    f2s(K) = sum(mask(:).*f2k(:));
end 


