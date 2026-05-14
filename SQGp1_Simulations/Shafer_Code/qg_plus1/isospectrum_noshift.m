function [FK] = isospectrum_noshift(Fkl);

% [FK] = ISOSPECTRUM(Fkl) 
%     Azimuthally integrate spectral field Fkl:  
%     FK = \int_0^{2\pi} Fkl K d\theta
%
%     Fkl fft2(F), with size(F) = [nx,nx], nx a power of 2, with no
%     dummy values removed and no shifting. 

[nkx,nky] = size(Fkl);
kmax = nky-1;

[kx_,ky_] = ndgrid([0:kmax -kmax-1:-1],0:kmax);
K_ = sqrt(kx_.^2+ky_.^2);
FK = zeros([kmax 1]);

for K = 1:kmax
    mask = (floor(K_+.5)-K)==0;
    FK(K) = sum(mask(:).*Fkl(:));
end 


