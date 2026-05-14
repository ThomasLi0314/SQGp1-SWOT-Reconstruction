function [flp] = lowpass(f,Kc)

% [] =
%     
%

fk = g2k(f);
% f2k = fk.*conj(fk);

[nkx,nky] = size(fk);
kmax = nky-1;

[kx_,ky_] = ndgrid([0:kmax -kmax-1:-1],0:kmax);
K_ = sqrt(kx_.^2+ky_.^2);

mask = (K_<Kc);
% flpk = mask.*f2k;f
flpk = mask.*fk;
flp = k2g(flpk);

end



