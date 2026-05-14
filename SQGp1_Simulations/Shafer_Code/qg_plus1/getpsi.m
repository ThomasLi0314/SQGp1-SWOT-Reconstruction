function [psi0,u0,v0,u1,v1,zeta0,zeta1,delta,p1,bx,by,u1o,v1o] = getpsi(b);

% [psi0,u0,v0,u1,v1,zeta0,zeta1,delta1] = getpsi(b)
%
% Input:  grid-space buoyancy field b from sqg_plus1 simulation.  
% Returns:  O(0) geostrophic streamfunction, velocities, vorticity,
%           O(1) ageostrophic velocities, vorticity, divergence 

[nx,ny] = size(b);
kmax = nx/2 - 1;
[ikx_,iky_] = ndgrid([0:kmax -kmax-1:-1],0:kmax);
K_   = sqrt(ikx_.^2 + iky_.^2);
IK_  = 1./K_;
IK_(K_==0)=0;   % for division by K_ w/o nans
ikx_ = 1i*ikx_;
iky_ = 1i*iky_;

bk   = g2k(b);
psik = bk.*IK_;
psi0 = k2g(psik);

bx   = k2g(ikx_.*bk);
by   = k2g(iky_.*bk);
bz   = k2g(K_.*bk);
psix = k2g(ikx_.*psik);
psiy = k2g(iky_.*psik);

Rxk = K_.*g2k(b.*psix);
Ryk = K_.*g2k(b.*psiy);
Sk  = IK_.*g2k(b.*bz);

u0 = -psiy;
v0 =  psix;

u1 = -psiy.*bz - 2*by.*b + k2g(iky_.*Sk + Ryk);
v1 =  psix.*bz + 2*bx.*b - k2g(ikx_.*Sk + Rxk);
u1o = -psiy.*bz + k2g(iky_.*Sk + Ryk);
v1o =  psix.*bz - k2g(ikx_.*Sk + Rxk);

% Divergence, vorticity

zeta0 = k2g(-K_.^2.*psik);
zeta1k = ikx_.*g2k(v1) - iky_.*g2k(u1);
zeta1 = k2g(zeta1k);
delta = k2g(ikx_.*g2k(u1) + iky_.*g2k(v1));

% Pressure

psi0xx = k2g(ikx_.*ikx_.*psik);
psi0xy = k2g(ikx_.*iky_.*psik);
psi0yy = k2g(iky_.*iky_.*psik);
J = psi0xy.*(psi0xx - psi0yy); % Incorrect. 

p1 = k2g(IK_.^(2) .* ( zeta1k + g2k(2*J) ));
