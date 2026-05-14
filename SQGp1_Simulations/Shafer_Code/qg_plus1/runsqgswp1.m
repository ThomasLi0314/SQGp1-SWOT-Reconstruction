
% Initialize SW QGp1

res = 256;

% Run parameters

Bu       = 2.5;
Ro       = 0.1;
drag     = 0;
numsteps = 5000;
savestep = 100;

% x-grid
nx        = 3*res/2;
L         = 2*pi;
x         = linspace(0,L*(nx-1)/nx,nx) - L/2;
[x_,y_]   = ndgrid(x,x);

% k-grid
kmax      = nx/2 - 1; 
k         = [0:kmax -kmax-1:-1];
[kx_,ky_] = ndgrid(k,k);
K_        = sqrt(kx_.^2+ky_.^2);
K_(1,1)   = 1;


% Two possible initial conditions below:


% I.  Held et al 95 elliptical vortex initial condition
%
% qmax = 1.;
% qin  = qmax * exp(-(6*x_/L).^2-(24*y_/L).^2);


% II. Random noise at wavenumber k0 with gaussian width dk

k0 = 10;
dk = 2;
e0 = 1e-3;

f     = exp(-(K_-k0).^2/dk^2)./K_;                 
psi0k = sqrt(f) .* exp(2*pi*1i*rand(size(f)));  % Initial O(0) psi
e     = real(sum(sum((K_.^2+Bu^(-1)).*psi0k.*conj(psi0k)))) % Geostroph energy
psi0k = sqrt(e0/e) * psi0k;                           % Set energy e0
psi0k(1,1) = 0;
qk    = -(K_.^2 + Bu^(-1)) .* psi0k;
qin   = ifftn(qk,'symmetric'); % Initial PV

% check energy 
e0 
e = real(sum(sum(-conj(psi0k).*qk)))    

% Plot initial q

pcolor(qin), shading interp, axis image, colorbar

% Run model

[qout,t,ke,ape] = qgswp1(qin,Bu,Ro,drag,numsteps,savestep);

% Plot final q

pcolor(qout(:,:,end)), shading interp, axis image, colorbar
