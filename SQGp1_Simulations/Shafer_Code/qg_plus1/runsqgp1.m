
kmax = 255;  % 512^2 in grid space
k0 = 10;     % initial wavenumber peak
delk = 3;    % width of initial peak
E0 = .1;
L = 2*pi;

% Wavenumber grid
[kx_,ky_] = ndgrid(-kmax:kmax,0:kmax);
K_ = sqrt(kx_.^2+ky_.^2);
K_(kmax+1,1) = 1;             % make irrelevant wavnumber 0=1

% Initialize streamfunction in spectral space, set total energy=1
rng(10);
f = exp(-(K_-k0).^2/delk^2);
psik = sqrt(f)./K_.*exp(2*pi*1i*rand(size(f)));
e = real(sum(sum(K_.*psik.*conj(psik))));
psik = E0*psik/sqrt(e);  % spectral streamfunction

% Get inital vorticity in grid space
bin = spec2grid(-K_.*psik);


Ro = .05;
numsteps = 50000;
savestep = 500;
dttune = .05;
hv = 8;
nutune = .1;
%[bout0,time0,ke0,b2s0] = sqg(bin,5000,500,dttune,nutune);
%  sqg(bin,numsteps,savestep,dttune,nutune)
%[qout,t,ke,ape,qmov] = qg2dk(bin,0,0,0,5000,50,.2,.2);
%  sqgp1(bin,Ro,numsteps,savestep,dttune,hv,nutune)
[b,t,meanb,ke,b2s] = sqgp1(bout1(:,:,100),Ro,5000,100,dttune,hv,nutune);

figure(1)
clf
loglog(real(b2s1))
grid
axis([1 100 1e-7 1])

figure(2)
clf
for j=1:size(bout1,3)
    pcolor(bout1(:,:,j)), shading interp, axis image
    pause(.1)
end
