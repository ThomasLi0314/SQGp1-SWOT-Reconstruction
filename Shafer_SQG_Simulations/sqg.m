function [bout,time,ke,b2s,bmov] = sqg(bin,numsteps,savestep,dttune,nutune)

%  [bout,time,ke,b2s,bmov] = sqg(bin,numsteps,savestep,dttune,nutune)
%
%  This version is spectral and uses AB3 timestepping.
%
%  Solves equation 
%
%  b_t + J(psi,b) = diss
%  b(k,l,t) = K*psi(k,l,t), with K^2 = k^2+l^2
%
%  This is SQG at z=0, with psi in negative half space z<0
%
%  Initial condition given by input field bin, which should have nx
%  = ny = 2^n.  Model is run for 'numsteps' time steps.  Optional
%  savestep is frequency at which psi is stored (default is 100,
%  meaning run will save tau every 100 timesteps). dttune is
%  nondimensional tuning factor for timestep.  
%
%  Outputs: qout saves b at frequency savestep, so final array has
%  dimension (size(bin,1),size(bin,2),floor(numsteps/savestep)+1).
%  Variable 'time' stores the model times at which psi is saved in
%  tau_out. 
%
%  Numerical details: Model is spectal, in square domain of length
%  2*pi x 2*pi.  Nonlinear terms are done in physical space using
%  dealiased product.  Timestep dt = dttune*dx/max(U).  Uses AB3
%  timestepping with trapezoidal diffusion.
%
%  See also GRID2SPEC and SPEC2GRID.

% Set defaults if not specified
%cvecd = [min(min(tauin)) max(max(tau_in))];  % Fix color scale for movie
%dtd = .1;
%if (nargin<3), error('need at least 3 input arguments: tau_in, psi and Pe'), end
%switch nargin
%  case 7, dttune=dtd; 
%  case 6, dttune=dtd; cvec=cvecd; 
%  case 5, dttune=dtd; cvec=cvecd; taubar=[0 0];
%  case 4, dttune=dtd; cvec=cvecd; taubar=[0 0]; savestep=100;
%  case 3, dttune=dtd; cvec=cvecd; taubar=[0 0]; savestep=100; numsteps=1000;
%end
    
hv = 8; % hyperviscosity exponent

% Check for outputs requested
makemov=false;
if (nargout>4), makemov=true;  end
   
% Get and check dimensions
[nx,ny] = size(bin);
if (nx~=ny), error('must have nx = ny'); end
if (mod(log2(nx),1)~=0), error('must have nx = 2^n, n integer'); end

% Get dimensions of spectral fields -- spectral fields are stored
% on upper half-plane (bottom half plane given by conjugate
% symmetry, since grid space fields are real).
kmax = nx/2 - 1;

% Set up arrays of wavenumbers and gridpoints
[kx_,ky_] = ndgrid(-kmax:kmax,0:kmax);
K_        = sqrt(kx_.^2 + ky_.^2);
K_(kmax+1,1)=.1;  % make irrel K^2(0,0) non-zero for division 

% Initialize fields for dealiased jacobian
alphak    = exp(1i*pi*(kx_+ky_)/nx);
alphakf   = fullspec(alphak);

kcut = sqrt(8./9.)*(kmax+1);
damask = ones(size(K_));
damask(K_>kcut) = 0.;
damask(1:kmax+1,1) = 0.;

% Get initial spectral PV
bk = grid2spec(bin);
psik = bk./(K_);

% Get grid and time spacings
u   = spec2grid(-1i*ky_.*psik);
v   = spec2grid(1i*kx_.*psik);
vmag = sqrt(u.^2+v.^2);
maxU = max(vmag(:));
dx = 2*pi/nx;
dt = dttune*dx/maxU;   % Courant condition
ens = sum(sum(real(bk.*conj(bk))));
nu = nutune*dx^hv*sqrt(ens);

disp(strcat('max(|u|) = ',num2str(maxU)))
disp(strcat('dt = ',num2str(dt)))
disp(strcat('nu = ',num2str(nu)))

%tmax = numsteps*dt;

% Set up array to hold saved output
if (savestep>numsteps)
  bout = zeros(size(bin));
else
  nframes = floor(numsteps/savestep)+1;
  bout = zeros(size(bin,1),size(bin,2),nframes);
  time = zeros(1,nframes); 
  ke = zeros(1,nframes); 
  b2s = zeros(kmax,nframes);
end

% Get initial RHS and past fields (nm1 = n-1, nm2 = n-2) for AB3
rhs = -jacobk(psik,bk);
rhs_nm1 = rhs;
rhs_nm2 = rhs;

% Set counters
frame = 0;
t = 0;

% Set initial frame color and figure if movie requested
if (makemov)
  plotstuff(bin,frame);
end

% Start timestepping
n=0;
keepgoing=true;
while keepgoing
    n=n+1;
    t = t+dt;  % clock
    
    % Check for blow up or number requested steps reached
    if (n==numsteps), disp('End reached'), keepgoing=false; end
    ens = sum(sum(real(bk.*conj(bk))));
    if (ens>1e6), disp('Blow up!'), ens, keepgoing=false;, 
        bout= spec2grid(bk);       % Exit if NaN
    end
    
    % Save output at frequency savestep
    if (mod(n,savestep)==0||n==1)  
        frame = frame+1;  
        ke(frame) = sum(sum(real(K_.^2.*psik.*conj(psik))));      % KE
        b2k = real(bk.*conj(bk));
        for K=1:kmax % compute isotropic spectrum 
            b2s(K,frame) = sum(sum(b2k(K_>K-1/2 & K_<=K+1/2)));
        end
        b = spec2grid(bk);
        bout(:,:,frame) = b;    % Output saved in gridspace
        time(frame) = t;
        disp(strcat('Wrote frame >',num2str(frame),' out of >',num2str(nframes)))
        disp(strcat('KE =',num2str(ke(frame))))
        if (makemov)
            h=plotstuff(b,frame);
            bmov(frame) = getframe(h);
        end
    end
    
    % Timestep and diffuse tracer (AB3+trap - Durran 3.81)
    a1 = dt*23/12;   a2 = -dt*16/12;  a3 = dt*5/12;   % AB3 factors
    filterup = (1 - (dt/2)*nu*K_.^hv);                % trapezoidal diffusion
    filterdn = 1./(1 + (dt/2)*nu*K_.^hv);

    bk = filterdn.*(filterup.*bk + a1*rhs + a2*rhs_nm1 + a3*rhs_nm2);
    psik = bk./(K_);
    
    % Save previous rhs and get next one
    rhs_nm2 = rhs_nm1;
    rhs_nm1 = rhs;
    rhs = -jacobk(psik,bk);
    
end

% Save final step to output field if we haven't been saving along
% the way
if (savestep>numsteps)
  bout = spec2grid(bk);  
end


%-------------------------------------------------------------------

function jk = jacobk(psik,bk)

% Calculate dealiased spectral jacobian using same method as QG code.

% Calculate derivatives in k-space and make packed k-space arrays

psikxa = fullspec(damask.*(1i*kx_.*psik)).*(1 + 1i*alphakf);
psikya = fullspec(damask.*(1i*ky_.*psik)).*(1 + 1i*alphakf);
bkxa   = fullspec(damask.*(1i*kx_.*bk  )).*(1 + 1i*alphakf);
bkya   = fullspec(damask.*(1i*ky_.*bk  )).*(1 + 1i*alphakf);

% Get complex to complex transforms (imag parts of results are
% derivatives on the staggered grid)

psixa = nx^2*ifft2(ifftshift(psikxa));
psiya = nx^2*ifft2(ifftshift(psikya));
bya   = nx^2*ifft2(ifftshift(bkya));
bxa   = nx^2*ifft2(ifftshift(bkxa));
  
% Get separate products on normal grid (real part) and shifted grid
% (imag part) in x-space
  
jg  = real(psixa).*real(bya) + 1i*imag(psixa).*imag(bya) ...
    - (real(psiya).*real(bxa) + 1i*imag(psiya).*imag(bxa));
  
% Take it back to k-space

Wk  = fftshift(fft2(jg))/nx^2;
  
% Extract spectral products on grid and shifted grid, and average.

Wk_up = Wk(2:end,kmax+2:end);
Wk_dn = rot90(rot90((conj(Wk(2:end,2:kmax+2)))));
  
jk = ((1 - 1i*conj(alphak)).*Wk_up + (1 + 1i*conj(alphak)).*Wk_dn)/4;
jk(1:kmax+1,1) = 0;

end

%-------------------------------------------------------------------

function [h] = plotstuff(b,frame)

  persistent cvec
  
  if (frame==0)
    figure(11)
    axis square
    cvec = [min(b(:)) max(b(:))]
    disp('move and reshape figure 10 as desired, then press any key')
    pause
  end
  h = figure(11);
  clf
  set(gca,'fontsize',16)
  pcolor((b)')
  shading interp
  colormap(jet)
  caxis(cvec)
  colorbar
  axis image
  axis off
  hold on
  title('Bouyancy')

  %subplot(1,2,2)
  %loglog(b2s,'linewidth',2)
  %grid
  %xlabel('K')
  %title('b^2 spectrum')
  
  %plot(time(1:frame),ke(1:frame),'linewidth',3)
  %grid
  % axis([0 tmax 0 gfac*gtvar(1)])
  %axis square
  %title('Energy')
  %xlabel('time')
end

end

