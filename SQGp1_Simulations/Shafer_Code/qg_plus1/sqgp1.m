 function [bout,time,meanb,ke,kes,bmov] = sqgp1(bin,Ro,numsteps,savestep,dttune,hv,nutune)

%  [bout,time,meanb,ke,kes,bmov] = sqgp1(bin,Ro,numsteps,savestep,dttune,hv,nutune)
%
%  Solves SQG+1 (Hakim, Snyder & Muraki, JAS 2002) at z=0, 
%  with psi in negative half space z<0, for constant stratification. 
%
%  Solves:   b_t + J(psi,b) + ub_x + vb_y = nu del^hv b
%  with:     b(k,l,t) = K*psi(k,l,t), 
%            (u,v) is O(Ro) correction to velocity field 
%            K^2 = k^2+l^2
%  
%  See HSM02 (psi above is phi^0 in the paper) and getrhs() in code for details.
%
%  Initial condition given by input field bin, which should have nx
%  = ny = 2^(integer).  Model is run for 'numsteps' time steps.  savestep
%  is frequency, in timesteps at which b is output. dttune is
%  nondimensional tuning factor for timestep, and nutune is a
%  tuning factor for nu (which is set to optimally remove variance
%  at the gridscale - or at least, that's the intention).
%
%  Outputs: bout saves b at frequency savestep, so final array has
%  dimension (size(bin,1),size(bin,2),floor(numsteps/savestep)+1).
%  Variable 'time' stores the model times at which output is
%  written. meanb is area-averaged b (relevant for SQG+1).  ke is a
%  timeseries of kinetic energy,  
%
%  Numerical details: Model is spectal, in square domain of length
%  2*pi x 2*pi.  Nonlinear terms are done in physical space using
%  dealiased product via Orszag method.  Timestep dt =
%  dttune*dx/max(U).  Uses AB3 timestepping with trapezoidal hyperviscosity.
%

% Check for outputs requested
makemov=false;
if (nargout>5), makemov=true;  end
   
% Make some fields and params global for use in rhs function

global nx alphak alphakf damask ikx_ iky_ kmax K_ IK_ nkx nky

% Get and check dimensions
[nx,ny] = size(bin);
if (nx~=ny), error('must have nx = ny'); end
if (mod(log2(nx),1)~=0), error('must have nx = 2^n, n integer'); end

% Get dimensions of spectral fields -- spectral fields are stored
% on upper half-plane (bottom half plane given by conjugate
% symmetry, since grid space fields are real).
kmax = nx/2 - 1;
nkx = 2*kmax+1;
nky = kmax+1;

% Set up arrays of wavenumbers and gridpoints
[ikx_,iky_] = ndgrid(-kmax:kmax,0:kmax);
K_   = sqrt(ikx_.^2 + iky_.^2);
ikx_ = 1i*ikx_;  % only ever used for computing horizontal derivatives
iky_ = 1i*iky_;
IK_  = 1./K_;
IK_(K_==0)=0;   % for division by K_ w/o nans

% Initialize fields for spectral products with Orszag dealiasing
alphak    = exp(pi*(ikx_+iky_)/nx);
alphakf   = fulspec(alphak);
kcut      = sqrt(8./9.)*(kmax+1);
damask    = ones(size(K_));
damask(K_>kcut)    = 0.;
damask(1:kmax+1,1) = 0.;  % shouldn't that be 1:kmax?
y
% Get initial spectral PV
bk = g2k(bin);
psik = bk.*IK_;  % psik = phi^0, which has no mean, or (k,l)=(0,0) value

% Get grid and time spacings
kek = real(k2g(K_.^2.*psik.*conj(psik)));
maxU = max(sqrt(kek(:)));
dx = 2*pi/nx;
dt = dttune*dx/maxU;   % Courant condition

% Set hyperviscous coefficient .. better to make this adaptive,
% along with dt
%ens0 = sum(sum(real(K_.^4.*psik.*conj(psik))));
ens0 = sum(sum(real(bk.*conj(bk))));
nu = nutune*dx^hv*sqrt(ens0);

disp(strcat('max(|u|) = ',num2str(maxU)))
disp(strcat('dt = ',num2str(dt)))
disp(strcat('nu = ',num2str(nu)))

% Set params for AB3+trapezoidal diffusion (Durran 3.81)
% .. should switch to exponential cutoff filter
a1 = dt*23/12;   a2 = -dt*16/12;  a3 = dt*5/12;   % AB3 factors
filterup = (1 - (dt/2)*nu*K_.^hv);                % trapezoidal diffusion
filterdn = 1./(1 + (dt/2)*nu*K_.^hv);

% Set up array to hold saved output
if (savestep>numsteps)
  bout = zeros(size(bin));
else
  nframes = floor(numsteps/savestep)+1;
  bout = zeros(size(bin,1),size(bin,2),nframes);
  time = zeros(1,nframes); 
  ke = zeros(1,nframes); 
  b2s = zeros(kmax,nframes);
  meanb = zeros(1,nframes);
end

% Get initial RHS and past fields (nm1 = n-1, nm2 = n-2) for AB3
rhs = getrhs(psik,bk,Ro);
rhs_nm1 = rhs;
rhs_nm2 = rhs;

% Set counters
frame = 0;
t = 0;
n = 0;

% Set initial frame color and figure if movie requested
if (makemov)
    plotstuff(bin,frame);
end

keepgoing = true;
while keepgoing
    n = n+1;
    t = t+dt;  % clock
    
    % Check for blow up or number requested steps reached
    if (n==numsteps), disp('End reached'), keepgoing=false; end
    ens = sum(sum(real(bk.*conj(bk))));
    if (ens>1e6), disp('Blow up!'), ens, keepgoing=false;, 
        bout= real(k2g(bk));       % Exit if NaN
    end
    
    % Save output at frequency savestep
    if (mod(n,savestep)==0||n==1)  
        frame = frame+1;  
        kek = K_.^2.*psik.*conj(psik);
        ke(frame) = sum(sum(real(kek)));   
        for K=1:kmax                       % Isotropic spectrum 
            kes(K,frame) = sum(sum(kek(K_>K-1/2 & K_<=K+1/2)));
        end
        meanb(frame) = bk(K_==0);           % Same mean(b)
        bout(:,:,frame) = real(k2g(bk));    % Output saved in gridspace
        time(frame) = t;
        disp(strcat('Wrote frame >',num2str(frame),' out of >',num2str(nframes)))
        disp(strcat('KE =',num2str(ke(frame))))
        if (makemov)
            h = plotstuff(bout(:,:,frame),frame);
            bmov(frame) = getframe(h);
        end
    end
    
    % Timestep, diffuse, invert
    bk = filterdn.*(filterup.*bk + a1*rhs + a2*rhs_nm1 + a3*rhs_nm2);
    psik = bk.*IK_;
    
    % Save previous rhs and get next one
    rhs_nm2 = rhs_nm1;
    rhs_nm1 = rhs;
    rhs = getrhs(psik,bk,Ro);
    
end

% Save final step if we haven't been saving along the way
if (savestep>numsteps)
  bout = real(k2g(bk));  
end

% End main program

%-------------------------------------------------------------------
% Internal functions
%-------------------------------------------------------------------

function fk = g2k(fg)
    
    % Just for transforming gridded input field
        
    global nx kmax
    
    fk = fftshift(fft2(fg))/nx^2;
    fk = fk(2:end,kmax+2:end);      

    return
    
%-------------------------------------------------------------------

function rhsk = getrhs(psik,bk,Ro)
    
    global ikx_ iky_ K_ IK_

    b    = k2g(bk);
    bx   = k2g(ikx_.*bk);
    by   = k2g(iky_.*bk);
    bz   = k2g(K_.*bk);
    psix = k2g(ikx_.*psik);
    psiy = k2g(iky_.*psik);
    
    Rxk = K_.*gp2k(gprod(b,psix));
    Ryk = K_.*gp2k(gprod(b,psiy));
    Sk  = IK_.*gp2k(gprod(b,bz));

    u = -gprod(psiy,1+Ro*bz) + Ro*k2g(iky_.*Sk + Ryk);
    v =  gprod(psix,1+Ro*bz) - Ro*k2g(ikx_.*Sk + Rxk);
    
    rhsk = -gp2k(gprod(u,bx) + gprod(v,by));
    
%-------------------------------------------------------------------
        
function fg = k2g(fk)

   % Transform to grid space, packing fields shifted by dx/2 into
   % imaginary part
    
   global nx damask alphakf 

   fkt = fulspec(damask.*fk).*(1 + 1i*alphakf);
   fg  = nx^2*ifft2(ifftshift(fkt));

   return
   
%-------------------------------------------------------------------

function prodg = gprod(f,g)
    
    % Product of real parts and imaginary parts, loaded
    % respectively into real and imaginary part of output
            
    prodg = real(f).*real(g) + 1i*imag(f).*imag(g);

    return
    
%-------------------------------------------------------------------

function prodk = gp2k(prodg)
    
    % Transform grid field, holding product of fields, 
    % with imaginary part holding field
    % shifted by dx/2, to k-space, and average result
        
    global nx alphak kmax
  
    Wk  = fftshift(fft2(prodg))/nx^2;
    
    % Extract spectral products on grid and shifted grid, and average.
    
    Wk_up = Wk(2:end,kmax+2:end);
    Wk_dn = rot90(rot90((conj(Wk(2:end,2:kmax+2)))));
    prodk = ((1 - 1i*conj(alphak)).*Wk_up + (1 + 1i*conj(alphak)).*Wk_dn)/4;

    return

%-------------------------------------------------------------------
    
function fkf = fulspec(fk);

    %     Assumes 'fk' contains upper-half plane of spectral field, 
    %     and specifies lower half plane by conjugate 
    %     symmetry.  fk should have dimensions
    %     (1:2*kmax+1,1:kmax+1), with kmax = 2^n-1
    %     so grid resolution will be 2^(n+1) x 2^(n+1).  

    global kmax nx nkx nky

    fkf = zeros(nx,nx);
    fup = fk;  % nkx x nky
    fup(kmax:-1:1,1) = conj(fup(kmax+2:nkx,1));
    fdn = conj(fup(nkx:-1:1,nky:-1:2));
    fkf(2:nx,nky+1:nx) = fup;
    fkf(2:nx,2:nky) = fdn;
        
    return
    
%-------------------------------------------------------------------

function [h] = plotstuff(b,frame)
    
    persistent cvec
    
    if (frame==0)
        figure(10)
        axis square
        cvec = [min(b(:)) max(b(:))]
        disp('move and reshape figure 10 as desired, then press any key')
        pause
    end
    h = figure(10);
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
    
    return
    
   