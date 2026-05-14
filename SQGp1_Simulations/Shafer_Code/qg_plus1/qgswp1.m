function [qout,time,ke,ape] = qgswp1(qin,Bu,Ro,drag,numsteps,savestep)

%  [qout,time,ke,ape] = qgswp1(qin,Bu,Ro,drag,numsteps,savestep)
%
%  Solves equation 
%
%  q_t + (uq)_x + (vq)_y = -drag*zeta + nu*del^a q
%
%  (u,v) = del^perp psi_0 + Ro del^perp psi_1 + Ro del chi
%
%  
%  Initial condition given by input field qin, which should have
%  dimensions n x n, where n = 3*2^j, with j an integer.  
%  Model is run for 'numsteps' time steps.  
%  Savestep is frequency at which psi is stored. dttune is
%  nondimensional tuning factor for timestep.  
%
%  Outputs: qout saves q at frequency savestep, so final array has
%  dimension (size(qin,1),size(qin,2),floor(numsteps/savestep)+1).
%  Variable 'time' stores the model times at which q is saved in
%  tau_out. 
%
%  Numerical details: Model is spectal, in square domain of length
%  2*pi x 2*pi.  Nonlinear terms are done in physical space using
%  dealiased product.  Timestep dt = dttune*dx/max(U).  Uses AB3
%  timestepping with integrating factor for NL terms.
    
hv = 4;         % Dq/Dt = ... + nu*del^{2*hv} q
dttune = 0.1;
nutune = 0.2;
    
% Check dimensions
[n,~] = size(qin);

if (size(qin,2) ~= n), error('Input must be square'); end
if (mod(log2(n/3),1) ~= 0), error('Input must have dim 3 * 2^integer'); end

% Dealiasing mask
damask = ones(n,n);
damask(n/3+1:2*n/3,n/3+1:2*n/3) = 0;
damask(1,1) = 0;  % Prevent accumulation of mean fields

% Static arrays for differential operators
kmax = n/2 - 1; 
k = [0:kmax -kmax-1:-1];
[dx_,dy_] = ndgrid(k,k);

dx_    = 1i*dx_;
dy_    = 1i*dy_;
dxx_   = dx_.^2;
dyy_   = dy_.^2;
dxy_   = dx_.*dy_;
Lap_   = dxx_ + dyy_;
invLq_  = damask.*(Lap_ - Bu^(-1)).^(-1);   % Inverse PV operator
invLapLq_ = damask.*Lap_.^(-1).*invLq_;      
invLapLq_(1,1) = 0;     % Prevent NaN

% Get initial spectral q and velocities
qk = fftn(qin); 
[u,v,h,q] = get_uvq(qk);

% For setting timestep.  Effective resolution is 2*n/3 and kmax = n/3 - 1
Umax = sqrt(max(max(u.^2+v.^2)));
dx = 3*pi/n;
dt = dttune*dx/Umax;                 % Courant condition
nu = nutune*dx/(dt*(n/3-1)^(2*hv));  % Lap^(hv) hyperviscosity

disp(strcat('max(|u|) = ',num2str(Umax)))
disp(strcat('dt = ',num2str(dt)))
disp(strcat('nu = ',num2str(nu)))

% Linear operator for RHS: q_t = -(uq)_x - (uq)_y + Lin*q
Lin   = - (drag + nu*abs(Lap_.^hv)); 
eLdt  = exp(Lin*dt);
eLdt2 = eLdt.^2;

% Timestepping factors
ab2 = dt*[3/2, -1/2];
ab3 = dt*[23/12, -16/12, 5/12];

% Set up array to hold saved output
if (savestep>numsteps)
    qout = zeros(n);
else
    nframes = floor(numsteps/savestep) + 1;
    qout = zeros(n,n,nframes);
    time = zeros(1,nframes); 
    ke = zeros(1,nframes); 
    ape = zeros(1,nframes);
end

% Initialize NonLinear RHS array;  dimension 3 holds stages for AB3
Nk = zeros(n,n,3);
 
% Set counters
counter = 0;  frame = 0;  t = 0;  keepgoing = true;

while keepgoing

    % Save output at frequency savestep
    if (mod(counter,savestep)==0||counter==0)  
        frame = frame+1;  
        [ke(frame),ape(frame)] = get_energy();
        qout(:,:,frame) = q;  
        time(frame) = t;
        
        disp(strcat('Wrote frame :', num2str(frame),' out of :',num2str(nframes)))
        disp(strcat('KE =',num2str(ke(frame))))
        disp(strcat('APE =',num2str(ape(frame))))
    end

    % Save previous non-lin RHS and get next
    Nk(:,:,3) = Nk(:,:,2);
    Nk(:,:,2) = Nk(:,:,1);
    Nk(:,:,1) = -get_advection(u,v,q);

    if (counter==0)     % Euler step
        qk = eLdt.*(qk + dt*Nk(:,:,1));
    elseif (counter==1) % AB2 step
        qk = eLdt.*(qk + ab2(1)*Nk(:,:,1) + ab2(2)*eLdt.*Nk(:,:,2)); 
    else
        qk = eLdt.*(qk + ab3(1)*Nk(:,:,1) + ab3(2)*eLdt.*Nk(:,:,2) + ab3(3)*eLdt2.*Nk(:,:,3));
    end
    
    % Update RHS terms
    [u,v,h,q] = get_uvq(qk);

    % Check for blow up
    ens = mean(mean(q.^2));
    if (ens>1e6 | ens == NaN | ens == Inf), 
        disp(strcat('Blow up! ens =',num2str(ens),'Counter =',num2str(counter)))
        keepgoing = false; 
        qout(:,:,frame) = ifftn(qk,'symmetric'); 
    end      

    if (counter==numsteps), disp('End reached'), keepgoing=false; end

    counter = counter + 1;
    t = t + dt;  

end

% Save final step to output field if not saved yet
if (savestep>numsteps)
    time = t;
    qout = ifftn(qk,'symmetric');  
end

%-------------------------------------------------------------------
% Internal functions:  The 'end' at the end of the file means that
% all internal functions see variables from main, but not visa-versa
%-------------------------------------------------------------------

function [div_uqk] = get_advection(u,v,q)
    
    div_uqk = damask.*(dx_.*fftn( u.*q ) + dy_.*fftn( v.*q )); 

end

%-------------------------------------------------------------------

function [u,v,h,q] = get_uvq(qk)
    
    persistent psi0k psi0 psi0x psi0y psi0xx psi0yy psi0xy 
    persistent Ak Hk Jk psi1k chi1k
    
    psi0k = invLq_.*qk;
    
    % Transform variables to grid space for nonlinear multiplies
    psi0    = ifftn( psi0k ,'symmetric');    
    psi0x   = ifftn( dx_ .*psi0k ,'symmetric');
    psi0y   = ifftn( dy_ .*psi0k ,'symmetric');
    psi0xx  = ifftn( dxx_.*psi0k ,'symmetric');
    psi0yy  = ifftn( dyy_.*psi0k ,'symmetric');
    psi0xy  = ifftn( dxy_.*psi0k ,'symmetric');
    
    zeta0 = psi0xx + psi0yy;
    q = zeta0 - Bu^(-1) * psi0;

    % Nonlinear functions in elliptic equations for potentials
    Ak =  damask.*fftn( psi0.*q );
    Hk =  damask.*fftn( psi0xx.*psi0yy - psi0xy.^2 );
    Jk = -damask.*(dx_.*fftn( psi0y.*zeta0 ) + dy_.*fftn( psi0x.*zeta0 ));
   
    psi1k = Bu * invLapLq_ .* ( Lap_.*Ak + 2*Hk );
    chi1k = Bu * invLapLq_ .* Jk;
    h1k   = invLq_ .* ( Ak + 2*Bu*Hk );
    
    u = -psi0y + Ro * ifftn( -dy_.*psi1k + dx_.*chi1k ,'symmetric');
    v =  psi0x + Ro * ifftn(  dx_.*psi1k + dy_.*chi1k ,'symmetric');
    h =  psi0 / Bu + Ro*ifftn(h1k, 'symmetric'); 
    
end

%-------------------------------------------------------------------

function [ke,ape] = get_energy()
    
% Parseval:  sum(sum(fx.*fx)) = sum(sum(fk.*conj(fk))) / n^2
    
    ke  = 0.5 * sum(sum(u.*u + v.*v));
    ape = 0.5/Bu * sum(sum(h.*h));

end

%-------------------------------------------------------------------

end % End of entire function

    

