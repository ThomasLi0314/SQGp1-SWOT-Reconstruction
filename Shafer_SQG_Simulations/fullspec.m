function sfield = fullspec(hfield)

% sfield = FULLSPEC(hfield)  
%     Assumes 'hfield' to contain upper-half plane of spectral field, 
%     and specifies lower half plane by conjugate 
%     symmetry (since physical field is assumed real-valued).  'hfield'
%     should have dimensions (1:2*kmax+1,1:kmax+1,:,:), kmax = 2^n-1,
%     hence physical resolution will be 2^(n+1) x 2^(n+1) x nz.  
%     NOTE:  The bottom row of the input field corresponds to ky = 0,
%     the kx<0 part is NOT assumed a priori to be conjugate-
%     symmetric with the kx>0 part.


nkx = size(hfield,1);  nky = size(hfield,2);
if (nkx+1 ~= 2*nky) 
   error('Not a spectral input field.')
end

hres = nkx+1;
kmax = nky-1;
fk = zeros(hres,hres,size(hfield,3),size(hfield,4));

fup = hfield;
fup(kmax:-1:1,1,:,:) = conj(fup(kmax+2:nkx,1,:,:));
%fup(kmax+1,1,:,:) = 0;
fdn = conj(fup(nkx:-1:1,nky:-1:2,:,:));
fk(2:hres,nky+1:hres,:,:) = fup;
fk(2:hres,2:nky,:,:) = fdn;
sfield = fk;
