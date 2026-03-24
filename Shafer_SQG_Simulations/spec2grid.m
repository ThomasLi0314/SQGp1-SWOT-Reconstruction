function [gfield] = spec2grid(sfield,da);

% [gfield] = SPEC2GRID(sfield,da) Transform one frame of SQG model
%     output to a grided (physical) representation.  Assumes 'sfield'
%     to be up-half plane, and specifies lower half plane by conjugate
%     sym (since physical field is assumed real-valued).  Input field
%     should have dimensions (1:2*kmax+1,1:kmax+1,:,:), where
%     kmax=2^n-1, hence physical resolution will be 2^(n+1) x 2^(n+1).
%     NOTE: bottom row of the input field corresponds to ky = 0, the
%     kx<0 part is NOT assumed a priori to be conjugate- symmetric
%     with the kx>0 part.  NOTE: grid2spec(spec2grid(fk)) = fk.
%     OPTIONAL: da = true pads input with 0s before transfoming to
%     gridspace, for dealiased products.  Default is da = false.
%
%     See also GRID2SPEC and IFFT2.

if (nargin>1)
  dealias=da;
else
  dealias=false;
end

% Get number of nonsingleton dimensions
nd = ndims(sfield);  
if (nd==2)
    [n1,n2]=size(sfield);
    if (n1==1|n2==1), nd = 1; end
end

switch nd
  case 1
    if (size(sfield,1)~=1), sfield=transpose(sfield); end
    if (dealias==false)
      hres = 2*length(sfield);  % Assumes sfield def'd on 0:kmax
      fk = [0 conj(sfield(end:-1:1)) sfield(2:end)];
      fk = ifftshift(fk);
      gfield = hres*real(ifft(fk));
    elseif (dealias==true)
      kmax = length(sfield)-1;
      kmaxbig = 3*(kmax+1)/2 - 1;
      hres = 2*(kmaxbig+1);
      sfieldbig = [sfield zeros(1,(kmax+1)/2)];
      fk = [0 conj(sfieldbig(end:-1:1)) sfieldbig(2:end)];
      fk = ifftshift(fk);
      gfield = hres*real(ifft(fk));
    end
  case 2
    if (dealias==false)
      hres = size(sfield,1)+1;
      fk = fullspec(sfield);
      fk = ifftshift(fk);
      gfield = hres*hres*real(ifft2(fk));
    elseif (dealias==true)
      kmax = size(sfield,2)-1;
      kmaxbig = 3*(kmax+1)/2 - 1;
      hres = 2*(kmaxbig+1);
      sfieldbig = zeros(2*kmaxbig+1,kmaxbig+1);
      offset = (kmax+1)/2;
      sfieldbig(offset+1:offset+size(sfield,1),1:size(sfield,2)) = sfield; 
      fk = fullspec(sfieldbig);
      fk = ifftshift(fk);
      gfield = hres*hres*real(ifft2(fk));      
    end
  case 3
    if (dealias==false)
      hres = size(sfield,1)+1;
      fk = fullspec(sfield);
      for n=1:size(sfield,3)
	fk(:,:,n) = ifftshift(fk(:,:,n));
	gfield(:,:,n) = hres*hres*real(ifft2(fk(:,:,n)));
      end
    elseif (dealias==true)
      kmax = size(sfield,2)-1;
      kmaxbig = 3*(kmax+1)/2 - 1;
      hres = 2*(kmaxbig+1);
      sfieldbig = zeros(2*kmaxbig+1,kmaxbig+1,size(sfield,3));
      offset = (kmax+1)/2;
      sfieldbig(offset+1:offset+size(sfield,1),1:size(sfield,2),:) = sfield; 
      fk = fullspec(sfieldbig);
      for n=1:size(sfield,3)
	fk(:,:,n) = ifftshift(fk(:,:,n));
	gfield(:,:,n) = hres*hres*real(ifft2(fk(:,:,n)));
      end
    end
  case 4
    if (dealias==false)
      hres = size(sfield,1)+1;
      fk = fullspec(sfield);
      for z=1:size(sfield,3)
	for t=1:size(sfield,4)
	  fk(:,:,z,t) = ifftshift(fk(:,:,z,t));
	  gfield(:,:,z,t) = hres*hres*real(ifft2(fk(:,:,z,t)));
	end
      end
    elseif (dealias==true)
      kmax = size(sfield,2)-1;
      kmaxbig = 3*(kmax+1)/2 - 1;
      hres = 2*(kmaxbig+1);
      sfieldbig = zeros(2*kmaxbig+1,kmaxbig+1,size(sfield,3),size(sfield,4));
      offset = (kmax+1)/2;
      sfieldbig(offset+1:offset+size(sfield,1),1:size(sfield,2),:,:) = sfield; 
      fk = fullspec(sfieldbig);
      for z=1:size(sfield,3)
	for t=1:size(sfield,4)
	  fk(:,:,z,t) = ifftshift(fk(:,:,z,t));
	  gfield(:,:,z,t) = hres*hres*real(ifft2(fk(:,:,z,t)));
	end
      end
    end
end

% ------------------------------------------------------------------------

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