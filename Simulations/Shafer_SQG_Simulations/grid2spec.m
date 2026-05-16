function [sfield] = grid2spec(gfield,da);

% [sfield] = GRID2SPEC(gfield,da)
%     Transform one frame of SQG model output
%     to a spectral representation.  Yields 'sfield' as upper-
%     half plane (lower half plane given by conjugate symmetry,
%     since physical field is assumed real-valued).  Output field
%     will have dimensions (1:2*kmax+1,1:kmax+1,:,:), where kmax = 2^n-1,
%     since physical resolution is 2^(n+1) x 2^(n+1).
%     NOTE: the bottom row of the output field corresponds to ky = 0,
%     the kx<0 part is conjugate-symmetric with the kx>0 part.
%     NOTE:  spec2grid(grid2spec(f)) = f. OPTIONAL:  da = true
%     assumes that input field was padded with zeros before being
%     transformed to grid space.  Returns unpadded spectral
%     output.  Default is da = false.
%
%
%     See also SPEC2GRID and FFT2.

if (nargin>1)
  dealias=da;
else
  dealias=false;
end

% Get number of nonsingleton dimensions
nd = ndims(gfield);  
if (nd==2)
    [n1,n2]=size(gfield);
    if (n1==1|n2==1), nd = 1; end
end

switch nd
  case 1
    if (size(gfield,1)~=1) % make it a column vector 
      sfield=transpose(gfield); 
    end
    if (dealias==false)
      kmax = length(gfield)/2 - 1;
      fk = fftshift(fft(gfield)/length(gfield));
      sfield = fk(kmax+2:2*kmax+2);  
    elseif (dealias==true)
      kmaxbig = length(gfield)/2 - 1;
      kmax = 2*(kmaxbig+1)/3 - 1;
      fk = fftshift(fft(gfield)/length(gfield));
      sfield = fk(kmaxbig+2:kmaxbig+1+kmax+1);  
    end
  case 2
    if (dealias==false)
      kmax = size(gfield,1)/2 - 1;
      fk = fft2(gfield)/prod(size(gfield));
      fk = fftshift(fk);
      sfield = fk(2:end,kmax+2:end);      
      sfield(1:kmax,1) = 0;
    elseif (dealias==true)
      kmaxbig = size(gfield,1)/2 - 1;
      kmax = 2*(kmaxbig+1)/3 - 1;
      fk = fft2(gfield)/prod(size(gfield));
      fk = fftshift(fk);
      offset = (kmax+1)/2+1;
      sfield = fk(offset+1:offset+2*kmax+1,kmaxbig+2:kmaxbig+2+kmax);
      sfield(1:kmax,1) = 0;
    end
  case 3
    if (dealias==false)
      kmax = size(gfield,1)/2 - 1;
      for n=1:size(gfield,3)
     	fk = fft2(gfield(:,:,n))/size(gfield,1)^2;
    	fk = fftshift(fk);
	    sfield(:,:,n) = fk(2:end,kmax+2:end);
      end
      sfield(1:kmax,1,:)=0;
    elseif (dealias==true)
      kmaxbig = size(gfield,1)/2 - 1;
      kmax = 2*(kmaxbig+1)/3 - 1;
      for n=1:size(gfield,3)
	    fk = fft2(gfield(:,:,n))/size(gfield,1)^2;
	    fk = fftshift(fk);
	    offset = (kmax+1)/2+1;
	    sfield(:,:,n) = fk(offset+1:offset+2*kmax+1,kmaxbig+2:kmaxbig+2+kmax);
      end
      sfield(1:kmax,1,:) = 0;      
    end
  case 4
    if (dealias==false)
      kmax = size(gfield,1)/2 - 1;
      for z=1:size(gfield,3)
	for t=1:size(gfield,4)
	  fk = fft2(gfield(:,:,z,t))/size(gfield,1)^2;
	  fk = fftshift(fk);
	  sfield(:,:,z,t) = fk(2:end,kmax+2:end);
	end
      end
      sfield(1:kmax,1,:,:)=0;
    elseif (dealias==true)
      kmaxbig = size(gfield,1)/2 - 1;
      kmax = 2*(kmaxbig+1)/3 - 1;
      for n=1:size(gfield,3)
	for t=1:size(gfield,4)
	  fk = fft2(gfield(:,:,n,t))/size(gfield,1)^2;
	  fk = fftshift(fk);
	  offset = (kmax+1)/2+1;
	  sfield(:,:,n,t) = ...
	      fk(offset+1:offset+2*kmax+1,kmaxbig+2:kmaxbig+2+kmax);
	end
      end
      sfield(1:kmax,1,:) = 0;      
    end
end


