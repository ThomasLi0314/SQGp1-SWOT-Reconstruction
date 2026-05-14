function [fg] = k2g(fk)
    
    [nx,nky] = size(fk);
    n = nx/2;
    
    % First ensure ky = 0 row is conjugate-symmetric
    fk(n+2:end,1,:) = conj(fk(n:-1:2,1,:));     
    
    % Make bigger field, populate lower half plane using conj sym.
    fkb = zeros(2*n,2*n,size(fk,3));
    fkb(1:n,1:n,:) = fk(1:n,:,:);               % [0..K,:]  
    fkb(1*n+2:end,1:n,:) = fk(n+2:end,:,:);     % [-K..-1,:] 
    fkb(2:end,n+2:end,:) = conj(fkb(end:-1:2,n:-1:2,:)); 
    fkb(1,n+2:end,:) = conj(fkb(1,n:-1:2,:));
    
    fg = nx^2*real(ifft(ifft(fkb,[],1),[],2));

end  % end of entire function


