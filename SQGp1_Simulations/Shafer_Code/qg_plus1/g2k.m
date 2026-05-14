function [fk] = g2k(fg)

    [nx,ny] = size(fg);    
    n = nx/2;
    normfac = 1/nx^2;
    fbk = normfac*fft(fft(fg,[],1),[],2);
    fk = fbk(:,1:n,:);
    fk(n+1:end,1,:) = 0;  % this part given by ConjSym
    fk(1,1,:) = 0;
    
end
