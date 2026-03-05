function phi0_3d = derive_phi0_3d(phi0_surf, K, z, Bu)
    [N, ~] = size(phi0_surf);
    nz = length(z);
    phi0_surf_hat = fft2(phi0_surf);
    phi0_3d = zeros(N, N, nz);
    mu = sqrt(Bu) * K;
    
    % Vectorized calculation
    decay = exp(mu .* reshape(z, 1, 1, nz));
    phi0_3d = real(ifft2(phi0_surf_hat .* decay));
end