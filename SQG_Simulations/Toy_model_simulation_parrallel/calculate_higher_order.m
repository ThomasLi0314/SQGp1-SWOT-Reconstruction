
function [F1, G1, Phi1] = calculate_higher_order(phi0_3d, K, kx, ky, z, Bu, N, nz)
    phi0_surf = phi0_3d(:,:,end);
    phi0_surf_hat = fft2(phi0_surf);
    mu = sqrt(Bu) * K;
    
    phi0_z = zeros(N, N, nz);
    
    % Vectorized calculation for phi0_z
    decay = exp(mu .* reshape(z, 1, 1, nz));
    f_z_hat = phi0_surf_hat .* decay .* mu;
    phi0_z = real(ifft2(f_z_hat));
    
    % Horizontal derivatives
    phi0_x = diff_spectral(phi0_3d, kx, 1);
    phi0_y = diff_spectral(phi0_3d, ky, 2);
    
    phi0_zx = diff_spectral(phi0_z, kx, 1);
    phi0_zy = diff_spectral(phi0_z, ky, 2);
    
    % Jacobians
    phi0_xx = diff_spectral(phi0_x, kx, 1);
    phi0_xy = diff_spectral(phi0_x, ky, 2);
    phi0_yy = diff_spectral(phi0_y, ky, 2);
    
    % J(Phi0_z, Phi0_x)
    jac_F = phi0_zx .* phi0_xy - phi0_zy .* phi0_xx;
    rhs_F = (2.0 / Bu) * jac_F;
    
    % J(Phi0_z, Phi0_y)
    jac_G = phi0_zx .* phi0_yy - phi0_zy .* phi0_xy;
    rhs_G = (2.0 / Bu) * jac_G;
    
    % Solve Poisson
    F1 = solve_poisson_z(rhs_F, K, z, Bu);
    G1 = solve_poisson_z(rhs_G, K, z, Bu);
    
    % Phi1
    Phi1 = (1.0 / (2 * Bu)) * (phi0_z.^2);
end