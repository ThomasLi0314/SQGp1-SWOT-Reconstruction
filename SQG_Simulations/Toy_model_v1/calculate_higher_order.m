
function [F1, G1, Phi1] = calculate_higher_order(phi0_3d, K, kx, ky, z, Bu, N, nz)
    % 1. Prepare Spectral Base
    phi0_surf = phi0_3d(:,:,end);
    phi0_surf_hat = fft2(phi0_surf);
    mu = sqrt(Bu) * K;
    
    % 2. Vertical Structure in Spectral Space
    % Shape: (1, 1, nz) for broadcasting
    decay = exp(mu .* reshape(z, 1, 1, nz)); 
    
    % Base 3D spectrum: Phi_hat(x, y, z)
    % Note: This assumes phi0_3d is fully defined by the surface mode.
    % If phi0_3d contains other modes, keep your original line for phi0_x/y.
    phi0_hat_3d = phi0_surf_hat .* decay; 
    
    % Spectrum of Vertical Derivative (d/dz)
    phi0_z_hat = phi0_hat_3d .* mu;

    % spectrum of second order vertical derivative
    phi0_zz_hat = phi0_z_hat .* mu;
    
    % 3. Calculate Derivatives in Spectral Space
    % Note: kx and ky should be N x N meshgrids
    
    % Derivatives of Phi (Horizontal)
    phi0_x_hat = phi0_hat_3d .* (1i * kx);
    phi0_y_hat = phi0_hat_3d .* (1i * ky);
    
    % Derivatives of Phi_z (Horizontal)
    phi0_zx_hat = phi0_z_hat .* (1i * kx);
    phi0_zy_hat = phi0_z_hat .* (1i * ky);
    
    % Second Derivatives (needed for Jacobian)
    phi0_xx_hat = phi0_x_hat .* (1i * kx);
    phi0_xy_hat = phi0_x_hat .* (1i * ky); % d/dy(d/dx)
    phi0_yy_hat = phi0_y_hat .* (1i * ky);
    
    % 4. Transform to Physical Space (Batch IFFT)
    phi0_z = real(ifft2(phi0_z_hat));
    phi0_zz = real(ifft2(phi0_zz_hat));
    phi0_zx = real(ifft2(phi0_zx_hat));
    phi0_zy = real(ifft2(phi0_zy_hat));
    
    phi0_xx = real(ifft2(phi0_xx_hat));
    phi0_xy = real(ifft2(phi0_xy_hat));
    phi0_yy = real(ifft2(phi0_yy_hat));

    nabla_phi0_z_sq = phi0_zx .^2 + phi0_zy .^2;
    
    % 5. Compute Jacobians (Non-linear products must happen in physical space)
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
    rhs_Phi = (1.0 / Bu) * (nabla_phi0_z_sq + (1.0 / Bu) * phi0_zz .* phi0_zz);
    Phi1 = solve_poisson_z(rhs_Phi, K, z, Bu);

    %% For checking
    % N = 32; % Grid Size
    % L = 2 * pi; % Length
    % Bu = 1; % Burger's number
    % Ro = 0.1; % Rossby Number \epsilon
    % z_min = -20.0; % minimum deep
    % nz = 32; % vertical grid size
    % f = 1; % Corriori force parameter
    % 
    % % Grid Setup
    % dx = L / N;
    % x = (0:N-1) * dx;
    % y = (0:N-1) * dx;
    % z = linspace(z_min, 0, nz);
    % dz = abs(z_min) / nz;
    % [X, Y, Z] = ndgrid(x, y, z);
    % rhs_Phi_analytic = exp(2 .* Z) .* (2 + 2 .* cos(X) .* cos(Y));
end