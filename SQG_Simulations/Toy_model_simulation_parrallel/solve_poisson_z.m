
function psi = solve_poisson_z(rhs, K, z, Bu)
    % Solves (-K^2 + 1/Bu * d^2/dz^2) psi = rhs
    [N, ~, nz] = size(rhs);
    rhs_hat = fft2(rhs);
    psi_hat = zeros(size(rhs_hat));
    
    dz = z(2) - z(1);
    dz2 = dz^2;
    n_int = nz - 2;
    
    K2 = K.^2;
    K2_flat = K2(:);
    M = N*N;
    
    rhs_flat_all = reshape(rhs_hat, M, nz);
    rhs_inner = rhs_flat_all(:, 2:end-1);
    
    d = Bu * dz2 * rhs_inner;
    
    % Vectorized Thomas Algorithm Coeffs
    % main: -2 - Bu*dz^2*K^2
    main_val = -2.0 - Bu * dz2 * K2_flat;
    
    % We need to loop carefully or broadcast.
    % To match Python logic:
    
    c_prime = zeros(M, n_int-1);
    d_prime = zeros(M, n_int);
    
    lower = 1;
    upper = 1;
    
    % i=1
    c_prime(:,1) = upper ./ main_val;
    d_prime(:,1) = d(:,1) ./ main_val;
    
    for i = 2:n_int-1
        temp = main_val - lower .* c_prime(:,i-1);
        c_prime(:,i) = upper ./ temp;
        d_prime(:,i) = (d(:,i) - lower .* d_prime(:,i-1)) ./ temp;
    end
    
    i = n_int;
    temp = main_val - lower .* c_prime(:,i-1);
    d_prime(:,i) = (d(:,i) - lower .* d_prime(:,i-1)) ./ temp;
    
    x = zeros(M, n_int);
    x(:, n_int) = d_prime(:, n_int);
    
    for i = n_int-1:-1:1
        x(:,i) = d_prime(:,i) - c_prime(:,i) .* x(:,i+1);
    end
    
    psi_hat_inner = reshape(x, N, N, n_int);
    psi_hat(:,:,2:end-1) = psi_hat_inner;
    
    psi = real(ifft2(psi_hat));
end
