function psi = solve_poisson_z(rhs, K, z, Bu)
    % Solves (-K^2 + 1/Bu * d^2/dz^2) psi = rhs
    % Boundary Conditions:
    % z -> -inf : Radiation (psi_z = mu * psi) at Index 1
    % z = top   : Dirichlet (psi = 0) at Index end
    
    [N, ~, nz] = size(rhs);
    rhs_hat = fft2(rhs);
    psi_hat = zeros(size(rhs_hat));
    
    dz = z(2) - z(1);
    dz2 = dz^2;

    % We solve for indices 1 to nz-1. 
    % Index 'nz' is the top boundary fixed at 0.
    n_int = nz - 1; 
    
    K2 = K.^2;
    K2_flat = K2(:);
    M = N*N;

    % Decay rate mu for vanishing at -infinity
    % Solution behaves like e^(mu*z), so d(psi)/dz = mu*psi
    mu = sqrt(Bu) * sqrt(K2_flat);

    rhs_flat_all = reshape(rhs_hat, M, nz);

    % We solve for the first n_int points (1 to nz-1)
    rhs_inner = rhs_flat_all(:, 1:n_int);
    
    d = Bu * dz2 * rhs_inner;

    % Base main diagonal value: -2 - Bu*dz^2*K^2
    main_val = -2.0 - Bu * dz2 * K2_flat;
    
    c_prime = zeros(M, n_int-1);
    d_prime = zeros(M, n_int);
    
    % --- FORWARD ELIMINATION (Thomas Algorithm) ---
    
    % 1. Bottom Boundary (Index 1) - Radiation Condition
    % Ghost point logic: (psi_2 - psi_0) / 2dz = mu * psi_1
    % => psi_0 = psi_2 - 2*dz*mu*psi_1
    % Stencil: psi_0 - 2*psi_1 + psi_2 = ...
    % Becomes: (psi_2 - 2*dz*mu*psi_1) - 2*psi_1 + psi_2 = ...
    %          psi_1 * (-2 - 2*dz*mu) + 2*psi_2
    
    % Modified main diagonal for bottom point:
    main_val_bottom = main_val - 2.0 * dz * mu;
    
    % Upper diagonal is 2.0 due to ghost point substitution
    upper_bottom = 2.0;
    
    % First step of TDMA
    c_prime(:,1) = upper_bottom ./ main_val_bottom;
    d_prime(:,1) = d(:,1) ./ main_val_bottom;
    
    % 2. Interior Loop (Indices 2 to n_int-1)
    lower = 1;
    upper = 1;
    
    for i = 2:n_int-1
        temp = main_val - lower .* c_prime(:,i-1);
        c_prime(:,i) = upper ./ temp;
        d_prime(:,i) = (d(:,i) - lower .* d_prime(:,i-1)) ./ temp;
    end
    
    % 3. Top Boundary Point (Index n_int)
    % This is the point just below the rigid lid (psi_nz = 0).
    % Stencil: psi_{nz-1} - 2*psi_{nz} + psi_{nz+1} (where psi_{nz+1} is 0)
    % No modification to diagonal needed, just standard Dirichlet end.
    
    i = n_int;
    temp = main_val - lower .* c_prime(:,i-1);
    % We don't need c_prime for the last point
    d_prime(:,i) = (d(:,i) - lower .* d_prime(:,i-1)) ./ temp;
    
    % --- BACK SUBSTITUTION ---
    x = zeros(M, n_int);
    x(:, n_int) = d_prime(:, n_int);
    
    for i = n_int-1:-1:1
        x(:,i) = d_prime(:,i) - c_prime(:,i) .* x(:,i+1);
    end
    
    % --- RECONSTRUCTION ---
    psi_hat_inner = reshape(x, N, N, n_int);
    
    % Fill indices 1 to n_int (leaving last index as 0)
    psi_hat(:,:,1:n_int) = psi_hat_inner;
    psi_hat(:,:,end) = 0; % Explicitly zero at top
    
    psi = real(ifft2(psi_hat));
end