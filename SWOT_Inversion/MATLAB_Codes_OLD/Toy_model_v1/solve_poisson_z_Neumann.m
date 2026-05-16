function psi = solve_poisson_z(rhs, K, z, Bu)
    % Solves (-K^2 + 1/Bu * d^2/dz^2) psi = rhs
    % Boundary Conditions:
    % z -> -inf : Radiation (psi_z = mu * psi) at Index 1
    % z = 0     : Neumann (psi_z = 0) at Index end
    
    [N, ~, nz] = size(rhs);
    rhs_hat = fft2(rhs);
    psi_hat = zeros(size(rhs_hat));
    
    dz = z(2) - z(1);
    dz2 = dz^2;

    % We solve for all indices 1 to nz. 
    % Index 'nz' is the top boundary with psi_z = 0.
    n_int = nz; 
    
    K2 = K.^2;
    K2_flat = K2(:);
    M = N*N;

    % Decay rate mu for vanishing at -infinity
    % Solution behaves like e^(mu*z), so d(psi)/dz = mu*psi
    mu = sqrt(Bu) * sqrt(K2_flat);

    rhs_flat_all = reshape(rhs_hat, M, nz);

    % We solve for all n_int points (1 to nz)
    rhs_inner = rhs_flat_all(:, 1:n_int);
    
    d = Bu * dz2 * rhs_inner;

    % Base main diagonal value: -2 - Bu*dz^2*K^2
    main_val = -2.0 - Bu * dz2 * K2_flat;
    
    c_prime = zeros(M, n_int-1);
    d_prime = zeros(M, n_int);
    
    % Identify K=0 mode to handle the null space 
    % (with Neumann top and radiation bottom, K=0 mode becomes purely Neumann, giving singular matrix)
    k0_idx = (K2_flat == 0);
    
    % --- FORWARD ELIMINATION (Thomas Algorithm) ---
    
    % 1. Bottom Boundary (Index 1) - Radiation Condition
    % Ghost point logic: (psi_2 - psi_0) / 2dz = mu * psi_1
    % => psi_0 = psi_2 - 2*dz*mu*psi_1
    % Stencil: psi_0 - 2*psi_1 + psi_2 = ...
    % Becomes: (psi_2 - 2*dz*mu*psi_1) - 2*psi_1 + psi_2 = ...
    %          psi_1 * (-2 - 2*dz*mu) + 2*psi_2
    
    % Modified main diagonal for bottom point:
    main_val_bottom = main_val - 2.0 * dz * mu;
    
    % Prevent division by zero for K=0 mode by setting diagonal to 1
    temp_bot = main_val_bottom;
    temp_bot(k0_idx) = 1;

    % Upper diagonal is 2.0 due to ghost point substitution
    upper_bottom = 2.0;

    % First step of TDMA
    c_prime(:,1) = upper_bottom ./ temp_bot;
    c_prime(k0_idx,1) = 0; % explicit zero for k=0 mode
    
    d_prime(:,1) = d(:,1) ./ temp_bot;
    d_prime(k0_idx,1) = 0; % explicit zero for k=0 mode
    
    % 2. Interior Loop (Indices 2 to n_int-1)
    lower = 1.0;
    upper = 1.0;
    
    for i = 2:n_int-1
        temp = main_val - lower .* c_prime(:,i-1);
        temp(k0_idx) = 1;
        
        c_prime(:,i) = upper ./ temp;
        c_prime(k0_idx,i) = 0;
        
        d_prime(:,i) = (d(:,i) - lower .* d_prime(:,i-1)) ./ temp;
        d_prime(k0_idx,i) = 0;
    end
    
    % 3. Top Boundary Point (Index n_int) - Neumann Condition (psi_z = 0)
    % Ghost point logic: (psi_{nz+1} - psi_{nz-1}) / 2dz = 0 => psi_{nz+1} = psi_{nz-1}
    % Stencil: psi_{nz-1} - 2*psi_{nz} + psi_{nz+1}
    % Becomes: 2*psi_{nz-1} - 2*psi_{nz} (without K2 term, with K2 it uses main_val)
    % So lower diagonal is 2.0
    
    i = n_int;
    lower_top = 2.0;
    temp = main_val - lower_top .* c_prime(:,i-1);
    temp(k0_idx) = 1;
    
    % We don't need c_prime for the last point
    d_prime(:,i) = (d(:,i) - lower_top .* d_prime(:,i-1)) ./ temp;
    d_prime(k0_idx,i) = 0;
    
    % --- BACK SUBSTITUTION ---
    x = zeros(M, n_int);
    x(:, n_int) = d_prime(:, n_int);
    
    for i = n_int-1:-1:1
        x(:,i) = d_prime(:,i) - c_prime(:,i) .* x(:,i+1);
    end
    
    % --- RECONSTRUCTION ---
    psi_hat_inner = reshape(x, N, N, n_int);
    
    % Fill indices 1 to n_int
    psi_hat(:,:,1:n_int) = psi_hat_inner;
    
    psi = real(ifft2(psi_hat));
end