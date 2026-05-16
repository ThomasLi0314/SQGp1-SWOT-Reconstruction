function cost = sqg_cost_function(phi0_flat, f, ssh_ture, K, kx, ky, z, Bu, Ro, N, nz, dx, dz)
    % 3D phi0
    phi0_surf = reshape(phi0_flat, N, N);
    phi0_3d_flat = derive_phi0_3d(phi0_surf, K, z, Bu);

    % Calculate Higher Order Terms
    [F1_guess, G1_guess, Phi1_guess] = calculate_higher_order(phi0_3d_flat, K, kx, ky, z, Bu, N, nz);

    % Compute the guess p1
    p1_guess = solve_p1(f, dx, dz, kx, ky, z, Bu, Ro, phi0_3d_flat, F1_guess, G1_guess, Phi1_guess);

    % Compute the guess SSH
    p1_guess_surf = p1_guess(:, :, end);
    ssh_guess = phi0_flat + Ro * p1_guess_surf;

    % Cost
    difference = ssh_guess - ssh_ture;
    cost = difference(:);
end