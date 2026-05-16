function p1 = solve_p1(f, dx, dz, kx, ky, z, Bu, Ro, Phi0, F1, G1, Phi1)
    % The formula for p is \nabla^2 p - f\zeta^1 = 2J(\Phi_x^0, \Phi_y^0)
    ikx = 1i * kx;
    iky = 1i * ky;
    neg_K2 = -kx.^2 - ky.^2;

    % Compute the z-derivative of F1 and G1
    [~, ~, F1_z] = gradient(F1, dx, dx, dz);
    [~, ~, G1_z] = gradient(G1, dx, dx, dz);

    
    % Compute hats
    phi0_hat = fft2(Phi0);
    F1_z_hat = fft2(F1_z);
    G1_z_hat = fft2(G1_z);
    Phi1_hat = fft2(Phi1);
    
    % Compute zeta1_hat
    zeta1_hat = neg_K2 .* Phi1_hat - iky .* F1_z_hat + ikx .* G1_z_hat;

    %% Cyclonic term

    % Compute the second derivative
    phi0_xx = real(ifft2( (ikx.^2) .* phi0_hat ));
    phi0_yy = real(ifft2( (iky.^2) .* phi0_hat ));
    phi0_xy = real(ifft2( (ikx .* iky) .* phi0_hat ));

    % Compute the Jacobian
    Jaco = phi0_xx .* phi0_yy - phi0_xy .* phi0_xy;

    % Compute the right-hand size
    term2 = fft2(2 * Jaco);

    %% Inversion
    % Compute p1_hat

    p1_hat = (zeta1_hat + term2) ./ neg_K2;

    % Handel singulartiy
    p1_hat(1,1,:) = 0;

    % Compute p1
    p1 = real(ifft2(p1_hat));

end