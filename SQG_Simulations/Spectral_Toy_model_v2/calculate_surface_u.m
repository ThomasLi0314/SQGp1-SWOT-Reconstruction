% This function returns the physical value of surface zonal velocity u.


function [u_surface, v_surface] = calculate_surface_u(phi0_s_hat, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu)

    %% u = -\Phi_y - F_1

    % Part 1: \Phi^1 = \Phi_z^2 + \PHi_z \Phi_zz

    phi0_s_z = real(ifft2(phi0_s_hat .* mu));
    Phi_1_term1 = fft2(1 / 2 * phi0_s_z .^2);
    phi0_s_zz = real(ifft2(phi0_s_hat .* mu .* mu));
    Phi_1_term2 = -fft2(phi0_s_z .* phi0_s_zz) .* inv_mu;

    % y_derivative for u_surface
    Phi1_s_hat_y = (Phi_1_term1 + Phi_1_term2) .* (1i) .* ky;

    % x_derivative for v_surface
    Phi1_s_hat_x = (Phi_1_term1 + Phi_1_term2) .* (1i) .* kx;


    % Part 2: F^1 = \Phi_y \Phi_z - \Phi_y^s \Phi_z^0
    phi0_s_y = real(ifft2(phi0_s_hat .* (1i) .* ky));
    phi0_s_zz = real(ifft2(phi0_s_hat .* mu .* mu));
    phi0_s_yz = real(ifft2(phi0_s_hat .* (1i) .* ky .* mu));
    phi0_s_z = real(ifft2(phi0_s_hat .* mu));

    F1_term1 = fft2(phi0_s_y .* phi0_s_zz + phi0_s_yz .* phi0_s_z);

    F1_term2 = -fft2(phi0_s_y .* phi0_s_z) .* mu;

    F1_s_hat_z = F1_term1 + F1_term2;

    % Part 3: G^1 = \Phi_x \Phi_z - \PHi_x^s \Phi_z^s
    phi0_s_x = real(ifft2(phi0_s_hat .* (1i) .* kx));
    phi0_s_xz = real(ifft2(phi0_s_hat .* (1i) .* kx .* mu));

    G1_term1 = fft2(phi0_s_x .* phi0_s_zz + phi0_s_xz .* phi0_s_z);

    G1_term2 = -fft2(phi0_s_x .* phi0_s_z) .* mu;

    G1_s_hat_z = G1_term1 + G1_term2;


    %% Sum up terms
    phi0_s_hat_y = phi0_s_hat .* (1i) .* ky;
    phi0_s_hat_x = phi0_s_hat .* (1i) .* kx;

    u_s_hat = -phi0_s_hat_y - epsilon .* (Phi1_s_hat_y + F1_s_hat_z) ./ Bu;
    v_s_hat = phi0_s_hat_x + epsilon .* (Phi1_s_hat_x - G1_s_hat_z) ./ Bu;

    u_surface = real(ifft2(u_s_hat));
    v_surface = real(ifft2(v_s_hat));
end