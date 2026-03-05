% This function calculates the Cyclogeostrophic correction term. Which is J(Phi_x, Phi_y)

% Everythin is at the surface and 

function J_Phi_s_hat = cyclogeo_term(phi0_s_hat, kx, ky)
    phi0_s_xx = ifft2(phi0_s_hat .* (-1) .* kx .^ 2);  
    phi0_s_yy = ifft2(phi0_s_hat .* (-1) .* ky .^ 2);
    phi0_s_xy = ifft2(phi0_s_hat .* (-1) .* ky .* kx);

    % Compute the jacobian
    J_Phi_s = phi0_s_xx .* phi0_s_yy - phi0_s_xy .^ 2;

    % Return to spectral space
    J_Phi_s_hat = 2 * fft2(J_Phi_s);
end