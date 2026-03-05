function cost = cost_function_lsq(phi0_s_hat_guess, f, kx, ky, mu, inv_mu, Bu, epsilon, K2, inv_K2, eta_s_hat_true)

    % this function compute the cost function for the optimization

    % cyclogeo term
    cyclogeo_term_guess = cyclogeo_term(phi0_s_hat_guess, kx, ky);
    vorticity_term_guess = vorticity_term(phi0_s_hat_guess, mu, inv_mu, kx, ky, K2, Bu);

    % p1 guess field
    % The equation is \nabla^2 p^1 = f \zeta^1 + 2J
    % So in Fourier space: p1_s_hat = -(f * zeta_s_hat + 2J_s_hat) / K^2
    p1_s_hat_guess = -(f * vorticity_term_guess + cyclogeo_term_guess) .* inv_K2;

    % SSH guess field
    eta_s_hat_guess = f * phi0_s_hat_guess + p1_s_hat_guess * epsilon; 

    % cost function 
    cost = abs(eta_s_hat_guess - eta_s_hat_true);
end