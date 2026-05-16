function cost = cost_function_fmin(phi0_s_guess, f, kx, ky, mu, inv_mu, Bu, epsilon, K2, inv_K2, eta_s_hat_true)

    % this function compute the cost function for the optimization

    % the input is the physica phi0_s_guess a fourier transform is performed here. 
    phi0_s_hat_guess = fft2(phi0_s_guess);

    % cyclogeo term
    cyclogeo_term_guess = cyclogeo_term(phi0_s_hat_guess, kx, ky);
    vorticity_term_guess = vorticity_term(phi0_s_hat_guess, mu, inv_mu, kx, ky, K2, Bu);

    % p1 guess field
    % The equation is \nabla^2 p^1 = f \zeta^1 + 2J
    % So in Fourier space: p1_s_hat = -(f * zeta_s_hat + 2J_s_hat) / K^2
    p1_s_hat_guess = -(f * vorticity_term_guess + cyclogeo_term_guess) .* inv_K2;

    % SSH guess field
    eta_s_hat_guess = f * phi0_s_hat_guess + p1_s_hat_guess * epsilon; 

    % %% Filter? Make the small scale more important. 
    % diff = abs(eta_s_hat_guess - eta_s_hat_true);
    % cost = sum(diff(:).^2);  

    %% FIlter! Yes.
    % Compute difference in spectral space
    diff_hat = abs(eta_s_hat_guess - eta_s_hat_true);
    
    % Spectral error weighting (err_spec = 1 + K^2),
    err_spec = 1 + K2; % can be something else.
    
    % Apply whitening / weighting to the cost function
    cost = sum( (diff_hat(:).^2) .* err_spec(:) );

    % If compute cost in real space?
    % eta_s_guess = real(ifft2(eta_s_hat_guess));
    % eta_s_true = real(ifft2(eta_s_hat_true));
    % diff = abs(eta_s_guess - eta_s_true);
    % cost = sum(diff(:).^2);  
end