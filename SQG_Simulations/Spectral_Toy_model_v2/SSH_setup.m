% this code defines some SSH initial field for benchmark. 

case_num = 4;

switch case_num
    case 1
        %% Case 1: sum

        phi0_s = cos(X) + cos(Y);
        
    case 2
        %% Case 2: Product
        phi0_s = sin(X) .* cos(Y);

    case 3
        %% Case 3: Submesoscale perturbation
        % Base large-scale flow (
        phi0_meso = sin(X) .* cos(Y);
        
        % Small-scale perturbations, high wavenumber components
        phi0_submeso = 0.1 * (sin(10*X) .* cos(12*Y) + cos(8*X) .* sin(15*Y));
        
        phi0_s = phi0_meso + phi0_submeso;
        
    case 4
        %% Case 4: Random field
        % In sqg velocity spectrum is k^(-5/3), so ssh spectrum is
        % k^(-11/3)
        spectral_slope = -11/3; 
        
        % White noise in physical space
        white_noise = randn(Nx, Ny);
        white_noise_hat = fft2(white_noise);
        
        % Create amplitude based on K from initialize.m
        amplitude = zeros(Nx, Ny);
        amplitude(K > 0) = K(K > 0).^(spectral_slope / 2);
        
        % Apply the amplitude
        phi0_s_hat_random = white_noise_hat .* amplitude;
        
        % Transfer to initial space
        phi0_s = real(ifft2(phi0_s_hat_random));
        
        % Normalize. 
        phi0_s = phi0_s / max(abs(phi0_s(:)));
        
    case 5
        %% Case 5: Gaussian
        x0 = pi; % Center of domain
        y0 = pi;
        R_eddy = 0.5; % Eddy radius
        
        phi0_s = exp(-((X - x0).^2 + (Y - y0).^2) / (2 * R_eddy^2));

    case 6
        %% Gaussian with white noise
        x0 = pi; % Center of domain
        y0 = pi;
        R_eddy = 0.5; % Eddy radius
        
        phi0_s = exp(-((X - x0).^2 + (Y - y0).^2) / (2 * R_eddy^2));

        white_noise = 0.01 * randn(Nx, Ny);
        phi0_s = phi0_s + white_noise;

    case 7
        %% Case 7 : Different scales of modes
        
        % Mode 1: large scale
        mode1 = 1.0 * cos(1*X + 1*Y);
        
        % Mode 2: medium scale
        mode2 = 0.5 * sin(3*X - 2*Y);
        
        % Mode 3: smaller scale
        mode3 = 0.2 * cos(7*X + 5*Y);
        
        % Mode 4: mesosclale with small perturbation. 
        mode4 = 0.05 * sin(12*X) .* cos(10*Y);
        
        % Sum
        phi0_s = mode1 + mode2 + mode3 + mode4;
        
        % Normalize for consistency
        phi0_s = phi0_s / max(abs(phi0_s(:)));
end


% Compute the fourier transform of the selected field
phi0_s_hat = fft2(phi0_s);
