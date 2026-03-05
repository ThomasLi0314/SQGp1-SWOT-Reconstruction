%% This is the main file
clear;
clc;
close all;

% Some important parameters can be changed here
% grid size
Nx = 64;
Ny = 64;

% Domain size
Lx = 2 * pi;
Ly = 2 * pi;

Ro = 0.01; %Rossby number same as epsilon

initialize;

SSH_setup;
%% Forward Part
% This part derives the true fourier SSH 

cyclogeo_term_true = cyclogeo_term(phi0_s_hat, kx, ky);
vorticity_term_true = vorticity_term(phi0_s_hat, mu, inv_mu, kx, ky, K2, Bu);

% True pressure field 
p1_s_hat_true = -(f * vorticity_term_true + cyclogeo_term_true) .* inv_K2;

eta_s_hat_true = f * phi0_s_hat + p1_s_hat_true * epsilon; 

fprintf('True SSH data generated\n');

%% Inversion part

% Initial guess very close to the true value
max_phi0_s = max(phi0_s(:));

%%%%%%%%%%%%%%%%%%% Initial Guess %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phi0_s_guess = phi0_s + 0.001 * max_phi0_s * randn(Nx, Ny);



phi0_s_hat_guess = fft2(phi0_s_guess);

% ===================== Optimization Settings =====================
% Change these variables to switch method and parallel status.
%   opt_method:   'lsq-trf'  | 'lsq-lm'  | 'fminunc'
%   use_parallel: true / false
opt_method   = 'fminunc';
use_parallel = false;
num_iteration = 300;
% =================================================================

% Build options and select solver
switch opt_method
    case 'lsq-trf'
        alg_name = 'trust-region-reflective';
        options = optimoptions('lsqnonlin', ...
            'Display', 'iter', ...
            'Algorithm', alg_name, ...
            'MaxIterations', num_iteration, ...
            'UseParallel', use_parallel);
    case 'lsq-lm'
        alg_name = 'levenberg-marquardt';
        options = optimoptions('lsqnonlin', ...
            'Display', 'iter', ...
            'Algorithm', alg_name, ...
            'MaxIterations', num_iteration, ...
            'UseParallel', use_parallel);
    case 'fminunc'
        alg_name = 'quasi-newton';
        options = optimoptions('fminunc', ...
            'Display', 'iter', ...
            'Algorithm', alg_name, ...
            'MaxIterations', num_iteration, ...
            'UseParallel', use_parallel);
    otherwise
        error('Unknown opt_method: %s. Use ''lsq-trf'', ''lsq-lm'', or ''fminunc''.', opt_method);
end

%% Build method label for plot title
if use_parallel
    par_str = 'parallel';
else
    par_str = 'serial';
end
method_label = sprintf('%s (%s, %s, %dx%d)', opt_method, alg_name, par_str, Nx, Ny);

% Cost function handles
cost_func_lsq = @(phi0_s_hat_guess) cost_function_lsq(phi0_s_hat_guess, f, kx, ky, mu, inv_mu, Bu, epsilon, K2, inv_K2, eta_s_hat_true);
cost_func_fmin = @(phi0_s_guess) cost_function_fmin(phi0_s_guess, f, kx, ky, mu, inv_mu, Bu, epsilon, K2, inv_K2, eta_s_hat_true);

%% Run the optimization
fprintf('Running optimization: %s\n', method_label);
tic;
try
    switch opt_method
        case {'lsq-trf', 'lsq-lm'}
            [phi0_s_hat_opt, resnorm] = lsqnonlin(cost_func_lsq, phi0_s_hat_guess, [], [], options);
        case 'fminunc'
            [phi0_s_opt, resnorm] = fminunc(cost_func_fmin, phi0_s_guess, options);
            % fminunc optimizes in physical space, convert back to spectral
            phi0_s_hat_opt = fft2(phi0_s_opt);
    end
    disp('Optimization Complete.');
catch ME
    disp('Optimization failed or interrupted.')
    disp(ME.message);
end
elapsed = toc;
fprintf('Elapsed time: %.2f s\n', elapsed);

%% Forward Second round to obtain the velocity field.

% Calculate the optimized surface u, v
[u_surface_opt, v_surface_opt] = calculate_surface_u(phi0_s_hat_opt, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu);

% Calculate the true surface u, v
[u_surface_true, v_surface_true] = calculate_surface_u(phi0_s_hat, mu, inv_mu, kx, ky, K2, inv_K2, epsilon, Bu);

% Compute surface vorticity:  zeta = dv/dx - du/dy  (in spectral space)
u_opt_hat = fft2(u_surface_opt);  v_opt_hat = fft2(v_surface_opt);
u_true_hat = fft2(u_surface_true); v_true_hat = fft2(v_surface_true);

zeta_opt  = real(ifft2( (1i).*kx.*v_opt_hat  - (1i).*ky.*u_opt_hat  ));
zeta_true = real(ifft2( (1i).*kx.*v_true_hat - (1i).*ky.*u_true_hat ));

%% Plotting: 2 rows (u, zeta) x 3 columns (optimized, true, difference)
figure('Position', [50, 50, 1400, 700]);

% ---- Row 1: Surface Zonal Velocity u ----
subplot(2, 3, 1);
imagesc(x, y, u_surface_opt');
set(gca, 'YDir', 'normal'); colorbar;
title('Optimized u_{surface}');
xlabel('x'); ylabel('y'); axis equal tight;

subplot(2, 3, 2);
imagesc(x, y, u_surface_true');
set(gca, 'YDir', 'normal'); colorbar;
title('True u_{surface}');
xlabel('x'); ylabel('y'); axis equal tight;

subplot(2, 3, 3);
imagesc(x, y, (u_surface_opt - u_surface_true)');
set(gca, 'YDir', 'normal'); colorbar;
title('u Difference');
xlabel('x'); ylabel('y'); axis equal tight;

% ---- Row 2: Surface Vorticity zeta ----
subplot(2, 3, 4);
imagesc(x, y, zeta_opt');
set(gca, 'YDir', 'normal'); colorbar;
title('Optimized \zeta_{surface}');
xlabel('x'); ylabel('y'); axis equal tight;

subplot(2, 3, 5);
imagesc(x, y, zeta_true');
set(gca, 'YDir', 'normal'); colorbar;
title('True \zeta_{surface}');
xlabel('x'); ylabel('y'); axis equal tight;

subplot(2, 3, 6);
imagesc(x, y, (zeta_opt - zeta_true)');
set(gca, 'YDir', 'normal'); colorbar;
title('\zeta Difference');
xlabel('x'); ylabel('y'); axis equal tight;

colormap jet;
sgtitle(sprintf('Surface Fields — %s  (%.1f s)', method_label, elapsed));
saveas(gcf, 'surface_comparison.png');
