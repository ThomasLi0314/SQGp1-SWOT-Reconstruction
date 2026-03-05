clear;
clc;
close all;

%% Parameters
N = 64; % Grid Size
L = 2 * pi; % Length
Bu = 1; % Burger's number
Ro = 0.1; % Rossby Number \epsilon
z_min = -1.0; % minimum deep
nz = 32; % vertical grid size
f = 1; % Corriori force parameter

% Grid Setup
dx = L / N;
x = (0:N-1) * dx;
y = (0:N-1) * dx;
z = linspace(z_min, 0, nz);
dz = abs(z_min) / nz;
[X, Y] = ndgrid(x, y);

% Spectral Grid
dk = 2 * pi / L;
ks = [0: N / 2 - 1, -N / 2:-1] * dk;
[kx, ky] = ndgrid(ks, ks);
K2 = kx.^2 + ky.^2;
K = sqrt(K2);

% Remove mean (k=0) to avoid division by zero
K(1,1) = 1.0; % Arbitrary value, similar to Python script


%% Initial Conditions
rng(42); % Set seed for reproducibility
% We produce a random $\Phi^0$ here. 
k_peak = 4;
slope = -3; % This slope matches the energy cascade in 3D turbulance
phase = rand(N, N) * 2 * pi;
% amplitude = (K ./ k_peak).^(slope) .* exp(-(K./k_peak).^2);
amplitude = cos(X);

amplitude(1,1) = 0;
% Compute the hat
phi0_hat = amplitude .* exp(1i * phase);
% Inverse transform to get to the physical space
phi0_surf = real(ifft2(phi0_hat));
% Normalize
phi0_surf = phi0_surf / std(phi0_surf(:));

%Derive 3D Potential $Phi^0$
phi0_3d_true = derive_phi0_3d(phi0_surf, K, z, Bu);

%Calculate Higher Order Terms (F1, G1, Phi1)
[F1_true, G1_true, Phi1_true] = calculate_higher_order(phi0_3d_true, K, kx, ky, z, Bu, N, nz);

% Compute the true p1
p1_true = solve_p1(f, dx, dz, kx, ky, z, Bu, Ro, phi0_3d_true, F1_true, G1_true, Phi1_true);

% Compute the true SSH
% Note: p1_true is 3D, we need the surface value (at z=0, which is index 'end')
p1_surf = p1_true(:, :, end);
ssh_true = phi0_surf + Ro * p1_surf;

disp(size(p1_surf));
disp(size(phi0_surf));

disp('Truth SSH Generated.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Optimization-Based Inversion
disp('Starting Inversion.......');

% Objective: Find phi0_3d_inv that minimizes || SSH_obs - SSH_model(phi0_3d_inv) ||^2
% Start with an Initial Guess (Linear Inversion)
% SSH ~ Phi0 (approx).

% Initial guess for the surface potential phi0(x,y), which is the leading order of SSH.
% phi0_guess_flat = reshape(ssh_true, [], 1);
% Plot the maximum value of phi0_surface
max_phi0_surf = max(phi0_surf(:));
phi0_surf_guess = phi0_surf + 0.0001 * max_phi0_surf * randn(N, N);

% Optimization Options
% Control the number of iterations.
num_iteration = 20;

% Check and start parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool;
end

% options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton', ...
%     'MaxIterations', num_iteration, 'UseParallel', true);
options = optimoptions('lsqnonlin', 'Display', 'iter', 'Algorithm', 'trust-region-reflective', 'MaxIterations', num_iteration ...
    ,'UseParallel', true);

% Compute the cost function
cost_func = @(phi0_guess) sqg_cost_function(phi0_guess, f, ssh_true, K, kx, ky, z, Bu, Ro, N, nz, dx, dz);

% Run Optimization
tic;
try
    [phi0_opt_flat, resnorm] = lsqnonlin(cost_func, phi0_surf_guess, [], [], options);
    phi0_surf_opt = reshape(phi0_opt_flat, N, N);
    disp('Optimization Complete.');
catch ME
    disp('Optimization failed or interrupted.');
    disp(ME.message);
    phi0_surf_opt = reshape(phi0_guess_flat, N, N); % Fallback
end
toc;

%% Validation
% Reconstruct full state from optimized surface
phi0_3d_opt = derive_phi0_3d(phi0_surf_opt, K, z, Bu);
[F1_opt, G1_opt, Phi1_opt] = calculate_higher_order(phi0_3d_opt, K, kx, ky, z, Bu, N, nz);

% % Compare w (Vertical Velocity)
% % w = Ro * (F1_x + G1_y)
% w_true = compute_w(F1_true, G1_true, kx, ky, Ro);
% w_opt = compute_w(F1_opt, G1_opt, kx, ky, Ro);

% Compute horizontal velocity
[usurf_true, vsurf_true] = compute_horizontal_surf(phi0_3d_true, F1_true, G1_true, Phi1_true, kx, ky, Ro, z, dx, dz);
[usurf_opt, vsurf_opt] = compute_horizontal_surf(phi0_3d_opt, F1_opt, G1_opt, Phi1_opt, kx, ky, Ro, z, dx, dz);

% Metrics
% z_idx = floor(nz/2);
% w_t_mid = w_true(:,:,z_idx);
% w_r_mid = w_opt(:,:,z_idx);
corr_score = corr2(usurf_true, usurf_opt);
rmse = sqrt(mean((usurf_true(:) - usurf_opt(:)).^2));
rel_err = rmse / std(usurf_true(:));

% fprintf('Validation at z-index %d:\n', z_idx);
fprintf('  Correlation: %.4f\n', corr_score);
fprintf('  RMSE: %.4e\n', rmse);
fprintf('  Rel Error: %.4f\n', rel_err);

% Plotting
figure('Name', 'Inversion Results', 'Position', [100 100 1200 400]);
subplot(1,3,1); pcolor(X, Y, usurf_true); shading interp; colorbar; title('True u'); axis square;
subplot(1,3,2); pcolor(X, Y, usurf_opt); shading interp; colorbar; title('Inverted u (Opt)'); axis square;
subplot(1,3,3); pcolor(X, Y, usurf_opt - usurf_true); shading interp; colorbar; title('Difference'); axis square;
