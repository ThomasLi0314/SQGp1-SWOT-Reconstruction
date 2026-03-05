clear;
clc;
close all;

%% ================================================================
%  SQG+1 INVERSION BENCHMARK
%  Compares lsqnonlin vs fminunc on a problem that mirrors the
%  real SQG+1 SSH-inversion:  find phi0_hat  s.t.  eta_hat(phi0_hat) = eta_hat_true
%  Residual targets ZERO (no noise added to the true solution).
%% ================================================================

rng(42);

%% --- Physical / spectral parameters (mirrors initialize.m) -------
Nx  = 64;  Ny  = 64;         % grid size (increase for harder problem)
Lx  = 2*pi; Ly  = 2*pi;     % domain length
f   = 1;                      % Coriolis parameter
Bu  = 1;                      % Burger number
Ro  = 0.1;                    % Rossby number  (epsilon)
epsilon = Ro;

dx = Lx/Nx;  dy = Ly/Ny;

% --- Wavenumber arrays (same convention as initialize.m) ----------
k_zonal      = [0:(Nx/2-1), -Nx/2:-1] * (2*pi/Lx);
l_meridional = [0:(Ny/2-1), -Ny/2:-1] * (2*pi/Ly);
[kx, ky] = ndgrid(k_zonal, l_meridional);

K2    = kx.^2 + ky.^2;
mu    = sqrt(Bu) * sqrt(K2);

inv_K2         = zeros(size(K2));
inv_K2(K2 > 0) = 1 ./ K2(K2 > 0);

inv_mu         = zeros(size(mu));
inv_mu(mu > 0) = 1 ./ mu(mu > 0);

%% --- Generate a smooth "true" phi0_hat ---------------------------
% Low-pass filter: only keep |k| <= k_cutoff  so the field is smooth
k_cutoff = 4;
phi0_true_hat = zeros(Nx, Ny);
phi0_true_hat(K2 <= k_cutoff^2 & K2 > 0) = ...
    (randn(sum(sum(K2 <= k_cutoff^2 & K2 > 0)), 1) + ...
  1i*randn(sum(sum(K2 <= k_cutoff^2 & K2 > 0)), 1));

% Real physical fields => Hermitian symmetry (enforce weakly)
phi0_true_hat = phi0_true_hat / norm(phi0_true_hat, 'fro');   % normalise

%% --- Forward model (mirrors cost_function.m pipeline) -----------
%  Given phi0_hat, compute eta_hat.
%  All sub-functions are inlined to keep the benchmark self-contained.

function J_hat = cyclogeo(phi_hat, kx, ky)
    % Cyclogeostrophic Jacobian  2*J(phi, phi) = 2*(phi_xx*phi_yy - phi_xy^2)
    phi_xx = ifft2(phi_hat .* (-kx.^2));
    phi_yy = ifft2(phi_hat .* (-ky.^2));
    phi_xy = ifft2(phi_hat .* (-kx.*ky));
    J_hat  = 2 * fft2(phi_xx .* phi_yy - phi_xy.^2);
end

function zeta_hat = vorticity(phi_hat, mu, inv_mu, kx, ky, K2, Bu)
    % 7-term SQG+1 vorticity correction (mirrors vorticity_term.m)
    phi_x    = ifft2(phi_hat .* (1i*kx));
    phi_y    = ifft2(phi_hat .* (1i*ky));
    phi_z    = ifft2(phi_hat .* mu);
    phi_zz   = ifft2(phi_hat .* mu.^2);
    phi_zx   = ifft2(phi_hat .* mu .* (1i*kx));
    phi_zy   = ifft2(phi_hat .* mu .* (1i*ky));
    phi_zzx  = ifft2(phi_hat .* mu.^2 .* (1i*kx));
    phi_zzy  = ifft2(phi_hat .* mu.^2 .* (1i*ky));
    phi_lap  = ifft2(phi_hat .* (-K2));
    phi_lapz = ifft2(phi_hat .* (-K2) .* mu);

    I1 = fft2(phi_x.*phi_zzx + phi_y.*phi_zzy);
    I2 = fft2(phi_lap .* phi_zz);
    I3 = fft2(2*(phi_zx.^2 + phi_zy.^2));
    I4 = fft2(2*phi_z .* phi_lapz);
    I5 = fft2(phi_z .* phi_zz) .* K2.^2 .* inv_mu;
    I6 = fft2(phi_y .* phi_z) .* (1i*ky) .* mu;
    I7 = fft2(phi_x .* phi_z) .* (1i*kx) .* mu;

    zeta_hat = (I1 + I2 + I3 + I4 + I5 + I6 + I7) / Bu;
end

function eta_hat = forward(phi_hat, f, epsilon, kx, ky, mu, inv_mu, K2, inv_K2, Bu)
    % SQG+1 forward map: phi0_hat -> eta_hat
    J_hat    = cyclogeo(phi_hat, kx, ky);
    zeta_hat = vorticity(phi_hat, mu, inv_mu, kx, ky, K2, Bu);
    p1_hat   = -(f * zeta_hat + J_hat) .* inv_K2;   % ∇²p¹ = f·ζ¹ + 2J  =>  p¹ = -(f·ζ+2J)/K²
    eta_hat  = f * phi_hat + epsilon * p1_hat;
end

%% --- Build true observation  -------------------------------------
eta_hat_true = forward(phi0_true_hat, f, epsilon, kx, ky, mu, inv_mu, K2, inv_K2, Bu);

%% --- Pack / unpack helpers (complex -> real vector) --------------
%  lsqnonlin & fminunc expect a real vector.
%  We stack  [real(phi_hat(:));  imag(phi_hat(:))].
N2  = Nx * Ny;

pack   = @(ph) [real(ph(:)); imag(ph(:))];
unpack = @(v)  reshape(v(1:N2), Nx, Ny) + 1i*reshape(v(N2+1:end), Nx, Ny);

x_true = pack(phi0_true_hat);    % reference (what we hope to recover)

%% --- Objective Functions -----------------------------------------
% Residual matrix in spectral space, then packed to real vector
res_fn = @(v) ...
    pack(forward(unpack(v), f, epsilon, kx, ky, mu, inv_mu, K2, inv_K2, Bu) - eta_hat_true);

fun_lsq  = @(v) res_fn(v);                        % for lsqnonlin
fun_fmin = @(v) sum(res_fn(v).^2);                % for fminunc

%% --- Initial guess  (perturb true solution slightly) -------------
perturb = 0.05;   % 5% perturbation in spectral amplitude
x0 = pack(phi0_true_hat .* (1 + perturb*(randn(Nx,Ny)+1i*randn(Nx,Ny))));

fprintf('Problem: %dx%d spectral grid  (%d real unknowns)\n', Nx, Ny, 2*N2);
fprintf('Initial residual norm²: %.4e\n', fun_fmin(x0));
fprintf('True residual norm²:    %.4e  (should be ~0)\n\n', fun_fmin(x_true));

%% --- Solver options ----------------------------------------------
options_lsq = optimoptions('lsqnonlin', ...
    'Display',             'iter', ...
    'MaxIterations',       200, ...
    'FunctionTolerance',   1e-12, ...
    'StepTolerance',       1e-12, ...
    'OptimalityTolerance', 1e-8);

options_fmin = optimoptions('fminunc', ...
    'Display',             'iter', ...
    'Algorithm',           'quasi-newton', ...
    'MaxIterations',       500, ...
    'FunctionTolerance',   1e-12, ...
    'StepTolerance',       1e-12, ...
    'OptimalityTolerance', 1e-8);

%% --- Run lsqnonlin -----------------------------------------------
fprintf('\n================  RUNNING LSQNONLIN  ================\n');
fprintf('  Algorithm : trust-region-reflective (default)\n');
fprintf('  Unknowns  : %d real DOF\n\n', 2*N2);
tic;
[x_lsq, resnorm_lsq, ~, exitflag_lsq, output_lsq] = ...
    lsqnonlin(fun_lsq, x0, [], [], options_lsq);
time_lsq = toc;
fprintf('\n  --> Finished in %.4f s  |  Exit flag: %d  |  Iterations: %d\n', ...
        time_lsq, exitflag_lsq, output_lsq.iterations);

%% --- Run fminunc -------------------------------------------------
fprintf('\n================  RUNNING FMINUNC  ==================\n');
fprintf('  Algorithm : quasi-newton (BFGS)\n');
fprintf('  Unknowns  : %d real DOF\n\n', 2*N2);
tic;
[x_fmin, fval_fmin, exitflag_fmin, output_fmin] = ...
    fminunc(fun_fmin, x0, options_fmin);
time_fmin = toc;
fprintf('\n  --> Finished in %.4f s  |  Exit flag: %d  |  Iterations: %d\n', ...
        time_fmin, exitflag_fmin, output_fmin.iterations);

%% --- Final benchmark summary -------------------------------------
phi_lsq  = unpack(x_lsq);
phi_fmin = unpack(x_fmin);

err_lsq  = norm(phi_lsq  - phi0_true_hat, 'fro') / norm(phi0_true_hat, 'fro');
err_fmin = norm(phi_fmin - phi0_true_hat, 'fro') / norm(phi0_true_hat, 'fro');

fprintf('\n================ BENCHMARK RESULTS ================\n');
fprintf('  Grid: %dx%d   |   DOF: %d\n', Nx, Ny, 2*N2);
fprintf('----------------------------------------------------\n');
fprintf('%-12s  %12s  %12s  %10s  %8s  %8s\n', ...
        'Solver', 'Final Cost', 'Rel phi Err', 'Func Evals', 'Iters', 'Time(s)');
fprintf('%-12s  %12.4e  %12.4e  %10d  %8d  %8.4f\n', ...
        'lsqnonlin',  resnorm_lsq, err_lsq,  output_lsq.funcCount,  output_lsq.iterations,  time_lsq);
fprintf('%-12s  %12.4e  %12.4e  %10d  %8d  %8.4f\n', ...
        'fminunc',    fval_fmin,   err_fmin, output_fmin.funcCount, output_fmin.iterations, time_fmin);
fprintf('====================================================\n');
fprintf('\nNoise-floor cost (irreducible):  0  (exact true solution used)\n');