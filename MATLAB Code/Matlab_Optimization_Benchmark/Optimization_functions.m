clear; clc; close all;

%% ================================================================
%  DEMO: lsqnonlin vs fminunc  —  Matrix Objective Targeting Zero
% ================================================================
%
%  PROBLEM:
%    Find x = [a; b; c]  such that  R(x) = 0,  where
%
%        R(x) = a*A  +  sin(b)*B  +  exp(c)*C  -  M
%
%    R is a 5x5 matrix. The goal is to make every element of R zero.
%
%  This is a system of 25 nonlinear equations in 3 unknowns —
%  a typical overdetermined nonlinear least-squares problem.
%
%  KEY DIFFERENCE between the two solvers:
%
%    lsqnonlin expects:   fun(x) -> R(x)        (the matrix / vector residual)
%                         It minimizes  sum(R.^2, 'all')  internally.
%                         It exploits the Jacobian structure of R for faster steps.
%
%    fminunc   expects:   fun(x) -> scalar       (you must square & sum yourself)
%                         It minimizes a generic scalar function.
%                         It has no knowledge that the scalar came from a residual.
%% ================================================================

rng(0);

%% --- Setup -------------------------------------------------------
n = 6000;   % matrix size

% Random basis matrices
A = randn(n, n);
B = randn(n, n);
C = randn(n, n);

% True parameters we want the solvers to recover
a_true = 2.0;
b_true = 1.0;     % sin(b_true) = sin(1) ≈ 0.841
c_true = 0.5;     % exp(c_true) = exp(0.5) ≈ 1.649
x_true = [a_true; b_true; c_true];

% Build the target matrix M = R(x_true) + 0  (exact, no noise)
M = a_true*A + sin(b_true)*B + exp(c_true)*C;

%% --- Residual function (matrix-valued) ---------------------------
%
%  This is the CORE objective. R(x) is a 5x5 matrix.
%  When R = 0, we have found the exact solution.
%
residual = @(x)  x(1)*A  +  sin(x(2))*B  +  exp(x(3))*C  -  M;

%% --- Objectives for each solver ----------------------------------

% lsqnonlin: hand it the raw residual matrix.
%   MATLAB will internally vectorise it to a column vector and minimise
%   the sum of squares.  You do NOT square it yourself.
fun_lsq  = @(x) residual(x);                       % R(x) 
% fminunc: must return a scalar.
%   We compute the Frobenius norm squared of the residual manually.
fun_fmin = @(x) sum(residual(x).^2, 'all');         % ||R(x)||^2_F

%% --- Initial guess -----------------------------------------------
x0 = [0; 0; 0];    % start far from the true solution

fprintf('True parameters:     a=%.4f   b=%.4f   c=%.4f\n', x_true);
fprintf('Initial cost ||R||²: %.6f\n\n', fun_fmin(x0));

%% --- Options (identical stopping criteria for fair comparison) ---
opts_lsq_trf = optimoptions('lsqnonlin', ...
    'Display',             'iter', ...
    'Algorithm',           'trust-region-reflective', ...
    'FunctionTolerance',   1e-12, ...
    'StepTolerance',       1e-12, ...
    'OptimalityTolerance', 1e-8);

opts_fmin = optimoptions('fminunc', ...
    'Display',             'iter', ...
    'Algorithm',           'quasi-newton', ...
    'FunctionTolerance',   1e-12, ...
    'StepTolerance',       1e-12, ...
    'OptimalityTolerance', 1e-8);

opts_lsq_lm = optimoptions('lsqnonlin', ...
    'Display',             'iter', ...
    'Algorithm',           'levenberg-marquardt', ...
    'FunctionTolerance',   1e-12, ...
    'StepTolerance',       1e-12, ...
    'OptimalityTolerance', 1e-8);

%% --- Solve with lsqnonlin ----------------------------------------

fprintf('===============  lsqnonlin  ===============\n');
tic;
[x_lsq, resnorm_lsq, ~, flag_lsq, out_lsq] = ...
    lsqnonlin(fun_lsq, x0, [], [], opts_lsq_trf);
t_lsq_trf = toc;
fprintf('\n  Exit flag %d  |  %d iters  |  %d func evals  |  %.4f s\n', ...
        flag_lsq, out_lsq.iterations, out_lsq.funcCount, t_lsq_trf);


fprintf('\n===============  lsqnonlin  ===============\n');
tic;
[x_lsq, resnorm_lsq, ~, flag_lsq, out_lsq] = ...
    lsqnonlin(fun_lsq, x0, [], [], opts_lsq_lm);
t_lsq_lm = toc;
fprintf('\n  Exit flag %d  |  %d iters  |  %d func evals  |  %.4f s\n', ...
        flag_lsq, out_lsq.iterations, out_lsq.funcCount, t_lsq_lm);


%% --- Solve with fminunc ------------------------------------------

fprintf('\n===============  fminunc  =================\n');
tic;
[x_fmin, fval_fmin, flag_fmin, out_fmin] = ...
    fminunc(fun_fmin, x0, opts_fmin);
t_fmin = toc;
fprintf('\n  Exit flag %d  |  %d iters  |  %d func evals  |  %.4f s\n', ...
        flag_fmin, out_fmin.iterations, out_fmin.funcCount, t_fmin);

%% --- Summary -----------------------------------------------------
fprintf('\n============================================================\n');
fprintf('%-12s  %8s  %8s  %8s   %10s  %10s  %8s\n', ...
        'Solver','a','b','c','Final ||R||²','Func Evals','Time(s)');
fprintf('%-12s  %8.5f  %8.5f  %8.5f   %10.2e  %10d  %8.4f\n', ...
        'True',       a_true,    b_true,    c_true,    0,           0,                  0);
fprintf('%-12s  %8.5f  %8.5f  %8.5f   %10.2e  %10d  %8.4f\n', ...
        'lsqnonlin_trf',  x_lsq(1), x_lsq(2),  x_lsq(3),  resnorm_lsq, out_lsq.funcCount,  t_lsq_trf);
fprintf('%-12s  %8.5f  %8.5f  %8.5f   %10.2e  %10d  %8.4f\n', ...
        'lsqnonlin_lm',  x_lsq(1), x_lsq(2),  x_lsq(3),  resnorm_lsq, out_lsq.funcCount,  t_lsq_lm);
fprintf('%-12s  %8.5f  %8.5f  %8.5f   %10.2e  %10d  %8.4f\n', ...
        'fminunc',    x_fmin(1),x_fmin(2), x_fmin(3),  fval_fmin,   out_fmin.funcCount, t_fmin);
fprintf('============================================================\n');

