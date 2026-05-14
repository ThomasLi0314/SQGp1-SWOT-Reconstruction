# SQG+1 Benchmark Run Log

- **Saved at:** 2026-04-27 22:56:57
- **Workspace file:** `workspace_0427_run2.pkl`

## 1. Parameters

| Parameter | Value |
|-----------|-------|
| Ro_sys (Rossby number)     | `0.1` |
| epsilon (perturbation amp) | `0.01` |
| Bu (Burger number)         | `1.0` |
| Nx × Ny                    | `512 × 512` |
| Lx × Ly                    | `6.283185307179586 × 6.283185307179586` |
| rms(phi0_s)                | `0.044998` |
| max \|phi0_s\|               | `0.158754` |

## 2. Perturbation Cases

| Case | rms(perturbation) | max \|perturbation\| | ratio to rms(phi0_s) |
|------|-------------------|----------------------|----------------------|
| A1_nothing | 0.000000e+00 | 0.000000e+00 | 0.000000 |
| B2_igw_packet | 4.689598e-04 | 1.728427e-03 | 0.010422 |
| C1_high_k_noise | 4.689598e-04 | 2.024387e-03 | 0.010422 |
| C2_low_k_noise | 4.689598e-04 | 1.520112e-03 | 0.010422 |
| C3_no_cutoff_noise | 4.689598e-04 | 1.020520e-03 | 0.010422 |

## 3. Optimization Results

| Case | Iterations | Time (s) | Final Loss | \|grad\| |
|------|------------|----------|------------|----------|
| A1_nothing | 1 | 4.6 | 1.3771e-26 | 2.1198e-11 |
| B2_igw_packet | 10000 | 2844.4 | 1.1356e-05 | 8.4048e-02 |
| C1_high_k_noise | 10000 | 3321.1 | 3.3340e-03 | 5.3540e-01 |
| C2_low_k_noise | 3000 | 846.9 | 1.2229e-02 | 1.1636e-02 |
| C3_no_cutoff_noise | 2000 | 589.3 | 1.2670e-02 | 2.0711e-02 |

## 4. Solver Settings

- Optimizer: **L-BFGS** (`jaxopt.LBFGS`)
- Initial guess: `phi0_s` (the QG surface streamfunction)

## 5. Additional Info

- **rng_seed:** `42`
- **perturbation_file:** `perturbation_setup_v3.py`
- **num_iterations:** `10000`
- **loss_threshold:** `1.00e-07`
- **grad_threshold:** `1.00e-06`
- **stagnation_ratio:** `0.9999`
- **notes:** `—`

