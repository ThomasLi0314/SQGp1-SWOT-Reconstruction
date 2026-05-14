# SQG+1 Benchmark Run Log

- **Saved at:** 2026-04-29 00:10:48
- **Workspace file:** `workspace_0428_run1.pkl`

## 1. Parameters

| Parameter | Value |
|-----------|-------|
| Ro_sys (Rossby number)     | `0.1` |
| epsilon (perturbation amp) | `0.001` |
| Bu (Burger number)         | `1.0` |
| Nx × Ny                    | `128 × 128` |
| Lx × Ly                    | `6.283185307179586 × 6.283185307179586` |
| rms(phi0_s)                | `0.057917` |
| max \|phi0_s\|               | `0.214354` |

## 2. Perturbation Cases

| Case | rms(perturbation) | max \|perturbation\| | ratio to rms(phi0_s) |
|------|-------------------|----------------------|----------------------|
| A1_nothing | 0.000000e+00 | 0.000000e+00 | 0.000000 |
| B2_igw_packet | 5.716629e-05 | 2.089099e-04 | 0.000987 |
| C1_high_k_noise | 5.716629e-05 | 2.341706e-04 | 0.000987 |
| C2_low_k_noise | 5.716629e-05 | 2.141077e-04 | 0.000987 |
| C3_no_cutoff_noise | 5.716629e-05 | 1.550357e-04 | 0.000987 |

## 3. Optimization Results

| Case | Iterations | Time (s) | Final Loss | \|grad\| |
|------|------------|----------|------------|----------|
| A1_nothing | 1 | 2.5 | 2.7445e-29 | 4.8546e-14 |
| B2_igw_packet | 10000 | 277.8 | 2.6685e-09 | 4.6131e-06 |
| C1_high_k_noise | 10000 | 191.3 | 2.1908e-06 | 3.6732e-05 |
| C2_low_k_noise | 2000 | 60.7 | 2.2316e-05 | 8.0167e-06 |
| C3_no_cutoff_noise | 2000 | 59.8 | 3.3517e-05 | 2.9125e-06 |

## 4. Solver Settings

- Optimizer: **L-BFGS** (`jaxopt.LBFGS`)
- Initial guess: `phi0_s` (the QG surface streamfunction)

## 5. Additional Info

- **rng_seed:** `42`
- **perturbation_file:** `perturbation_setup_v3.py`
- **num_iterations:** `10000`
- **loss_threshold:** `1.00e-09`
- **grad_threshold:** `1.00e-08`
- **stagnation_ratio:** `0.9999`
- **notes:** `—`

