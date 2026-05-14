# SQG+1 Benchmark Run Log

- **Saved at:** 2026-04-29 15:59:43
- **Workspace file:** `workspace.pkl`

## 1. Parameters

| Parameter                  | Value                                   |
| -------------------------- | --------------------------------------- |
| Ro_sys (Rossby number)     | `0.05`                                  |
| epsilon (perturbation amp) | `0.05`                                  |
| Bu (Burger number)         | `1.0`                                   |
| Nx × Ny                    | `512 × 512`                             |
| Lx × Ly                    | `6.283185307179586 × 6.283185307179586` |
| rms(phi0_s)                | `0.046853`                              |
| max \|phi0_s\|             | `0.303521`                              |

## 2. Perturbation Cases

| Case       | rms(perturbation) | max \|perturbation\| | ratio to rms(phi0_s) |
| ---------- | ----------------- | -------------------- | -------------------- |
| sqgp1_data | 2.739662e-02      | 2.095175e-01         | 0.584738             |

## 3. Optimization Results

| Case       | Iterations | Time (s) | Final Loss | \|grad\|   |
| ---------- | ---------- | -------- | ---------- | ---------- |
| sqgp1_data | 10000      | 3127.2   | 3.4983e-02 | 1.6444e+01 |

## 4. Solver Settings

- Optimizer: **L-BFGS** (`jaxopt.LBFGS`)
- Initial guess: `phi0_s` (the QG surface streamfunction)

## 5. Additional Info

- **data_source:** `sqgp1_for_thomas.mat (real SQG+1 simulation)`
- **perturbation:** `none — eta_tar built directly from psi0n + Ro*p1n`
- **num_iterations:** `10000`
- **loss_threshold:** `2.50e-06`
- **grad_threshold:** `2.50e-05`
- **stagnation_ratio:** `0.9999`
- **notes:** `—`

I didn't add any filter to the initial data or change the cost function.
