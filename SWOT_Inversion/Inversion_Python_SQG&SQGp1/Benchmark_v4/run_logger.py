"""
run_logger.py
-------------
Generate a timestamped Markdown run-summary log alongside the saved workspace.

Usage (from notebook):
    from run_logger import write_run_log
    write_run_log(save_path, to_save, extra_info={"rng_seed": 42})
"""

import os
import datetime
import jax.numpy as jnp


def _rms(field):
    return float(jnp.sqrt(jnp.mean(jnp.array(field) ** 2)))


def write_run_log(pkl_path: str, workspace: dict, *, extra_info: dict = None):
    """
    Write a Markdown log file next to the saved .pkl workspace.

    Parameters
    ----------
    pkl_path   : str   – full path of the .pkl file just saved.
    workspace  : dict  – the same dict dumped to the pkl.
    extra_info : dict  – any additional key-value pairs to include
                         (e.g. rng_seed, perturbation_setup_file, notes …).
    """
    log_path = os.path.splitext(pkl_path)[0] + "_run_log.md"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── extract fields from workspace ─────────────────────────────────────
    Ro_sys  = workspace.get("Ro_sys")
    epsilon = workspace.get("epsilon")
    Bu      = workspace.get("Bu")
    Nx      = workspace.get("Nx")
    Ny      = workspace.get("Ny")
    Lx      = workspace.get("Lx")
    Ly      = workspace.get("Ly")
    phi0_s  = workspace.get("phi0_s")
    cases   = workspace.get("cases", {})
    results = workspace.get("benchmark_results", {})

    lines = []
    lines.append(f"# SQG+1 Benchmark Run Log")
    lines.append(f"")
    lines.append(f"- **Saved at:** {ts}")
    lines.append(f"- **Workspace file:** `{os.path.basename(pkl_path)}`")
    lines.append(f"")

    # ── 1. Physical / numerical parameters ────────────────────────────────
    lines.append(f"## 1. Parameters")
    lines.append(f"")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Ro_sys (Rossby number)     | `{Ro_sys}` |")
    lines.append(f"| epsilon (perturbation amp) | `{epsilon}` |")
    lines.append(f"| Bu (Burger number)         | `{Bu}` |")
    lines.append(f"| Nx × Ny                    | `{Nx} × {Ny}` |")
    lines.append(f"| Lx × Ly                    | `{Lx} × {Ly}` |")
    if phi0_s is not None:
        lines.append(f"| rms(phi0_s)                | `{_rms(phi0_s):.6f}` |")
        lines.append(f"| max \|phi0_s\|               | `{float(jnp.max(jnp.abs(jnp.array(phi0_s)))):.6f}` |")
    lines.append(f"")

    # ── 2. Perturbation cases ─────────────────────────────────────────────
    lines.append(f"## 2. Perturbation Cases")
    lines.append(f"")
    if cases:
        lines.append(f"| Case | rms(perturbation) | max \|perturbation\| | ratio to rms(phi0_s) |")
        lines.append(f"|------|-------------------|----------------------|----------------------|")
        signal_rms = _rms(phi0_s) if phi0_s is not None else 1.0
        for name, (perturb, _) in cases.items():
            p_rms = _rms(perturb)
            p_max = float(jnp.max(jnp.abs(jnp.array(perturb))))
            ratio = p_rms / signal_rms if signal_rms > 0 else 0.0
            lines.append(f"| {name} | {p_rms:.6e} | {p_max:.6e} | {ratio:.6f} |")
        lines.append(f"")
    else:
        lines.append(f"_(no case data in workspace)_")
        lines.append(f"")

    # ── 3. Optimization results ───────────────────────────────────────────
    lines.append(f"## 3. Optimization Results")
    lines.append(f"")
    if results:
        lines.append(f"| Case | Iterations | Time (s) | Final Loss | \|grad\| |")
        lines.append(f"|------|------------|----------|------------|----------|")
        for name, r in results.items():
            lines.append(
                f"| {name} "
                f"| {r.get('iters', '—')} "
                f"| {r.get('elapsed', 0):.1f} "
                f"| {r.get('final_loss', 0):.4e} "
                f"| {r.get('final_grad', 0):.4e} |"
            )
        lines.append(f"")
    else:
        lines.append(f"_(no results in workspace)_")
        lines.append(f"")

    # ── 4. Solver settings (from results metadata if present) ─────────────
    lines.append(f"## 4. Solver Settings")
    lines.append(f"")
    lines.append(f"- Optimizer: **L-BFGS** (`jaxopt.LBFGS`)")
    lines.append(f"- Initial guess: `phi0_s` (the QG surface streamfunction)")
    lines.append(f"")

    # ── 5. Extra info ─────────────────────────────────────────────────────
    if extra_info:
        lines.append(f"## 5. Additional Info")
        lines.append(f"")
        for k, v in extra_info.items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append(f"")

    # ── write ─────────────────────────────────────────────────────────────
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Run log saved to: {log_path}")
    return log_path
