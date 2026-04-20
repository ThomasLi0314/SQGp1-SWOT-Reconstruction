import jax
import jax.numpy as jnp
from jax import jit
import jaxopt
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Input subfiles. 
from cost_functions import cost_function_fmin
from ssh_setup import ssh_setup
from physics_functions import forward_ssh

import scipy.io as sio

# 64-bit precision is highly recommended for accurate optimization and numerics
jax.config.update("jax_enable_x64", True)

def find_shafer_data(N):
    search_dir = r"D:\Documents\College\Research\Oceangrophy\Shafer_Project\SQG_Simulations\Shafer Simulation output"
    if not os.path.exists(search_dir):
        return None, None
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".mat"):
                file_path = os.path.join(root, file)
                try:
                    mat = sio.loadmat(file_path)
                    if 'bout' in mat:
                        b_s_data = mat['bout']
                        if len(b_s_data.shape) >= 2 and b_s_data.shape[0] == N and b_s_data.shape[1] == N:
                            return file_path, b_s_data[:, :, -1]
                except Exception:
                    pass
    return None, None

def run_single_trial(N, scale):
    print(f"\n--- Benchmark Config: Grid {N}x{N}, Noise Scale {scale} ---")
    Nx = Ny = N
    Lx = Ly = 2.0 * jnp.pi
    Ro = 0.1
    Bu = 1.0
    f = 1.0
    epsilon = Ro
    case_num = 4

    dx = Lx / Nx
    dy = Ly / Ny
    x = jnp.arange(Nx) * dx
    y = jnp.arange(Ny) * dy
    X, Y = jnp.meshgrid(x, y, indexing='ij')

    dk = 2.0 * jnp.pi / Nx
    dl = 2.0 * jnp.pi / Ny
    k_zonal = jnp.concatenate([jnp.arange(Nx // 2), jnp.arange(-Nx // 2, 0)]) * dk
    l_meridional = jnp.concatenate([jnp.arange(Ny // 2), jnp.arange(-Ny // 2, 0)]) * dl
    kx, ky = jnp.meshgrid(k_zonal, l_meridional, indexing='ij')

    K2 = kx**2 + ky**2
    K = jnp.sqrt(K2)
    inv_K2 = jnp.where(K2 > 0, 1.0 / K2, 0.0)

    mu = jnp.sqrt(Bu) * K
    inv_mu = jnp.where(mu > 0, 1.0 / mu, 0.0)

    # Base SSH setup from Shafer Simulation
    file_path, b_s_custom = find_shafer_data(N)
    if b_s_custom is None:
        print(f"  [ERROR] No {N}x{N} Shafer simulation data found! Skipping benchmark scale {scale}...")
        return None
        
    print(f"  Loaded true simulation data from: {file_path}")
    b_s_custom = b_s_custom / jnp.max(jnp.abs(b_s_custom))
    b_s_hat = jnp.fft.fft2(b_s_custom)
    phi0_s_hat_base = jnp.where(mu > 0, b_s_hat / mu, 0.0)
    phi0_s = jnp.real(jnp.fft.ifft2(phi0_s_hat_base))

    # Perturbed SSH Setup with filtered noise
    max_phi0_s = jnp.max(phi0_s)
    key2 = jax.random.PRNGKey(0)
    
    if scale > 0.0:
        noise = jax.random.normal(key2, (Nx, Ny))
        noise_hat = jnp.fft.fft2(noise)
        noise_hat = jnp.where(K < 5, noise_hat, 0.0)
        noise_filtered = jnp.real(jnp.fft.ifft2(noise_hat))
        phi0_s_true = phi0_s + scale * max_phi0_s * noise_filtered / jnp.max(jnp.abs(noise_filtered))
    else:
        phi0_s_true = phi0_s

    phi0_s_true_hat = jnp.fft.fft2(phi0_s_true)

    # Forward Part: generate "True observation"
    eta_s_hat_true = forward_ssh(phi0_s_true_hat, f, kx, ky, mu, inv_mu, K2, inv_K2, Bu, epsilon)

    # Initial guess for optimization
    phi0_s_guess = phi0_s
    phi0_flat_guess = phi0_s_guess.ravel()

    # Optimization Setup
    @jit
    def loss_fn(phi0_s_flat):
        phi0_s_2d = phi0_s_flat.reshape(Nx, Ny)
        return cost_function_fmin(
            phi0_s_2d, f, kx, ky, mu, inv_mu, Bu, epsilon, K2, inv_K2, eta_s_hat_true
        )

    # Because JAX traces optimizations heavily, intercepting the exact variables
    # to evaluate custom metrics step-by-step causes the graph compiler to hang via Python PyTree unpacking.
    # To reliably benchmark, we use the compiled fast native `jaxopt` run with a tight
    # underlying tolerance on the gradient, effectively driving our L-infinity norm < 1e-5.
    solver = jaxopt.LBFGS(fun=loss_fn, maxiter=1000000, tol=1e-8)
    
    # ── Compilation (Warm-up) ──
    print("  Compiling JIT functions natively...")
    t_compile = time.time()
    
    run_jitted = jax.jit(solver.run)
    compiled_run = run_jitted.lower(phi0_flat_guess).compile()
    
    compile_time = time.time() - t_compile
    print(f"  Compilation finished. (Took {compile_time:.2f}s)")
    
    # ── Actual Measurement ──
    print(f"  Starting optimized execution loop...")
    
    t0 = time.time()
    params, state = compiled_run(phi0_flat_guess)
    params.block_until_ready()
    elapsed = time.time() - t0
    
    phi0_s_opt = params.reshape(Nx, Ny)
    final_loss = float(state.value)
    final_grad_norm = float(state.error)  # L-BFGS sets .error to the grad norm
    num_iters = int(state.iter_num)
    
    # relative error: ||phi_opt - phi_true||_2 / ||phi_true||_2
    diff_norm = jnp.linalg.norm(phi0_s_opt - phi0_s_true)
    true_norm = jnp.linalg.norm(phi0_s_true)
    relative_err = float(diff_norm / true_norm)
    
    print(f"  Done -> {elapsed:.2f}s | iter={num_iters} | Loss={final_loss:.1e} | Rel_Err={relative_err:.4f}")
    
    return {
        "N": int(N),
        "scale": float(scale),
        "elapsed_time_s": float(elapsed),
        "num_iters": int(num_iters),
        "final_loss": float(final_loss),
        "final_grad_norm": float(final_grad_norm),
        "relative_error": float(relative_err)
    }

def main():
    # Defines the benchmarking sweep parameters
    # N_values = [64, 128, 256]
    N_values = [128]
    # scale_values = [0.001, 0.01, 0.05]
    scale_values = [0.05]
    
    results = []
    
    print("=" * 60)
    print(" STARTING L-BFGS OPTIMIZATION BENCHMARK ")
    print("=" * 60)
    
    for N in N_values:
        for scale in scale_values:
            res = run_single_trial(N, scale)
            if res is not None:
                results.append(res)
            
    # Save the results
    df = pd.DataFrame(results)
    
    output_dir = r"D:\Documents\College\Research\Oceangrophy\Shafer_Project\Output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")
    
    df.to_csv(out_file, index=False)
    
    # ── Plots ──
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        use_sns = True
    except ImportError:
        use_sns = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if use_sns:
        sns.lineplot(data=df, x='N', y='elapsed_time_s', hue='scale', marker='o', ax=ax1)
        sns.lineplot(data=df, x='N', y='num_iters', hue='scale', marker='o', ax=ax2)
    else:
        for scale in scale_values:
            subset = df[df['scale'] == scale]
            ax1.plot(subset['N'], subset['elapsed_time_s'], marker='o', label=f'Scale {scale}')
            ax2.plot(subset['N'], subset['num_iters'], marker='o', label=f'Scale {scale}')
        ax1.legend(title='scale')
        ax2.legend(title='scale')

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xticks(N_values)
    ax1.set_xticklabels([str(n) for n in N_values])
    ax1.set_title('Runtime vs Grid Size')
    ax1.set_xlabel('Grid Size N (NxN)')
    ax1.set_ylabel('Elapsed Time (s)')
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    ax2.set_xscale('log', base=2)
    ax2.set_xticks(N_values)
    ax2.set_xticklabels([str(n) for n in N_values])
    ax2.set_title('Iterations vs Grid Size')
    ax2.set_xlabel('Grid Size N (NxN)')
    ax2.set_ylabel('Number of Iterations')
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"benchmark_plots_{timestamp}.png")
    plt.savefig(plot_file, dpi=300)

    print("=" * 60)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:\n{out_file}")
    print(f"Plots saved to:\n{plot_file}")

if __name__ == "__main__":
    main()
