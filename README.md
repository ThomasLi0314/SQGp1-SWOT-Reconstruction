# SQGp1-SWOT-Reconstruction

_TODO: One-paragraph project overview — what the SQG+1 inverse problem is, what data you're inverting (SWOT SSH), what the goal of this repo is, and who the collaborators are (Shafer, Ryan, Tatsu, advisor)._

---

## Repository Layout

```
Shafer_Project/
├── MATLAB Code/                # MATLAB optimizer benchmarking
├── Output/                     # Saved simulation & benchmark outputs
├── Presentations_Notes/        # Writeups, slides, LaTeX notes
├── Random_Testings/            # Sandbox / experiments
├── Ryan_codes/                 # Ryan's reference implementation (JAX)
├── SWOT_Inversion/             # Main inversion codebase
├── Simulations/                # Forward simulation drivers
├── SQG_commands.tex            # Shared LaTeX macro definitions
```

---

## Top-level files

- [`SQG_commands.tex`](SQG_commands.tex) — _TODO: shared LaTeX macros used across the writeups in `Presentations_Notes/`._

---

## [`MATLAB Code/`](MATLAB%20Code/)

_TODO: high-level purpose of this folder._

### [`Matlab_Optimization_Benchmark/`](MATLAB%20Code/Matlab_Optimization_Benchmark/)

- [`optimization_tool_benchmark.m`](MATLAB%20Code/Matlab_Optimization_Benchmark/optimization_tool_benchmark.m) — _TODO_
- [`Optimization_functions.m`](MATLAB%20Code/Matlab_Optimization_Benchmark/Optimization_functions.m) — _TODO_

---

## [`Output/`](Output/)

_TODO: overall convention — timestamped run folders contain `potentials.npz` + comparison PNGs._

- `20260317_*`, `20260318_*`, `20260327_*` `_Shafer_Simulation_random_IC/` — _TODO: dated forward-simulation outputs from random initial conditions._
- [`Benchmark Result/`](Output/Benchmark%20Result/) — _TODO: dated benchmark sweeps (folders `0412`, `0413`, `0414`, `0416`, `0427`, `0428`)._
- [`Default_Case_4/run_1/`](Output/Default_Case_4/run_1/) — _TODO: a baseline 512×512 reference run used for comparison._
- [`SQGp1_Benchmark/`](Output/SQGp1_Benchmark/) — _TODO: SQG+1 inversion benchmark results._
  - [`0429/`](Output/SQGp1_Benchmark/0429/), [`0430/`](Output/SQGp1_Benchmark/0430/) — _TODO: per-day workspace pickles + `workspace_run_log.md` notes._
  - [`Global/`](Output/SQGp1_Benchmark/Global/) — _TODO: diagnostics related to the Kriging/GLS loss exploration (initial comparisons, weighted PSD plots)._

---

## [`Presentations_Notes/`](Presentations_Notes/)

_TODO: all writing — slides, advisor notes, the running thesis-style notes document._

### [`Endorsement_Letter/`](Presentations_Notes/Endorsement_Letter/)

_TODO: source + compiled PDF of an endorsement / recommendation letter._

### [`Presentation/`](Presentations_Notes/Presentation/)

- [`Presentation.tex`](Presentations_Notes/Presentation/Presentation.tex), [`Presentation.pdf`](Presentations_Notes/Presentation/Presentation.pdf) — _TODO_
- [`Figure/`](Presentations_Notes/Presentation/Figure/) — _TODO: figures used in the slide deck._

### [`Ryan_notes/`](Presentations_Notes/Ryan_notes/)

_TODO: archival LaTeX notes from Ryan (`main.tex`, `main_archive.*`, `command.tex`)._

### [`SQG_notes/`](Presentations_Notes/SQG_notes/)

_TODO: master `SQG_notes.tex` document organized into chapters._

- [`Outline/`](Presentations_Notes/SQG_notes/Outline/) — _TODO: thesis/proposal outline (`main.tex`, `proposal.sty`, `references.bib`)._
- [`Literature_review/`](Presentations_Notes/SQG_notes/Literature_review/) — _TODO: Chapter 0, with referenced papers (e.g. Ross et al. 2023)._
- [`Chapter1_SQG/`](Presentations_Notes/SQG_notes/Chapter1_SQG/) — _TODO_
- [`Chapter_2_SQG1/`](Presentations_Notes/SQG_notes/Chapter_2_SQG1/) — _TODO_
- [`Chapter_3_SQG/`](Presentations_Notes/SQG_notes/Chapter_3_SQG/) — _TODO_
- [`Optimization_methods/`](Presentations_Notes/SQG_notes/Optimization_methods/) — _TODO_
- [`Benchmark_notes/`](Presentations_Notes/SQG_notes/Benchmark_notes/) — _TODO_

---

## [`Random_Testings/`](Random_Testings/)

_TODO: scratch / experimental code, not part of the main pipeline._

### [`GPU_Python/`](Random_Testings/GPU_Python/)

_TODO: GPU port of the spectral solver — same module layout as `SWOT_Inversion/Inversion_Python_SQG&SQGp1` but configured for CUDA._

- [`spectral_main.py`](Random_Testings/GPU_Python/spectral_main.py), [`physics_functions.py`](Random_Testings/GPU_Python/physics_functions.py), [`cost_functions.py`](Random_Testings/GPU_Python/cost_functions.py), [`plotting.py`](Random_Testings/GPU_Python/plotting.py), [`ssh_setup.py`](Random_Testings/GPU_Python/ssh_setup.py) — _TODO_
- [`setup_gpu_env.bat`](Random_Testings/GPU_Python/setup_gpu_env.bat) — _TODO: Windows env setup for JAX-CUDA._

---

## [`Ryan_codes/`](Ryan_codes/)

Reference implementation from Ryan (mirror of [Empyreal092/SQGp1_Reconstruction](https://github.com/Empyreal092/SQGp1_Reconstruction)). _TODO: how you use it as ground truth for your own port._

- [`Method_summary.pdf`](Ryan_codes/Method_summary.pdf) — math + algorithm summary
- [`AGU24_eSQGp1recons_Poster.pdf`](Ryan_codes/AGU24_eSQGp1recons_Poster.pdf) — AGU 2024 poster
- [`SQGp1_dim_snippet.ipynb`](Ryan_codes/SQGp1_dim_snippet.ipynb) — _TODO_
- [`channel_numbered_notebooks/`](Ryan_codes/channel_numbered_notebooks/) — frozen post-Tatsu/Shafer-meeting notebooks (the AGU poster pipeline): `0_Truth_data`, `1_Fourier_analy`, `2_Geostrophic_balance`, `3_Cyclo_JAXparrow`, `4_SQGp1`, `45_Cyclo`
- [`SQGp1_numbered_notebooks/`](Ryan_codes/SQGp1_numbered_notebooks/) — same procedure applied to a SQG+1 model run (`0_Truth_data`, `4_SQGp1_artificialdata`)
- [`subroutine/`](Ryan_codes/subroutine/) — helpers: `UV_calc.py`, `isospec_rfft.py`, `rel_err.py`, `rfft2.py`, [`fastjmd95/`](Ryan_codes/subroutine/fastjmd95/)
- [`data/`](Ryan_codes/data/)
  - [`data_LLC/`](Ryan_codes/data/data_LLC/) — MITgcm LLC4320 Cape Basin (download instructions in [`Download.txt`](Ryan_codes/data/data_LLC/Download.txt))
  - [`data_matlab/SQGp1MATLAB.mat`](Ryan_codes/data/data_matlab/SQGp1MATLAB.mat) — _TODO_

---

## [`SWOT_Inversion/`](SWOT_Inversion/) — main inversion codebase

_TODO: this is where the active development lives._

### [`Inversion_Python_SQG&SQGp1/`](SWOT_Inversion/Inversion_Python_SQG&SQGp1/)

The primary Python/JAX inversion pipeline.

Core modules:

- [`spectral_main.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/spectral_main.py) — _TODO: main entry point._
- [`spectral_main_notebook.ipynb`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/spectral_main_notebook.ipynb) — _TODO: notebook driver._
- [`physics_functions.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/physics_functions.py) — _TODO: surface velocity, vorticity, SQG+1 operators._
- [`cost_functions.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/cost_functions.py) — _TODO: L-BFGS objective (currently L2; Kriging/GLS variant in progress per advisor 2026-04-28)._
- [`ssh_setup.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/ssh_setup.py) — _TODO: SSH initial-condition setup._
- [`plotting.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/plotting.py) — _TODO: side-by-side comparison figures, KE spectra, etc._
- [`run_benchmark.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/run_benchmark.py) — _TODO: standalone benchmark sweep over (grid size, noise scale)._
- [`tmp_check.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/tmp_check.py) — scratch
- [`Agent.md`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Agent.md) — chat log of refactor history (kept for reference)

Benchmark generations (each iterates on the previous):

- [`Benchmark/`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark/) — v1 prototype (`Benchmark_v1.ipynb`, `perturbation_setup.py`, `High_k_benchmark/`, `workspace_*.pkl`)
- [`Benchmark_v2/`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v2/) — adds case-split perturbation setups (`C1`, `C2`, `C3`) + [`pipeline_description.txt`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v2/pipeline_description.txt)
- [`Benchmark_v3/`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v3/) — adds [`run_logger.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v3/run_logger.py) + debug helper
- [`Benchmark_v4/`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v4/) — _TODO: current generation. Includes [`Field_Verification.ipynb`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v4/Field_Verification.ipynb), [`spectrum_diagnostics.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v4/spectrum_diagnostics.py), [`lowpass.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v4/lowpass.py), reference paper `2604.14009v1.pdf`._

### [`Inversion_Channel_Model/`](SWOT_Inversion/Inversion_Channel_Model/)

_TODO: same module layout but adapted for the channel-model geometry (vs. doubly-periodic)._

### [`MATLAB_Codes_OLD/`](SWOT_Inversion/MATLAB_Codes_OLD/) — superseded

- [`Toy_model_v1/`](SWOT_Inversion/MATLAB_Codes_OLD/Toy_model_v1/) — _TODO: original MATLAB toy model (3D Poisson, cost function, higher-order solve)._
- [`Spectral_Toy_model_v2/`](SWOT_Inversion/MATLAB_Codes_OLD/Spectral_Toy_model_v2/) — _TODO: spectral rewrite; the Python pipeline was ported from this._

### [`Shafer Simulation output/`](SWOT_Inversion/Shafer%20Simulation%20output/)

_TODO: timestamped outputs from the Shafer SQG simulator (forward runs feeding the inversion)._

---

## [`Simulations/`](Simulations/) — forward solvers

### [`Shafer_SQG_Simulations/`](Simulations/Shafer_SQG_Simulations/)

Shafer's MATLAB SQG forward solver.

- [`sqg.m`](Simulations/Shafer_SQG_Simulations/sqg.m) — _TODO: time-stepping core._
- [`runsqg_random_IC.m`](Simulations/Shafer_SQG_Simulations/runsqg_random_IC.m) — _TODO: driver with random initial conditions._
- [`fullspec.m`](Simulations/Shafer_SQG_Simulations/fullspec.m), [`grid2spec.m`](Simulations/Shafer_SQG_Simulations/grid2spec.m), [`spec2grid.m`](Simulations/Shafer_SQG_Simulations/spec2grid.m) — spectral transforms
- [`sqg_code_notes.txt`](Simulations/Shafer_SQG_Simulations/sqg_code_notes.txt) — notes

### [`SQGp1_Simulations/Shafer_Code/`](Simulations/SQGp1_Simulations/Shafer_Code/)

- [`qg_plus1.lnk`](Simulations/SQGp1_Simulations/Shafer_Code/qg_plus1.lnk) — _TODO: shortcut to Shafer's QG+1 code (target path on local machine)._

---

## Workflow / How to run

_TODO: short cookbook — e.g.:_

1. _Generate truth data with `Simulations/Shafer_SQG_Simulations/runsqg_random_IC.m` → outputs land in `SWOT_Inversion/Shafer Simulation output/`._
2. _Run inversion via `SWOT_Inversion/Inversion_Python_SQG&SQGp1/spectral_main_notebook.ipynb` or `python run_benchmark.py`._
3. _Outputs (`potentials.npz` + comparison PNGs) are written to `Output/<timestamp>_\*/`.\_
4. _Plotting / post-analysis from the `Benchmark_v*/` notebooks._

## Environment

_TODO: Python version, JAX (CPU vs CUDA), MATLAB version, key dependencies._
