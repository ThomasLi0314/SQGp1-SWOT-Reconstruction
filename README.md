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

## [`MATLAB Code/`](MATLAB%20Code/)

This is code I wrote at the beginning of this project where I wrote everything in MATLAB. These are all old codes I no longer use anymore.

## [`Output/`](Output/)

\_TODO: Storing all the Outputs of simulations, benchmarks etc.

## [`Presentations_Notes/`](Presentations_Notes/)

Storing all the TEX files for presentations, Notes and Reports.

### [`Endorsement_Letter/`](Presentations_Notes/Endorsement_Letter/)

Irrelevent to this project, for CPT Application.

### [`Presentation/`](Presentations_Notes/Presentation/)

Presentation slides I prepared for explaining this project to others.

### [`Ryan_notes/`](Presentations_Notes/Ryan_notes/)

Ryan's notes on this project. Introducing basics of SQGp1 model.

### [`SQG_notes/`](Presentations_Notes/SQG_notes/)

My notes for this project.

- [`Outline/`](Presentations_Notes/SQG_notes/Outline/)
- [`Literature_review/`](Presentations_Notes/SQG_notes/Literature_review/) — As it's name suggest.
- [`Chapter1_SQG/`](Presentations_Notes/SQG_notes/Chapter1_SQG/)
- [`Chapter_2_SQG1/`](Presentations_Notes/SQG_notes/Chapter_2_SQG1/)
- [`Chapter_3_SQG/`](Presentations_Notes/SQG_notes/Chapter_3_SQG/)
- [`Optimization_methods/`](Presentations_Notes/SQG_notes/Optimization_methods/)
- [`Benchmark_notes/`](Presentations_Notes/SQG_notes/Benchmark_notes/)

---

## [`Random_Testings/`](Random_Testings/)

Irrelevent Testing at early stage of this project for getting used to optimization toolbox.

### [`GPU_Python/`](Random_Testings/GPU_Python/)

I was trying to use GPU for calculation using python. This is the folder for test codes.

---

## [`Ryan_codes/`](Ryan_codes/)

Reference implementation from Ryan (mirror of [Empyreal092/SQGp1_Reconstruction](https://github.com/Empyreal092/SQGp1_Reconstruction)).

## [`SWOT_Inversion/`](SWOT_Inversion/) — main inversion codebase

Main Files Currently.

### [`Inversion_Python_SQG&SQGp1/`](SWOT_Inversion/Inversion_Python_SQG&SQGp1/)

The primary Python/JAX inversion pipeline.

Core modules:

- [`physics_functions.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/physics_functions.py) — Containing functions defined under SQGp1 model for calculating physical fields.
- [`cost_functions.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/cost_functions.py) — \_TODO: Cost function for Optimization (inversion)
- [`plotting.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/plotting.py) — Codes for plotting

Benchmark generations (each iterates on the previous):

- [`Benchmark/`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark/) — v1 prototype (`Benchmark_v1.ipynb`, `perturbation_setup.py`, `High_k_benchmark/`, `workspace_*.pkl`)
- [`Benchmark_v2/`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v2/) — adds case-split perturbation setups (`C1`, `C2`, `C3`) + [`pipeline_description.txt`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v2/pipeline_description.txt)
- [`Benchmark_v3/`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v3/)
- [`Benchmark_v4/`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v4/) — Current Version.

Includes [`Field_Verification.ipynb`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v4/Field_Verification.ipynb) for comparing field from Shafer's SQGp1 simulaltion and mine own inversion.
[`lowpass.py`](SWOT_Inversion/Inversion_Python_SQG%26SQGp1/Benchmark_v4/lowpass.py) A low pass filter copying the stle in Shafer's Code.

### [`Inversion_Channel_Model/`](SWOT_Inversion/Inversion_Channel_Model/)

Tried to implement the same procedure using channel model outputs.

### [`MATLAB_Codes_OLD/`](SWOT_Inversion/MATLAB_Codes_OLD/) — superseded

None, relevent. From early stages of this project.

### [`Shafer Simulation output/`](SWOT_Inversion/Shafer%20Simulation%20output/)

Output from Shafer's SQG simulator.

---

## [`Simulations/`](Simulations/) — forward solvers

### [`Shafer_SQG_Simulations/`](Simulations/Shafer_SQG_Simulations/)

Shafer's MATLAB SQG simulation

### [`SQGp1_Simulations/Shafer_Code/`](Simulations/SQGp1_Simulations/Shafer_Code/)

- [`qg_plus1.lnk`](Simulations/SQGp1_Simulations/Shafer_Code/qg_plus1.lnk) Link to Share google drive storing codes for SQGp1 simulation.

---

## Environment

_TODO: Python version, JAX (CPU vs CUDA), MATLAB version, key dependencies._
