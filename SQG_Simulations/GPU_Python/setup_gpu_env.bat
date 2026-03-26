@echo off
REM ============================================================
REM  Setup script for the GPU-accelerated SQG+1 Spectral Model
REM  Creates a conda environment with JAX GPU support
REM ============================================================

set ENV_NAME=sqg_gpu

echo ============================================================
echo  Setting up conda environment: %ENV_NAME%
echo  This will install JAX with CUDA 12 GPU support.
echo ============================================================

REM Create conda environment with Python 3.11
call conda create -n %ENV_NAME% python=3.11 -y

REM Activate the environment
call conda activate %ENV_NAME%

REM Install JAX with CUDA 12 support (compatible with CUDA 13.0 drivers)
pip install --upgrade "jax[cuda12]"

REM Install other dependencies
pip install jaxopt scipy matplotlib

echo.
echo ============================================================
echo  Setup complete!
echo  To use the GPU environment:
echo    conda activate %ENV_NAME%
echo    python spectral_main.py
echo ============================================================
pause
