import scipy.io as sio
import os
import glob

dirs = [
    r"D:\Documents\College\Research\Oceangrophy\Shafer_Project\Output",
    r"D:\Documents\College\Research\Oceangrophy\Shafer_Project\SQG_Simulations\Shafer Simulation output"
]

for d in dirs:
    if os.path.exists(d):
        for root, _, files in os.walk(d):
            for file in files:
                if file.endswith(".mat"):
                    path = os.path.join(root, file)
                    try:
                        mat = sio.loadmat(path)
                        if 'bout' in mat:
                            print(f"Found: {path} -> shape {mat['bout'].shape}")
                    except Exception as e:
                        print(f"Failed to load {path}: {e}")
