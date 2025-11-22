#!/usr/bin/env python3
"""
Run 5-fold cross validation specifically for 3D model
"""
import os
from TrainerFitKFold3D import run_kfold_training
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == "__main__":
    print("Starting 5-fold cross validation for 3D AMD Grading...")
    print("Using configuration: configs/config_3d.toml")
    print("=" * 70)
    
    run_kfold_training('configs/config_3d.toml')