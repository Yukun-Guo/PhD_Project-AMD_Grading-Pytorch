#!/usr/bin/env python3
"""
Run 5-fold cross validation specifically for Bio model
"""

import os
from TrainerFitKFold import run_kfold_training

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    print("Starting 5-fold cross validation for Bio AMD Grading...")
    print("Using configuration: configs/config_bio.toml")
    print("=" * 70)
    
    run_kfold_training('configs/config_bio.toml')