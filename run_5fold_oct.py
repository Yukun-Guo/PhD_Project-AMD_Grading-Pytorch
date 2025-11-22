#!/usr/bin/env python3
"""
Run 5-fold cross validation specifically for OCT model
"""

import os
from TrainerFitKFold import run_kfold_training
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    print("Starting 5-fold cross validation for OCT AMD Grading...")
    print("Using configuration: configs/config_oct.toml")
    print("=" * 70)
    
    run_kfold_training('configs/config_oct.toml')