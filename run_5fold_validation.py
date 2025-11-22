#!/usr/bin/env python3
"""
Script to run 5-fold cross validation on AMD Grading data

This script will:
1. Load configuration from configs/config_oct.toml, configs/config_bio.toml, or configs/config_3d.toml
2. Perform 5-fold cross validation training
3. Evaluate each fold with comprehensive metrics
4. Save results to logs/5-k-validation_{model_type}/

Usage:
    python run_5fold_validation.py [config_path]
    
    # For OCT model (2D multi-modal)
    python run_5fold_validation.py configs/config_oct.toml
    
    # For Bio model (2D multi-modal)
    python run_5fold_validation.py configs/config_bio.toml
    
    # For 3D model (.mat files)
    python run_5fold_validation.py configs/config_3d.toml
    
    # Default (OCT) if no argument provided
    python run_5fold_validation.py

Results will be saved to:
    logs/5-k-validation_{model_type}/YYYYMMDD_HHMMSS/
    ├── fold_1/
    │   ├── validation_results.json
    │   ├── validation_metrics.csv
    │   ├── confusion_matrix.png
    │   └── validation_summary.txt
    ├── fold_2/
    │   └── ...
    ├── ...
    ├── cross_validation_summary.txt
    └── cross_validation_metrics.csv
"""

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
        # Determine model type and import appropriate module
        if '3d' in config_path.lower():
            from TrainerFitKFold3D import run_kfold_training
            model_type = '3D'
        elif 'oct' in config_path.lower():
            from TrainerFitKFold import run_kfold_training
            model_type = 'OCT'
        elif 'bio' in config_path.lower():
            from TrainerFitKFold import run_kfold_training
            model_type = 'Bio'
        else:
            from TrainerFitKFold import run_kfold_training
            model_type = 'Unknown'
        
        print(f"Starting 5-fold cross validation for {model_type} AMD Grading...")
        print(f"Using configuration: {config_path}")
        print("This will train 5 models and evaluate each fold comprehensively.")
        print(f"Results will be saved to logs/5-k-validation_{model_type.lower()}/")
        print("=" * 70)
        
        run_kfold_training(config_path)
    else:
        from TrainerFitKFold import run_kfold_training
        
        print("Starting 5-fold cross validation for OCT AMD Grading (default)...")
        print("Using configuration: configs/config_oct.toml")
        print("This will train 5 models and evaluate each fold comprehensively.")
        print("Results will be saved to logs/5-k-validation_oct/")
        print("=" * 70)
        print("Tips:")
        print("- Use 'python run_5fold_validation.py configs/config_bio.toml' for Bio model")
        print("- Use 'python run_5fold_validation.py configs/config_3d.toml' for 3D model")
        print("=" * 70)
        
        run_kfold_training('configs/config_oct.toml')