# 5-Fold Cross Validation for AMD Grading Models

This document describes the updated K-fold cross validation system that supports OCT, Bio, and 3D models.

## Overview

The K-fold cross validation system now supports:
- **OCT Model**: Uses `configs/config_oct.toml` and saves results to `logs/5-k-validation_oct/`
- **Bio Model**: Uses `configs/config_bio.toml` and saves results to `logs/5-k-validation_bio/`
- **3D Model**: Uses `configs/config_3d.toml` and saves results to `logs/5-k-validation_3d/`

## Features

### Data Support
- **2D Models (OCT/Bio)**: Handles 4 image modalities: MNV, Fluid, GA, Drusen
- **3D Model**: Handles single .mat files with 3D volume data
- Automatic data type detection for proper preprocessing
- Same comprehensive validation metrics as `PredictionRun.py`

### Comprehensive Metrics
For each fold:
- Accuracy, Precision, Recall, F1-Score (macro, micro, weighted)
- Per-class metrics for all 4 AMD grading classes
- AUC-ROC scores
- Confusion matrices with visualizations

### Results Organization
```
logs/5-k-validation_{model_type}/YYYYMMDD_HHMMSS/
├── fold_1/
│   ├── validation_results.json      # Raw predictions and probabilities
│   ├── validation_metrics.csv       # Detailed metrics table
│   ├── confusion_matrix.png         # Confusion matrix visualization
│   └── validation_summary.txt       # Human-readable summary
├── fold_2/ ... fold_5/              # Same structure for all folds
├── cross_validation_summary.txt     # Summary across all folds
└── cross_validation_metrics.csv     # Mean ± std metrics
```

## Usage

### Method 1: Direct Script Execution
```bash
# For OCT model (2D multi-modal)
python TrainerFitKFold.py configs/config_oct.toml

# For Bio model (2D multi-modal)
python TrainerFitKFold.py configs/config_bio.toml

# For 3D model (.mat files)
python TrainerFitKFold3D.py configs/config_3d.toml

# Default (OCT) if no argument provided
python TrainerFitKFold.py
```

### Method 2: Convenience Scripts
```bash
# Generic script with argument (auto-detects model type)
python run_5fold_validation.py configs/config_oct.toml
python run_5fold_validation.py configs/config_bio.toml
python run_5fold_validation.py configs/config_3d.toml

# Model-specific scripts
python run_5fold_oct.py      # OCT model
python run_5fold_bio.py      # Bio model
python run_5fold_3d.py       # 3D model
```

## Key Improvements

1. **Model Type Detection**: Automatically detects OCT vs Bio from config filename
2. **Data Type Handling**: Uses appropriate preprocessing (`data_type="oct"` vs `data_type="bio"`)
3. **Flexible Configuration**: Supports any config file path via command line
4. **Organized Output**: Results saved to model-specific directories
5. **Comprehensive Validation**: Same metrics as existing validation pipeline

## Data Requirements

### 2D Models (OCT/Bio)
- Multi-modal images: `{caseID}_mnv.png`, `{caseID}_fluid.png`, `{caseID}_ga.png`, `{caseID}_drusen.png`
- Labels from CSV file specified in config
- 4-class AMD grading: Normal (0), Early AMD (1), Intermediate AMD (2), Advanced AMD (3)

### 3D Model
- 3D volume data: `{caseID}.mat` files 
- Labels from same CSV file as 2D models
- Same 4-class AMD grading system

## Configuration Differences

### OCT Model (`config_oct.toml`)
- Data path: `./data/img_oct`
- Uses OCT-specific normalization and transforms
- Image dimensions: 304x304 (2D)
- Batch size: 8

### Bio Model (`config_bio.toml`)  
- Data path: `./data/img_bio`
- Uses Bio-specific normalization and transforms
- Image dimensions: 304x304 (2D)
- Batch size: 8

### 3D Model (`config_3d.toml`)
- Data path: `./data/oct3d`
- Uses 3D-specific normalization and transforms
- Volume dimensions: 192x256x256 (3D)
- Batch size: 16

All models:
- Same CSV label file: `./data/trainingset.csv`
- Same number of classes: 4

## Cross-Validation Summary

After all 5 folds complete, the system generates:
- Mean ± standard deviation for all metrics across folds
- Per-fold performance comparison
- Overall model performance assessment

This enables robust evaluation of model performance with proper statistical measures of uncertainty.