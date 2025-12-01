# Analysis with Multiple Averaging Types

## Overview
The `analyze_5fold_results.py` script has been updated to support all three averaging types (micro, macro, and weighted) for comprehensive cross-validation analysis.

## Changes Made

### 1. Metric Extraction
- Modified `extract_metrics_by_class()` to extract metrics for all three averaging types from the validation_metrics.csv files
- Structure: `metrics_data[avg_type][metric][fold_num][class_name]`
- Extracts Overall metrics separately for each averaging type (Micro, Macro, Weighted)

### 2. Statistics Calculation
- Updated `calculate_statistics()` to compute mean±std for each averaging type separately
- Structure: `stats[avg_type][metric_name][class_name]`
- Processes specificity for all averaging types (currently all use macro averaging for overall specificity)

### 3. Comparison Tables
- Modified `create_comparison_table()` to accept `avg_type` parameter
- Generates separate comparison tables for each averaging type

### 4. Visualizations
- Updated `create_visualizations()` to create separate plots for each averaging type
- Updated `create_perclass_heatmap()` to create separate heatmaps for each averaging type
- Files naming convention: `*_micro.png`, `*_macro.png`, `*_weighted.png`

### 5. Output Files
- **Summary Files**: Three separate text files for each averaging type
  - `analysis_summary_micro.txt`
  - `analysis_summary_macro.txt`
  - `analysis_summary_weighted.txt`

- **CSV Files**: Detailed comparison CSVs for each averaging type
  - `detailed_comparison_micro.csv`
  - `detailed_comparison_macro.csv`
  - `detailed_comparison_weighted.csv`

- **Visualization Files**: Plots for each averaging type
  - `overall_performance_comparison_micro.png`
  - `overall_performance_comparison_macro.png`
  - `overall_performance_comparison_weighted.png`
  - `perclass_performance_heatmap_micro.png`
  - `perclass_performance_heatmap_macro.png`
  - `perclass_performance_heatmap_weighted.png`

- **Raw Results**: JSON file containing all averaging types
  - `raw_analysis_results.json` (nested structure with avg_type as a level)

## Understanding Averaging Types

### Micro Averaging
- Calculates metrics globally by counting total true positives, false negatives, and false positives
- Gives equal weight to each sample
- Best for imbalanced datasets when you care about overall accuracy
- **Formula**: $\text{Metric}_{\text{micro}} = \frac{\sum_{i=1}^{n} TP_i}{\sum_{i=1}^{n} (TP_i + FP_i)}$

### Macro Averaging
- Calculates metrics for each class independently and then takes the unweighted mean
- Gives equal weight to each class
- Best when all classes are equally important
- **Formula**: $\text{Metric}_{\text{macro}} = \frac{1}{n} \sum_{i=1}^{n} \text{Metric}_i$

### Weighted Averaging
- Calculates metrics for each class and takes the average weighted by class support (number of samples)
- Accounts for class imbalance
- Best when you want to account for class distribution
- **Formula**: $\text{Metric}_{\text{weighted}} = \frac{\sum_{i=1}^{n} w_i \cdot \text{Metric}_i}{\sum_{i=1}^{n} w_i}$ where $w_i$ is the support of class $i$

## Results Location
All results are saved to: `analysis_results/<timestamp>/`

Example: `analysis_results/20251201_110128/`

## Quick Summary Output
The script now prints quick summaries for all three averaging types showing:
- Overall performance metrics for each model
- Sensitivity, Specificity, F1-Score, and AUC-ROC
- Mean ± Standard Deviation across 5 folds

## Usage
```bash
python analyze_5fold_results.py
```

The script automatically:
1. Finds the latest 5-fold validation results for OCT, BIO, and 3D models
2. Extracts metrics for all three averaging types
3. Computes statistics (mean±std) across folds
4. Generates separate summary files and visualizations for each averaging type
5. Saves comprehensive results including raw JSON data

## Key Metrics by Averaging Type

### Typical Values (from latest run)
| Model | Averaging | Sensitivity | Specificity | F1-Score | AUC-ROC |
|-------|-----------|-------------|-------------|----------|---------|
| OCT   | Micro     | 0.80±0.03   | 0.92±0.01   | 0.80±0.03| 0.96±0.01|
| OCT   | Macro     | 0.73±0.07   | 0.92±0.01   | 0.74±0.06| 0.94±0.02|
| OCT   | Weighted  | 0.80±0.03   | 0.92±0.01   | 0.80±0.03| 0.91±0.02|
| BIO   | Micro     | 0.83±0.03   | 0.93±0.01   | 0.83±0.03| 0.96±0.01|
| BIO   | Macro     | 0.78±0.05   | 0.93±0.01   | 0.77±0.05| 0.95±0.01|
| BIO   | Weighted  | 0.83±0.03   | 0.93±0.01   | 0.83±0.03| 0.93±0.02|
| 3D    | Micro     | 0.83±0.07   | 0.93±0.03   | 0.83±0.07| 0.97±0.02|
| 3D    | Macro     | 0.71±0.14   | 0.93±0.03   | 0.72±0.14| 0.95±0.02|
| 3D    | Weighted  | 0.83±0.07   | 0.93±0.03   | 0.82±0.07| 0.94±0.03|

## Notes
- Per-class metrics are identical across averaging types (they represent individual class performance)
- Overall metrics differ based on the averaging method used
- Specificity computation currently uses macro averaging for overall values across all averaging types
- The script preserves backward compatibility with existing folder structures and file formats
