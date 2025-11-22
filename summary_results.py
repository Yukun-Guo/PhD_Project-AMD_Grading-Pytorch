#!/usr/bin/env python3
"""
Simple Summary Script for 5-Fold Cross-Validation Results

This script provides a focused summary of sensitivity, specificity, F1-score, 
and AUC-ROC with mean and standard deviation across 5 folds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def print_summary_table():
    """Print a clean summary table"""
    
    # Find the latest analysis results
    analysis_dir = Path("analysis_results")
    if not analysis_dir.exists():
        print("No analysis results found. Please run analyze_5fold_results.py first.")
        return
    
    latest_dir = max([d for d in analysis_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    results_file = latest_dir / "raw_analysis_results.json"
    
    if not results_file.exists():
        print("Analysis results not found.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    models = list(results.keys())
    class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
    
    print("=" * 100)
    print("5-FOLD CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 100)
    print(f"Models: {', '.join([m.upper() for m in models])}")
    print(f"Generated from: {latest_dir}")
    print()
    
    # Overall Performance Summary
    print("OVERALL PERFORMANCE (Mean ± Standard Deviation)")
    print("-" * 100)
    print(f"{'Model':<8} {'Sensitivity':<15} {'Specificity':<15} {'F1-Score':<15} {'AUC-ROC':<15}")
    print("-" * 100)
    
    for model in models:
        row = f"{model.upper():<8}"
        
        for metric in ['sensitivity', 'specificity', 'f1_score', 'auc_roc']:
            if metric in results[model] and 'Overall' in results[model][metric]:
                stats = results[model][metric]['Overall']
                value_str = f"{stats['mean']:.4f}±{stats['std']:.4f}"
                row += f" {value_str:<15}"
            else:
                row += f" {'N/A':<15}"
        
        print(row)
    
    print()
    
    # Per-Class Performance Summary
    print("PER-CLASS PERFORMANCE (Mean ± Standard Deviation)")
    print("-" * 100)
    
    for class_name in class_names:
        print(f"\n{class_name}:")
        print(f"{'Model':<8} {'Sensitivity':<15} {'Specificity':<15} {'F1-Score':<15} {'AUC-ROC':<15}")
        print("-" * 68)
        
        for model in models:
            row = f"{model.upper():<8}"
            
            for metric in ['sensitivity', 'specificity', 'f1_score', 'auc_roc']:
                if (metric in results[model] and 
                    class_name in results[model][metric]):
                    stats = results[model][metric][class_name]
                    value_str = f"{stats['mean']:.4f}±{stats['std']:.4f}"
                    row += f" {value_str:<15}"
                else:
                    row += f" {'N/A':<15}"
            
            print(row)
    
    print("\n" + "=" * 100)
    
    # Statistical significance (if available)
    if len(models) >= 2:
        print("\nSTATISTICAL SIGNIFICANCE TESTS (p-values for paired t-tests)")
        print("-" * 60)
        
        from scipy import stats
        
        model_pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i+1, len(models))]
        
        for metric in ['sensitivity', 'specificity', 'f1_score', 'auc_roc']:
            metric_name = metric.replace('_', '-').title()
            print(f"\n{metric_name}:")
            
            for model1, model2 in model_pairs:
                if (metric in results[model1] and 'Overall' in results[model1][metric] and
                    metric in results[model2] and 'Overall' in results[model2][metric]):
                    
                    values1 = results[model1][metric]['Overall']['values']
                    values2 = results[model2][metric]['Overall']['values']
                    
                    if len(values1) > 1 and len(values2) > 1:
                        _, p_value = stats.ttest_rel(values1, values2)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        print(f"  {model1.upper()} vs {model2.upper()}: p={p_value:.4f} {significance}")
    
    print("\nLegend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("=" * 100)


if __name__ == "__main__":
    print_summary_table()