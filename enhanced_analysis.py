#!/usr/bin/env python3
"""
Enhanced 5-Fold Cross-Validation Results Analysis

This script provides comprehensive analysis with additional statistical tests
and visualization options.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_statistical_comparison_table():
    """Create a statistical comparison table with significance tests"""
    
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON ANALYSIS")
    print("="*80)
    
    # Find the latest analysis results
    analysis_dir = Path("analysis_results")
    if not analysis_dir.exists():
        print("No analysis results found. Please run analyze_5fold_results.py first.")
        return
    
    # Get the latest results
    timestamp_dirs = [d for d in analysis_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        print("No analysis results found.")
        return
    
    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
    results_file = latest_dir / "raw_analysis_results.json"
    
    if not results_file.exists():
        print("Raw results file not found.")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create statistical comparison
    models = list(results.keys())
    metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
    
    print(f"Models available: {', '.join([m.upper() for m in models])}")
    
    # Overall performance comparison table
    comparison_table = []
    
    for metric in metrics:
        metric_name = metric.replace('_', '-').title()
        print(f"\n{metric_name} Comparison:")
        print("-" * 50)
        
        row_data = {'Metric': metric_name}
        
        for model in models:
            if metric in results[model] and 'Overall' in results[model][metric]:
                stats_data = results[model][metric]['Overall']
                mean_val = stats_data['mean']
                std_val = stats_data['std']
                values = stats_data['values']
                
                # Calculate confidence interval (95%)
                if len(values) > 1:
                    confidence_interval = stats.t.interval(
                        0.95, len(values)-1,
                        loc=mean_val,
                        scale=stats.sem(values)
                    )
                    ci_text = f"[{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]"
                else:
                    ci_text = "N/A"
                
                row_data[f'{model.upper()}_Mean'] = mean_val
                row_data[f'{model.upper()}_Std'] = std_val
                row_data[f'{model.upper()}_CI'] = ci_text
                
                print(f"{model.upper():8}: {mean_val:.4f} ± {std_val:.4f} (95% CI: {ci_text})")
        
        comparison_table.append(row_data)
        
        # Perform statistical tests if we have multiple models
        if len(models) >= 2:
            model_pairs = [(models[i], models[j]) for i in range(len(models)) for j in range(i+1, len(models))]
            
            for model1, model2 in model_pairs:
                if (metric in results[model1] and 'Overall' in results[model1][metric] and
                    metric in results[model2] and 'Overall' in results[model2][metric]):
                    
                    values1 = results[model1][metric]['Overall']['values']
                    values2 = results[model2][metric]['Overall']['values']
                    
                    if len(values1) > 1 and len(values2) > 1:
                        # Perform paired t-test
                        t_stat, p_value = stats.ttest_rel(values1, values2)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        
                        print(f"  {model1.upper()} vs {model2.upper()}: t={t_stat:.3f}, p={p_value:.4f} {significance}")
    
    # Save comparison table
    comparison_df = pd.DataFrame(comparison_table)
    output_file = latest_dir / "statistical_comparison.csv" 
    comparison_df.to_csv(output_file, index=False)
    print(f"\nStatistical comparison saved to: {output_file}")
    
    return comparison_df


def create_detailed_per_class_analysis():
    """Create detailed per-class analysis with rankings"""
    
    print("\n" + "="*80)
    print("DETAILED PER-CLASS PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Find the latest analysis results
    analysis_dir = Path("analysis_results")
    latest_dir = max([d for d in analysis_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    results_file = latest_dir / "raw_analysis_results.json"
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    models = list(results.keys())
    metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
    class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
    
    # Create ranking for each class and metric
    rankings = {}
    
    for class_name in class_names:
        print(f"\n{class_name} Performance:")
        print("-" * 40)
        
        class_rankings = {}
        
        for metric in metrics:
            metric_name = metric.replace('_', '-').title()
            model_scores = []
            
            for model in models:
                if (metric in results[model] and 
                    class_name in results[model][metric]):
                    
                    mean_score = results[model][metric][class_name]['mean']
                    std_score = results[model][metric][class_name]['std']
                    model_scores.append((model.upper(), mean_score, std_score))
            
            # Sort by mean score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            class_rankings[metric] = model_scores
            
            print(f"  {metric_name}:")
            for rank, (model, mean_val, std_val) in enumerate(model_scores, 1):
                print(f"    {rank}. {model}: {mean_val:.4f} ± {std_val:.4f}")
        
        rankings[class_name] = class_rankings
    
    # Create summary ranking table
    ranking_summary = []
    
    for class_name in class_names:
        for metric in metrics:
            if metric in rankings[class_name]:
                for rank, (model, mean_val, std_val) in enumerate(rankings[class_name][metric], 1):
                    ranking_summary.append({
                        'Class': class_name,
                        'Metric': metric.replace('_', '-').title(),
                        'Rank': rank,
                        'Model': model,
                        'Score': f"{mean_val:.4f} ± {std_val:.4f}",
                        'Mean': mean_val,
                        'Std': std_val
                    })
    
    ranking_df = pd.DataFrame(ranking_summary)
    output_file = latest_dir / "detailed_rankings.csv"
    ranking_df.to_csv(output_file, index=False)
    print(f"\nDetailed rankings saved to: {output_file}")
    
    return ranking_df


def create_comprehensive_visualization():
    """Create comprehensive visualization with subplots"""
    
    # Load results
    analysis_dir = Path("analysis_results")
    latest_dir = max([d for d in analysis_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    results_file = latest_dir / "raw_analysis_results.json"
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    models = list(results.keys())
    metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
    metric_names = ['Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
    class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # Overall performance radar chart
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for model in models:
        values = []
        for metric in metrics:
            if metric in results[model] and 'Overall' in results[model][metric]:
                values.append(results[model][metric]['Overall']['mean'])
            else:
                values.append(0)
        
        values += values[:1]  # Complete the circle
        ax1.plot(angles, values, 'o-', linewidth=2, label=model.upper())
        ax1.fill(angles, values, alpha=0.25)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metric_names)
    ax1.set_ylim(0, 1)
    ax1.set_title('Overall Performance Radar Chart', size=12, pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Individual metric comparisons
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = plt.subplot(2, 3, idx + 2)
        
        # Collect overall data
        model_names = []
        means = []
        stds = []
        
        for model in models:
            if metric in results[model] and 'Overall' in results[model][metric]:
                model_names.append(model.upper())
                means.append(results[model][metric]['Overall']['mean'])
                stds.append(results[model][metric]['Overall']['std'])
        
        if model_names:
            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7, 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)])
            ax.set_title(f'{metric_name} - Overall')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.0)
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                       f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Model comparison heatmap
    ax6 = plt.subplot(2, 3, 6)
    
    # Create comparison matrix
    comparison_data = []
    for model in models:
        row = []
        for metric in metrics:
            if metric in results[model] and 'Overall' in results[model][metric]:
                row.append(results[model][metric]['Overall']['mean'])
            else:
                row.append(0)
        comparison_data.append(row)
    
    if comparison_data:
        im = ax6.imshow(comparison_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax6.set_xticks(range(len(metrics)))
        ax6.set_xticklabels(metric_names, rotation=45)
        ax6.set_yticks(range(len(models)))
        ax6.set_yticklabels([m.upper() for m in models])
        ax6.set_title('Performance Heatmap')
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metrics)):
                if comparison_data[i][j] > 0:
                    text = ax6.text(j, i, f'{comparison_data[i][j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    plt.savefig(latest_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive visualization saved to: {latest_dir / 'comprehensive_analysis.png'}")


def main():
    """Main function for enhanced analysis"""
    print("Enhanced 5-Fold Cross-Validation Analysis")
    print("=" * 60)
    
    # First, run the basic analysis if results don't exist
    analysis_dir = Path("analysis_results")
    if not analysis_dir.exists() or not list(analysis_dir.iterdir()):
        print("Running basic analysis first...")
        os.system("python analyze_5fold_results.py")
    
    # Run enhanced analyses
    try:
        statistical_df = create_statistical_comparison_table()
        ranking_df = create_detailed_per_class_analysis()
        create_comprehensive_visualization()
        
        print("\n" + "="*80)
        print("ENHANCED ANALYSIS COMPLETED")
        print("="*80)
        print("Generated files:")
        
        latest_dir = max([d for d in analysis_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        for file in sorted(latest_dir.iterdir()):
            if file.is_file():
                print(f"  - {file.name}")
        
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()