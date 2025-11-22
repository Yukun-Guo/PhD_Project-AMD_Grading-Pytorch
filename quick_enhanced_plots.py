#!/usr/bin/env python3
"""
Quick Enhanced Visualizations - Focus on Key Plots with P-values
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

def load_results():
    """Load the latest analysis results"""
    analysis_dir = Path("analysis_results")
    latest_dir = max([d for d in analysis_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    results_file = latest_dir / "raw_analysis_results.json"
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results, latest_dir

def calculate_pvalue(results, model1, model2, metric, class_name='Overall'):
    """Calculate p-value between two models"""
    try:
        if (metric in results[model1] and class_name in results[model1][metric] and
            metric in results[model2] and class_name in results[model2][metric]):
            
            values1 = results[model1][metric][class_name]['values']
            values2 = results[model2][metric][class_name]['values']
            
            if len(values1) > 1 and len(values2) > 1:
                _, p_value = stats.ttest_rel(values1, values2)
                return p_value
    except:
        pass
    return None

def get_significance_symbol(p_value):
    """Convert p-value to significance symbol"""
    if p_value is None:
        return ""
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

def create_enhanced_overall_with_pvalues():
    """Create the enhanced overall comparison with p-values"""
    results, output_dir = load_results()
    models = list(results.keys())
    metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
    metric_names = ['Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('5-Fold Cross-Validation - Enhanced Overall Performance\nwith Statistical Significance', 
                 fontsize=16, fontweight='bold')
    
    for idx, (metric, title) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        # Collect data
        model_names = []
        means = []
        stds = []
        
        for model in models:
            if model in results and metric in results[model] and 'Overall' in results[model][metric]:
                model_names.append(model.upper())
                stats_data = results[model][metric]['Overall']
                means.append(stats_data['mean'])
                stds.append(stats_data['std'])
        
        if len(model_names) >= 2:
            # Create bars
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_names)]
            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.8, color=colors)
            
            ax.set_title(f'{title}', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{title} Score', fontsize=12)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
            
            # Add p-value comparison (for 2 models)
            if len(models) == 2:
                p_value = calculate_pvalue(results, models[0], models[1], metric)
                if p_value is not None:
                    y_max = max([m + s for m, s in zip(means, stds)])
                    y_line = y_max + 0.1
                    
                    # Draw significance line
                    x1 = bars[0].get_x() + bars[0].get_width()/2
                    x2 = bars[1].get_x() + bars[1].get_width()/2
                    
                    ax.plot([x1, x2], [y_line, y_line], 'k-', linewidth=2)
                    ax.plot([x1, x1], [y_line-0.02, y_line+0.02], 'k-', linewidth=2)
                    ax.plot([x2, x2], [y_line-0.02, y_line+0.02], 'k-', linewidth=2)
                    
                    # Add p-value text
                    significance = get_significance_symbol(p_value)
                    p_text = f"p={p_value:.3f} {significance}"
                    ax.text((x1 + x2) / 2, y_line + 0.03, p_text, 
                           ha='center', va='bottom', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            ax.set_ylim(0, max([m + s for m, s in zip(means, stds)]) + 0.2)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_overall_with_pvalues.png', dpi=300, bbox_inches='tight')
    print(f"Enhanced overall comparison with p-values saved to: {output_dir / 'enhanced_overall_with_pvalues.png'}")
    plt.close()

def create_simple_perclass_comparison():
    """Create a simplified per-class comparison"""
    results, output_dir = load_results()
    models = list(results.keys())
    metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
    metric_names = ['Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
    class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle('Per-Class Performance Comparison with Statistical Significance', 
                 fontsize=18, fontweight='bold')
    
    for class_idx, class_name in enumerate(class_names):
        for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[class_idx, metric_idx]
            
            # Collect data
            model_names = []
            means = []
            stds = []
            
            for model in models:
                if (model in results and metric in results[model] and 
                    class_name in results[model][metric]):
                    model_names.append(model.upper())
                    stats_data = results[model][metric][class_name]
                    means.append(stats_data['mean'])
                    stds.append(stats_data['std'])
            
            if model_names and len(model_names) >= 2:
                # Create bars
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_names)]
                bars = ax.bar(model_names, means, yerr=stds, capsize=3, alpha=0.8, color=colors)
                
                # Set titles
                if class_idx == 0:
                    ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
                if metric_idx == 0:
                    ax.set_ylabel(f'{class_name}', fontsize=12, fontweight='bold')
                
                # Add value labels
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
                
                # Add p-value for 2 models
                if len(models) == 2:
                    p_value = calculate_pvalue(results, models[0], models[1], metric, class_name)
                    if p_value is not None:
                        y_max = max([m + s for m, s in zip(means, stds)])
                        significance = get_significance_symbol(p_value)
                        
                        # Add simple p-value text
                        ax.text(0.5, 0.95, f"p={p_value:.3f} {significance}", 
                               transform=ax.transAxes, ha='center', va='top',
                               fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
                
                ax.set_ylim(0, min(1.1, max([m + s for m, s in zip(means, stds)]) + 0.15))
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'simple_perclass_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Per-class comparison saved to: {output_dir / 'simple_perclass_comparison.png'}")
    plt.close()

if __name__ == "__main__":
    print("Creating enhanced visualizations with p-values...")
    
    try:
        create_enhanced_overall_with_pvalues()
        create_simple_perclass_comparison()
        print("\nAll visualizations completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()