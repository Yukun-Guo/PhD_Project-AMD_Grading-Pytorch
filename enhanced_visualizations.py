#!/usr/bin/env python3
"""
Enhanced Visualizations for 5-Fold Cross-Validation Results

This script creates enhanced visualizations with statistical significance indicators
and comprehensive per-class performance comparisons.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class EnhancedVisualizer:
    def __init__(self):
        self.class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
        self.metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
        self.metric_names = ['Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
        
    def load_results(self):
        """Load the latest analysis results"""
        analysis_dir = Path("analysis_results")
        if not analysis_dir.exists():
            raise FileNotFoundError("No analysis results found.")
        
        latest_dir = max([d for d in analysis_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        results_file = latest_dir / "raw_analysis_results.json"
        
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.models = list(self.results.keys())
        self.output_dir = latest_dir
        
        print(f"Loaded results for models: {self.models}")
        print(f"Output directory: {self.output_dir}")
    
    def calculate_pvalue(self, model1, model2, metric, class_name='Overall'):
        """Calculate p-value between two models for a specific metric and class"""
        try:
            if (metric in self.results[model1] and class_name in self.results[model1][metric] and
                metric in self.results[model2] and class_name in self.results[model2][metric]):
                
                values1 = self.results[model1][metric][class_name]['values']
                values2 = self.results[model2][metric][class_name]['values']
                
                if len(values1) > 1 and len(values2) > 1:
                    _, p_value = stats.ttest_rel(values1, values2)
                    return p_value
        except Exception as e:
            print(f"Error calculating p-value for {model1} vs {model2}, {metric}, {class_name}: {e}")
        
        return None
    
    def get_significance_symbol(self, p_value):
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
            return ""
    
    def add_significance_bars(self, ax, x_positions, heights, model_pairs, metric, class_name='Overall'):
        """Add significance bars with p-values above bar plots"""
        y_max = max(heights) if heights else 1.0
        bar_height = 0.05 * y_max
        text_height = 0.03 * y_max
        
        for i, (model1, model2) in enumerate(model_pairs):
            p_value = self.calculate_pvalue(model1, model2, metric, class_name)
            if p_value is not None:
                # Determine positions
                idx1 = next((j for j, model in enumerate(self.models) if model == model1), None)
                idx2 = next((j for j, model in enumerate(self.models) if model == model2), None)
                
                if idx1 is not None and idx2 is not None and idx1 < len(x_positions) and idx2 < len(x_positions):
                    x1, x2 = x_positions[idx1], x_positions[idx2]
                    y = y_max + (i + 1) * (bar_height + text_height + 0.02)
                    
                    # Draw horizontal line
                    ax.plot([x1, x2], [y, y], 'k-', linewidth=1)
                    # Draw vertical lines
                    ax.plot([x1, x1], [y - bar_height/2, y + bar_height/2], 'k-', linewidth=1)
                    ax.plot([x2, x2], [y - bar_height/2, y + bar_height/2], 'k-', linewidth=1)
                    
                    # Add p-value text
                    significance = self.get_significance_symbol(p_value)
                    p_text = f"p={p_value:.3f}{significance}" if significance != "ns" else f"p={p_value:.3f}"
                    ax.text((x1 + x2) / 2, y + text_height, p_text, 
                           ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    def create_enhanced_overall_comparison(self):
        """Create enhanced overall performance comparison with p-values"""
        title_font_size = 40
        label_font_size = 38
        value_font_size = 38
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('5-Fold Cross-Validation Results - Overall Performance Comparison\nwith Statistical Significance', 
                     fontsize=title_font_size, fontweight='bold')
        
        # Generate all model pairs for comparison
        model_pairs = [(self.models[i], self.models[j]) 
                      for i in range(len(self.models)) 
                      for j in range(i+1, len(self.models))]
        
        for idx, (metric, title) in enumerate(zip(self.metrics, self.metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Collect data for plotting
            model_names = []
            means = []
            stds = []
            
            for model in self.models:
                if model in self.results and metric in self.results[model]:
                    if 'Overall' in self.results[model][metric]:
                        model_names.append(model.upper())
                        stats_data = self.results[model][metric]['Overall']
                        means.append(stats_data['mean'])
                        stds.append(stats_data['std'])
            
            if model_names:
                # Create bars with different colors
                colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(model_names)))
                bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.8, color=colors)
                
                ax.set_title(f'{title}', fontsize=title_font_size, fontweight='bold')
                ax.set_ylabel(f'{title} Score', fontsize=label_font_size)
                ax.set_ylim(0, min(1.2, max(means) + max(stds) + 0.3))
                
                # Add value labels on bars
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                           f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', 
                           fontsize=value_font_size, fontweight='bold')
                
                # Add significance bars
                x_positions = [bar.get_x() + bar.get_width()/2 for bar in bars]
                heights_with_error = [m + s for m, s in zip(means, stds)]
                self.add_significance_bars(ax, x_positions, heights_with_error, 
                                         model_pairs, metric, 'Overall')
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enhanced_overall_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Enhanced overall comparison saved to: {self.output_dir / 'enhanced_overall_performance_comparison.png'}")
        plt.close()
    
    def create_perclass_comparison(self):
        """Create comprehensive per-class performance comparison"""
        n_classes = len(self.class_names)
        n_metrics = len(self.metrics)
        
        fig, axes = plt.subplots(n_classes, n_metrics, figsize=(20, 16))
        fig.suptitle('Per-Class Performance Comparison with Statistical Significance', 
                     fontsize=18, fontweight='bold')
        
        model_pairs = [(self.models[i], self.models[j]) 
                      for i in range(len(self.models)) 
                      for j in range(i+1, len(self.models))]
        
        for class_idx, class_name in enumerate(self.class_names):
            for metric_idx, (metric, metric_name) in enumerate(zip(self.metrics, self.metric_names)):
                ax = axes[class_idx, metric_idx]
                
                # Collect data for this class and metric
                model_names = []
                means = []
                stds = []
                
                for model in self.models:
                    if (model in self.results and 
                        metric in self.results[model] and 
                        class_name in self.results[model][metric]):
                        
                        model_names.append(model.upper())
                        stats_data = self.results[model][metric][class_name]
                        means.append(stats_data['mean'])
                        stds.append(stats_data['std'])
                
                if model_names and means:
                    # Create bars
                    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(model_names)))
                    bars = ax.bar(model_names, means, yerr=stds, capsize=3, alpha=0.8, color=colors)
                    
                    # Set title and labels
                    if class_idx == 0:
                        ax.set_title(f'{metric_name}', fontsize=18, fontweight='bold')
                    if metric_idx == 0:
                        ax.set_ylabel(f'{class_name}', fontsize=18, fontweight='bold', rotation=0, ha='right')
                    
                    ax.set_ylim(0, min(1.2, max(means) + max(stds) + 0.25))
                    
                    # Add value labels
                    for bar, mean, std in zip(bars, means, stds):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                               f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', 
                               fontsize=16, fontweight='bold')
                    
                    # Add significance bars
                    x_positions = [bar.get_x() + bar.get_width()/2 for bar in bars]
                    heights_with_error = [m + s for m, s in zip(means, stds)]
                    self.add_significance_bars(ax, x_positions, heights_with_error, 
                                             model_pairs, metric, class_name)
                    
                    # Rotate x-axis labels for better readability
                    ax.tick_params(axis='x', rotation=45, labelsize=10)
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=18)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'perclass_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Per-class comparison saved to: {self.output_dir / 'perclass_performance_comparison.png'}")
        plt.close()
    
    def create_significance_matrix(self):
        """Create a matrix showing all pairwise comparisons"""
        if len(self.models) < 2:
            print("Need at least 2 models for significance matrix")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Significance Matrix (p-values)', fontsize=20, fontweight='bold')
        
        model_pairs = [(self.models[i], self.models[j]) 
                      for i in range(len(self.models)) 
                      for j in range(i+1, len(self.models))]
        
        for idx, (metric, metric_name) in enumerate(zip(self.metrics, self.metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Create matrix for overall + per-class comparisons
            categories = ['Overall'] + self.class_names
            n_categories = len(categories)
            n_pairs = len(model_pairs)
            
            # Create data matrix
            matrix_data = np.full((n_pairs, n_categories), np.nan)
            pair_labels = []
            
            for pair_idx, (model1, model2) in enumerate(model_pairs):
                pair_labels.append(f"{model1.upper()}\nvs\n{model2.upper()}")
                
                for cat_idx, category in enumerate(categories):
                    p_value = self.calculate_pvalue(model1, model2, metric, category)
                    if p_value is not None:
                        matrix_data[pair_idx, cat_idx] = -np.log10(p_value)  # Use -log10 for better visualization
            
            # Create heatmap
            im = ax.imshow(matrix_data, cmap='RdYlBu_r', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(n_categories))
            ax.set_xticklabels(categories, rotation=45, ha='right')
            ax.set_yticks(range(n_pairs))
            ax.set_yticklabels(pair_labels, fontsize=16)
            ax.set_title(f'{metric_name} - Significance Matrix\n(-log10 p-value)', fontsize=18)
            
            # Add text annotations
            for i in range(n_pairs):
                for j in range(n_categories):
                    if not np.isnan(matrix_data[i, j]):
                        # Convert back to p-value for display
                        p_val = 10**(-matrix_data[i, j])
                        significance = self.get_significance_symbol(p_val)
                        text_color = 'white' if matrix_data[i, j] > 1.3 else 'black'  # -log10(0.05) ≈ 1.3
                        ax.text(j, i, f'{p_val:.3f}\n{significance}', 
                               ha="center", va="center", color=text_color, fontsize=16)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('-log10(p-value)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'significance_matrix.png', dpi=300, bbox_inches='tight')
        print(f"Significance matrix saved to: {self.output_dir / 'significance_matrix.png'}")
        plt.close()
    
    def create_model_ranking_chart(self):
        """Create a ranking chart showing which model performs best for each metric/class"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Collect all combinations
        combinations = []
        for metric_idx, metric in enumerate(self.metrics):
            for class_name in ['Overall'] + self.class_names:
                combinations.append((metric, class_name, self.metric_names[metric_idx]))
        
        y_positions = range(len(combinations))
        model_positions = {model: i for i, model in enumerate(self.models)}
        
        # For each combination, find the best performing model
        for y, (metric, class_name, metric_name) in enumerate(combinations):
            model_scores = []
            
            for model in self.models:
                if (model in self.results and 
                    metric in self.results[model] and 
                    class_name in self.results[model][metric]):
                    score = self.results[model][metric][class_name]['mean']
                    model_scores.append((model, score))
            
            if model_scores:
                # Sort by score (descending)
                model_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Plot bars for each model
                for rank, (model, score) in enumerate(model_scores):
                    color_intensity = 1.0 - (rank * 0.3)  # Best model gets full color
                    color = plt.cm.get_cmap('viridis')(color_intensity)
                    
                    ax.barh(y, score, height=0.8, left=0, alpha=0.8, 
                           color=color, label=model.upper() if y == 0 else "")
                    
                    # Add score text
                    ax.text(score + 0.01, y, f'{model.upper()}: {score:.3f}', 
                           va='center', fontsize=9, fontweight='bold')
        
        # Customize the plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{metric_name}\n{class_name}" for metric, class_name, metric_name in combinations])
        ax.set_xlabel('Performance Score', fontsize=18)
        ax.set_title('Model Performance Ranking by Metric and Class', fontsize=20, fontweight='bold')
        ax.set_xlim(0, 1.2)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_ranking_chart.png', dpi=300, bbox_inches='tight')
        print(f"Model ranking chart saved to: {self.output_dir / 'model_ranking_chart.png'}")
        plt.close()
    
    def run_all_visualizations(self):
        """Run all visualization methods"""
        try:
            self.load_results()
            
            print("Creating enhanced overall comparison...")
            self.create_enhanced_overall_comparison()
            
            print("Creating per-class comparison...")
            self.create_perclass_comparison()
            
            print("Creating significance matrix...")
            self.create_significance_matrix()
            
            print("Creating model ranking chart...")
            self.create_model_ranking_chart()
            
            print("\nAll enhanced visualizations completed!")
            print(f"Results saved in: {self.output_dir}")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    visualizer = EnhancedVisualizer()
    visualizer.run_all_visualizations()