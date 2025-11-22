#!/usr/bin/env python3
"""
5-Fold Cross-Validation Results Analysis for AMD Grading Models

This script analyzes the cross-validation results for OCT, Bio, and 3D models,
calculating mean and standard deviation of various metrics across folds.

Features:
- Analyzes sensitivity (recall), specificity, F1-score, and AUC-ROC
- Calculates per-class and overall performance statistics
- Generates comparison tables and visualizations
- Saves comprehensive analysis results

Usage:
    python analyze_5fold_results.py
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
import warnings
warnings.filterwarnings('ignore')

class CrossValidationAnalyzer:
    """Analyzer for 5-fold cross-validation results"""
    
    def __init__(self, base_dir="logs"):
        self.base_dir = Path(base_dir)
        self.models = ['oct', 'bio', '3d']
        self.class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
        self.results = {}
        
    def find_latest_results(self):
        """Find the latest results for each model"""
        latest_results = {}
        
        for model in self.models:
            validation_dir = self.base_dir / f"5-k-validation_{model}"
            if validation_dir.exists():
                # Find the latest timestamp directory
                timestamp_dirs = [d for d in validation_dir.iterdir() if d.is_dir()]
                if timestamp_dirs:
                    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
                    latest_results[model] = latest_dir
                    print(f"Found {model.upper()} results: {latest_dir}")
                else:
                    print(f"No results found for {model.upper()} model")
            else:
                print(f"No validation directory found for {model.upper()} model")
        
        return latest_results
    
    def load_fold_metrics(self, model_dir):
        """Load metrics from all folds for a model"""
        fold_metrics = {}
        
        for fold_num in range(1, 6):  # folds 1-5
            fold_dir = model_dir / f"fold_{fold_num}"
            metrics_file = fold_dir / "validation_metrics.csv"
            
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                fold_metrics[fold_num] = df
            else:
                print(f"Warning: Missing metrics file for fold {fold_num}")
        
        return fold_metrics
    
    def load_fold_predictions(self, model_dir):
        """Load predictions from all folds for a model"""
        fold_predictions = {}
        
        for fold_num in range(1, 6):  # folds 1-5
            fold_dir = model_dir / f"fold_{fold_num}"
            predictions_file = fold_dir / "validation_results.json"
            
            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    data = json.load(f)
                fold_predictions[fold_num] = data
            else:
                print(f"Warning: Missing predictions file for fold {fold_num}")
        
        return fold_predictions
    
    def calculate_specificity_per_fold(self, fold_predictions):
        """Calculate specificity for each fold and class"""
        specificities = {}
        
        for fold_num, data in fold_predictions.items():
            y_true = np.array(data['y_true'])
            y_pred = np.array(data['y_pred'])
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=range(4))
            
            # Calculate specificity for each class
            fold_spec = []
            for class_idx in range(4):
                # True negatives: correctly predicted as not this class
                tn = np.sum(cm) - (np.sum(cm[class_idx, :]) + np.sum(cm[:, class_idx]) - cm[class_idx, class_idx])
                # False positives: incorrectly predicted as this class
                fp = np.sum(cm[:, class_idx]) - cm[class_idx, class_idx]
                
                if tn + fp > 0:
                    specificity = tn / (tn + fp)
                else:
                    specificity = 0.0
                
                fold_spec.append(specificity)
            
            specificities[fold_num] = fold_spec
        
        return specificities
    
    def extract_metrics_by_class(self, fold_metrics):
        """Extract metrics organized by class and metric type"""
        metrics_data = {
            'sensitivity': {},  # recall
            'f1_score': {},
            'auc_roc': {}
        }
        
        for fold_num, df in fold_metrics.items():
            # Initialize fold data
            for metric in metrics_data:
                metrics_data[metric][fold_num] = {}
            
            # Extract per-class metrics
            for class_name in self.class_names:
                # Sensitivity (Recall)
                recall_row = df[(df['Metric'] == 'Recall') & 
                               (df['Type'] == 'Per-Class') & 
                               (df['Class'] == class_name)]
                if not recall_row.empty:
                    metrics_data['sensitivity'][fold_num][class_name] = recall_row['Value'].iloc[0]
                
                # F1-Score
                f1_row = df[(df['Metric'] == 'F1-Score') & 
                           (df['Type'] == 'Per-Class') & 
                           (df['Class'] == class_name)]
                if not f1_row.empty:
                    metrics_data['f1_score'][fold_num][class_name] = f1_row['Value'].iloc[0]
                
                # AUC-ROC
                auc_row = df[(df['Metric'] == 'AUC-ROC') & 
                            (df['Type'] == 'Per-Class') & 
                            (df['Class'] == class_name)]
                if not auc_row.empty:
                    metrics_data['auc_roc'][fold_num][class_name] = auc_row['Value'].iloc[0]
            
            # Extract overall metrics
            for metric_name, csv_metric in [('sensitivity', 'Recall'), ('f1_score', 'F1-Score'), ('auc_roc', 'AUC-ROC')]:
                overall_row = df[(df['Metric'] == csv_metric) & 
                                (df['Type'] == 'Macro') & 
                                (df['Class'] == 'All')]
                if not overall_row.empty:
                    metrics_data[metric_name][fold_num]['Overall'] = overall_row['Value'].iloc[0]
        
        return metrics_data
    
    def calculate_statistics(self, metrics_data, specificities):
        """Calculate mean and standard deviation for all metrics"""
        stats = {}
        
        # Process sensitivity, f1_score, auc_roc
        for metric_name, fold_data in metrics_data.items():
            stats[metric_name] = {}
            
            # Get all classes including Overall
            all_classes = set()
            for fold_metrics in fold_data.values():
                all_classes.update(fold_metrics.keys())
            
            for class_name in all_classes:
                values = []
                for fold_num in range(1, 6):
                    if fold_num in fold_data and class_name in fold_data[fold_num]:
                        values.append(fold_data[fold_num][class_name])
                
                if values:
                    stats[metric_name][class_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                        'values': values
                    }
        
        # Process specificity
        stats['specificity'] = {}
        
        # Per-class specificity
        for class_idx, class_name in enumerate(self.class_names):
            values = []
            for fold_num in range(1, 6):
                if fold_num in specificities:
                    values.append(specificities[fold_num][class_idx])
            
            if values:
                stats['specificity'][class_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                    'values': values
                }
        
        # Overall specificity (macro average)
        if len(self.class_names) == 4:
            overall_spec_values = []
            for fold_num in range(1, 6):
                if fold_num in specificities:
                    fold_mean_spec = np.mean(specificities[fold_num])
                    overall_spec_values.append(fold_mean_spec)
            
            if overall_spec_values:
                stats['specificity']['Overall'] = {
                    'mean': np.mean(overall_spec_values),
                    'std': np.std(overall_spec_values, ddof=1) if len(overall_spec_values) > 1 else 0.0,
                    'values': overall_spec_values
                }
        
        return stats
    
    def analyze_model(self, model, model_dir):
        """Analyze results for a single model"""
        print(f"\nAnalyzing {model.upper()} model...")
        
        # Load fold metrics and predictions
        fold_metrics = self.load_fold_metrics(model_dir)
        fold_predictions = self.load_fold_predictions(model_dir)
        
        if not fold_metrics or not fold_predictions:
            print(f"Insufficient data for {model.upper()} model")
            return None
        
        print(f"Loaded data from {len(fold_metrics)} folds")
        
        # Calculate specificity
        specificities = self.calculate_specificity_per_fold(fold_predictions)
        
        # Extract metrics by class
        metrics_data = self.extract_metrics_by_class(fold_metrics)
        
        # Calculate statistics
        stats = self.calculate_statistics(metrics_data, specificities)
        
        return stats
    
    def create_comparison_table(self):
        """Create comparison table across all models"""
        comparison_data = []
        
        metrics_order = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
        metric_names = ['Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
        
        for model in self.models:
            if model in self.results:
                for metric_name, csv_metric in zip(metric_names, metrics_order):
                    if csv_metric in self.results[model]:
                        # Overall performance
                        if 'Overall' in self.results[model][csv_metric]:
                            stats = self.results[model][csv_metric]['Overall']
                            comparison_data.append({
                                'Model': model.upper(),
                                'Metric': metric_name,
                                'Class': 'Overall',
                                'Mean': stats['mean'],
                                'Std': stats['std'],
                                'Mean±Std': f"{stats['mean']:.4f}±{stats['std']:.4f}"
                            })
                        
                        # Per-class performance
                        for class_name in self.class_names:
                            if class_name in self.results[model][csv_metric]:
                                stats = self.results[model][csv_metric][class_name]
                                comparison_data.append({
                                    'Model': model.upper(),
                                    'Metric': metric_name,
                                    'Class': class_name,
                                    'Mean': stats['mean'],
                                    'Std': stats['std'],
                                    'Mean±Std': f"{stats['mean']:.4f}±{stats['std']:.4f}"
                                })
        
        return pd.DataFrame(comparison_data)
    
    def create_visualizations(self, output_dir):
        """Create visualizations of the results"""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create overall performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('5-Fold Cross-Validation Results Comparison', fontsize=16)
        
        metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
        metric_titles = ['Sensitivity (Recall)', 'Specificity', 'F1-Score', 'AUC-ROC']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Collect data for plotting
            model_names = []
            means = []
            stds = []
            
            for model in self.models:
                if model in self.results and metric in self.results[model]:
                    if 'Overall' in self.results[model][metric]:
                        model_names.append(model.upper())
                        stats = self.results[model][metric]['Overall']
                        means.append(stats['mean'])
                        stds.append(stats['std'])
            
            if model_names:
                bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_title(f'{title} - Overall Performance')
                ax.set_ylabel(title)
                ax.set_ylim(0, 1.0)
                
                # Add value labels on bars
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                           f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create per-class performance heatmap
        self.create_perclass_heatmap(output_dir)
    
    def create_perclass_heatmap(self, output_dir):
        """Create heatmap showing per-class performance across models"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Per-Class Performance Heatmap (Mean Values)', fontsize=16)
        
        metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
        metric_titles = ['Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Create data matrix
            data_matrix = []
            model_labels = []
            
            for model in self.models:
                if model in self.results and metric in self.results[model]:
                    row = []
                    for class_name in self.class_names:
                        if class_name in self.results[model][metric]:
                            row.append(self.results[model][metric][class_name]['mean'])
                        else:
                            row.append(0.0)
                    data_matrix.append(row)
                    model_labels.append(model.upper())
            
            if data_matrix:
                # Create heatmap
                sns.heatmap(data_matrix, 
                           xticklabels=self.class_names,
                           yticklabels=model_labels,
                           annot=True, 
                           fmt='.3f', 
                           cmap='RdYlBu_r',
                           vmin=0, vmax=1,
                           ax=ax)
                ax.set_title(f'{title} - Per Class')
                ax.set_xlabel('AMD Classes')
                ax.set_ylabel('Models')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'perclass_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_detailed_results(self, output_dir):
        """Save detailed results to files"""
        # Create comparison table
        comparison_df = self.create_comparison_table()
        
        # Save to CSV
        comparison_df.to_csv(output_dir / 'detailed_comparison.csv', index=False)
        
        # Create formatted summary tables
        with open(output_dir / 'analysis_summary.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("5-FOLD CROSS-VALIDATION ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models analyzed: {', '.join([m.upper() for m in self.models if m in self.results])}\n\n")
            
            # Overall performance summary
            f.write("OVERALL PERFORMANCE COMPARISON\n")
            f.write("-" * 50 + "\n")
            
            overall_df = comparison_df[comparison_df['Class'] == 'Overall']
            pivot_df = overall_df.pivot(index='Model', columns='Metric', values='Mean±Std')
            f.write(pivot_df.to_string())
            f.write("\n\n")
            
            # Per-class detailed results
            f.write("PER-CLASS PERFORMANCE DETAILS\n")
            f.write("-" * 50 + "\n")
            
            for class_name in self.class_names:
                f.write(f"\n{class_name}:\n")
                class_df = comparison_df[comparison_df['Class'] == class_name]
                if not class_df.empty:
                    class_pivot = class_df.pivot(index='Model', columns='Metric', values='Mean±Std')
                    f.write(class_pivot.to_string())
                f.write("\n")
        
        # Save raw results as JSON
        results_copy = {}
        for model, model_results in self.results.items():
            results_copy[model] = {}
            for metric, metric_results in model_results.items():
                results_copy[model][metric] = {}
                for class_name, stats in metric_results.items():
                    results_copy[model][metric][class_name] = {
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'values': [float(v) for v in stats['values']]
                    }
        
        with open(output_dir / 'raw_analysis_results.json', 'w') as f:
            json.dump(results_copy, f, indent=2)
    
    def run_analysis(self):
        """Run complete analysis"""
        print("Starting 5-fold cross-validation analysis...")
        print("=" * 60)
        
        # Find latest results
        latest_results = self.find_latest_results()
        
        if not latest_results:
            print("No validation results found!")
            return
        
        # Analyze each model
        for model, model_dir in latest_results.items():
            result = self.analyze_model(model, model_dir)
            if result:
                self.results[model] = result
        
        if not self.results:
            print("No valid results to analyze!")
            return
        
        # Create output directory
        output_dir = Path("analysis_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving analysis results to: {output_dir}")
        
        # Save detailed results
        self.save_detailed_results(output_dir)
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Print quick summary
        self.print_quick_summary()
    
    def print_quick_summary(self):
        """Print a quick summary of the results"""
        print("\n" + "=" * 60)
        print("QUICK SUMMARY")
        print("=" * 60)
        
        for model in self.models:
            if model in self.results:
                print(f"\n{model.upper()} Model:")
                print("-" * 20)
                
                for metric in ['sensitivity', 'specificity', 'f1_score', 'auc_roc']:
                    if metric in self.results[model] and 'Overall' in self.results[model][metric]:
                        stats = self.results[model][metric]['Overall']
                        metric_name = metric.replace('_', '-').title()
                        print(f"{metric_name:12}: {stats['mean']:.4f} ± {stats['std']:.4f}")


def main():
    """Main function"""
    analyzer = CrossValidationAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()