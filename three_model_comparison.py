#!/usr/bin/env python3
"""
Comprehensive Three-Model Comparison for 5-Fold Cross-Validation
Compares OCT, BIO, and 3D models with statistical significance testing
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from itertools import combinations

class ThreeModelComparison:
    def __init__(self):
        self.class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
        self.metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
        self.metric_names = ['Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']
        self.model_colors = {'oct': '#1f77b4', 'bio': '#ff7f0e', '3d': '#2ca02c'}
        
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
        
        print(f"✅ Loaded results for models: {[m.upper() for m in self.models]}")
        print(f"📁 Output directory: {self.output_dir}\n")
    
    def calculate_pvalue(self, model1, model2, metric, class_name='Overall'):
        """Calculate p-value between two models"""
        try:
            if (metric in self.results[model1] and class_name in self.results[model1][metric] and
                metric in self.results[model2] and class_name in self.results[model2][metric]):
                
                values1 = self.results[model1][metric][class_name]['values']
                values2 = self.results[model2][metric][class_name]['values']
                
                if len(values1) > 1 and len(values2) > 1:
                    _, p_value = stats.ttest_rel(values1, values2)
                    return p_value
        except:
            pass
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
            return "ns"
    
    def create_overall_comparison_enhanced(self):
        """Create enhanced overall comparison with all 3 models and p-values"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('5-Fold Cross-Validation: Overall Performance Comparison\nOCT vs BIO vs 3D Models with Statistical Significance', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Get model pairs for comparison
        model_pairs = list(combinations(self.models, 2))
        
        for idx, (metric, title) in enumerate(zip(self.metrics, self.metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Collect data
            model_names = []
            means = []
            stds = []
            colors = []
            
            for model in self.models:
                if model in self.results and metric in self.results[model] and 'Overall' in self.results[model][metric]:
                    model_names.append(model.upper())
                    stats_data = self.results[model][metric]['Overall']
                    means.append(stats_data['mean'])
                    stds.append(stats_data['std'])
                    colors.append(self.model_colors.get(model, '#333333'))
            
            if len(model_names) >= 2:
                # Create bars
                x_pos = np.arange(len(model_names))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=8, alpha=0.85, color=colors, edgecolor='black', linewidth=2)
                
                ax.set_title(f'{title}', fontsize=13, fontweight='bold', pad=10)
                ax.set_ylabel(f'{title} Score', fontsize=12, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(model_names, fontsize=11, fontweight='bold')
                
                # Add value labels on bars
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                           f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold')
                
                # Add significance bars
                y_max = max([m + s for m, s in zip(means, stds)])
                y_offset = y_max + 0.08
                
                for i, (model1, model2) in enumerate(model_pairs):
                    if model1 in self.models and model2 in self.models:
                        idx1 = self.models.index(model1)
                        idx2 = self.models.index(model2)
                        
                        if idx1 < len(x_pos) and idx2 < len(x_pos):
                            p_value = self.calculate_pvalue(model1, model2, metric)
                            
                            x1, x2 = x_pos[idx1], x_pos[idx2]
                            y = y_offset + (i * 0.06)
                            
                            # Draw significance line
                            ax.plot([x1, x2], [y, y], 'k-', linewidth=1.5)
                            ax.plot([x1, x1], [y - 0.02, y + 0.02], 'k-', linewidth=1.5)
                            ax.plot([x2, x2], [y - 0.02, y + 0.02], 'k-', linewidth=1.5)
                            
                            # Add p-value text
                            if p_value is not None:
                                significance = self.get_significance_symbol(p_value)
                                p_text = f"p={p_value:.3f} {significance}"
                                ax.text((x1 + x2) / 2, y + 0.015, p_text, 
                                       ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax.set_ylim(0, min(1.15, y_offset + 0.15))
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'three_model_overall_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Three-model overall comparison saved")
        plt.close()
    
    def create_perclass_heatmap_comparison(self):
        """Create comprehensive per-class heatmap comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Per-Class Performance Comparison: OCT vs BIO vs 3D', 
                     fontsize=16, fontweight='bold')
        
        for metric_idx, (metric, metric_name) in enumerate(zip(self.metrics, self.metric_names)):
            ax = axes[metric_idx // 2, metric_idx % 2]
            
            # Create data matrix: rows = classes, columns = models
            data_matrix = []
            
            for class_name in self.class_names:
                row = []
                for model in self.models:
                    if (model in self.results and metric in self.results[model] and 
                        class_name in self.results[model][metric]):
                        value = self.results[model][metric][class_name]['mean']
                        row.append(value)
                    else:
                        row.append(0)
                data_matrix.append(row)
            
            # Create heatmap
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(len(self.models)))
            ax.set_xticklabels([m.upper() for m in self.models], fontweight='bold', fontsize=11)
            ax.set_yticks(range(len(self.class_names)))
            ax.set_yticklabels(self.class_names, fontsize=11)
            ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            
            # Add text annotations
            for i in range(len(self.class_names)):
                for j in range(len(self.models)):
                    if data_matrix[i][j] > 0:
                        text_color = 'white' if data_matrix[i][j] < 0.5 else 'black'
                        ax.text(j, i, f'{data_matrix[i][j]:.3f}',
                               ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Performance Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'three_model_perclass_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✅ Per-class heatmap comparison saved")
        plt.close()
    
    def create_radar_comparison(self):
        """Create radar chart comparison of all three models"""
        fig = plt.figure(figsize=(16, 5))
        
        # Overall performance radar
        ax1 = plt.subplot(1, 3, 1, projection='polar')
        
        angles = np.linspace(0, 2 * np.pi, len(self.metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for model in self.models:
            values = []
            for metric in self.metrics:
                if metric in self.results[model] and 'Overall' in self.results[model][metric]:
                    values.append(self.results[model][metric]['Overall']['mean'])
                else:
                    values.append(0)
            values += values[:1]
            
            ax1.plot(angles, values, 'o-', linewidth=2.5, label=model.upper(), markersize=6)
            ax1.fill(angles, values, alpha=0.15)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels([m.replace('-', '\n') for m in self.metric_names], fontsize=10, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.set_title('Overall Performance Radar', fontsize=12, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.95)
        ax1.grid(True)
        
        # Per-class radar charts
        for class_idx, class_name in enumerate(self.class_names[:2]):  # Show first 2 classes in detail
            ax = plt.subplot(1, 3, class_idx + 2, projection='polar')
            
            for model in self.models:
                values = []
                for metric in self.metrics:
                    if (metric in self.results[model] and 
                        class_name in self.results[model][metric]):
                        values.append(self.results[model][metric][class_name]['mean'])
                    else:
                        values.append(0)
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2.5, label=model.upper(), markersize=6)
                ax.fill(angles, values, alpha=0.15)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('-', '\n') for m in self.metric_names], fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_title(f'{class_name} Performance', fontsize=12, fontweight='bold', pad=20)
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'three_model_radar_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Radar chart comparison saved")
        plt.close()
    
    def create_statistical_summary_table(self):
        """Create comprehensive statistical summary table"""
        print("\n" + "="*100)
        print("COMPREHENSIVE THREE-MODEL STATISTICAL COMPARISON")
        print("="*100)
        
        # Overall comparison with p-values
        print("\n📊 OVERALL PERFORMANCE WITH P-VALUES:")
        print("-" * 100)
        print(f"{'Metric':<15} {'OCT':<20} {'BIO':<20} {'3D':<20} {'Best Model':<15}")
        print("-" * 100)
        
        model_scores = {'oct': 0, 'bio': 0, '3d': 0}
        
        for metric, metric_name in zip(self.metrics, self.metric_names):
            scores = {}
            for model in self.models:
                if metric in self.results[model] and 'Overall' in self.results[model][metric]:
                    stats_data = self.results[model][metric]['Overall']
                    scores[model] = stats_data['mean']
            
            best_model = max(scores.items(), key=lambda x: x[1])[0]
            model_scores[best_model] += 1
            
            oct_str = f"{scores.get('oct', 0):.4f}±{self.results['oct'][metric]['Overall']['std']:.4f}" if 'oct' in scores else "N/A"
            bio_str = f"{scores.get('bio', 0):.4f}±{self.results['bio'][metric]['Overall']['std']:.4f}" if 'bio' in scores else "N/A"
            d3_str = f"{scores.get('3d', 0):.4f}±{self.results['3d'][metric]['Overall']['std']:.4f}" if '3d' in scores else "N/A"
            
            print(f"{metric_name:<15} {oct_str:<20} {bio_str:<20} {d3_str:<20} {best_model.upper():<15}")
        
        print("-" * 100)
        print(f"{'Overall Leader':<15} {model_scores['oct']}/4 metrics{'':<10} {model_scores['bio']}/4 metrics{'':<10} {model_scores['3d']}/4 metrics")
        
        # Per-class detailed analysis
        print("\n\n🎯 PER-CLASS DETAILED ANALYSIS:")
        print("-" * 100)
        
        for class_name in self.class_names:
            print(f"\n{class_name}:")
            class_scores = {'oct': 0, 'bio': 0, '3d': 0}
            
            for metric, metric_name in zip(self.metrics, self.metric_names):
                scores = {}
                for model in self.models:
                    if (metric in self.results[model] and 
                        class_name in self.results[model][metric]):
                        value = self.results[model][metric][class_name]['mean']
                        scores[model] = value
                
                if scores:
                    best_model = max(scores.items(), key=lambda x: x[1])[0]
                    class_scores[best_model] += 1
                    
                    scores_str = ", ".join([f"{model.upper()}: {scores[model]:.4f}" for model in self.models if model in scores])
                    print(f"  {metric_name:<15}: {scores_str} → Best: {best_model.upper()}")
            
            best_overall = max(class_scores.items(), key=lambda x: x[1])
            print(f"  → {best_overall[0].upper()} leads in {best_overall[1]}/4 metrics")
        
        print("\n" + "="*100)
        
        # Statistical significance summary
        print("\n📈 STATISTICAL SIGNIFICANCE TESTS (Paired t-test):")
        print("-" * 100)
        
        for metric, metric_name in zip(self.metrics, self.metric_names):
            print(f"\n{metric_name}:")
            print(f"  Overall Performance:")
            
            for model1, model2 in combinations(self.models, 2):
                p_value = self.calculate_pvalue(model1, model2, metric)
                if p_value is not None:
                    significance = self.get_significance_symbol(p_value)
                    mean1 = self.results[model1][metric]['Overall']['mean']
                    mean2 = self.results[model2][metric]['Overall']['mean']
                    better = model1.upper() if mean1 > mean2 else model2.upper()
                    print(f"    {model1.upper()} vs {model2.upper()}: p={p_value:.4f} {significance} ({better} better)")
    
    def run_all_comparisons(self):
        """Run all comparison analyses"""
        try:
            self.load_results()
            
            print("📊 Creating overall performance comparison...")
            self.create_overall_comparison_enhanced()
            
            print("🗺️  Creating per-class heatmap comparison...")
            self.create_perclass_heatmap_comparison()
            
            print("🔷 Creating radar chart comparison...")
            self.create_radar_comparison()
            
            print("📋 Creating statistical summary...")
            self.create_statistical_summary_table()
            
            print("\n" + "="*100)
            print("✅ ALL THREE-MODEL COMPARISONS COMPLETED!")
            print("="*100)
            print(f"\n📁 Results saved in: {self.output_dir}")
            print("\nGenerated files:")
            print("  • three_model_overall_comparison.png")
            print("  • three_model_perclass_heatmap.png")
            print("  • three_model_radar_comparison.png")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    comparator = ThreeModelComparison()
    comparator.run_all_comparisons()