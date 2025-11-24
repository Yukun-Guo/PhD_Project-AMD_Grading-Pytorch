#!/usr/bin/env python3
"""
Enhanced Visualizations for Three-Model Comparison
- 2 significant digits for all values
- Larger fonts for clarity
- More spacious layout
- Professional appearance
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from itertools import combinations

class EnhancedThreeModelVisualizer:
    def __init__(self):
        self.class_names = ['Normal', 'Early AMD', 'Intermediate AMD', 'Advanced AMD']
        self.metrics = ['sensitivity', 'specificity', 'f1_score', 'auc_roc']
        self.metric_names = ['Sensitivity', 'Specificity', 'F1-Score', 'AROC']
        # Model order: BIO -> OCT -> 3D with display names
        self.model_order = ['bio', 'oct', '3d']
        self.model_display_names = {
            'bio': 'Biomarker Model',
            'oct': '2D OCT/OCTA Model',
            '3d': '3D OCT/OCTA Model'
        }
        self.model_colors = {'oct': '#1f77b4', 'bio': '#ff7f0e', '3d': '#2ca02c'}
        
        # Set up matplotlib for better quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 18
        plt.rcParams['font.weight'] = 'bold'
        
    def load_results(self):
        """Load the latest analysis results"""
        analysis_dir = Path("analysis_results")
        if not analysis_dir.exists():
            raise FileNotFoundError("No analysis results found.")
        
        latest_dir = max([d for d in analysis_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
        results_file = latest_dir / "raw_analysis_results.json"
        
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Use fixed model order
        self.models = [m for m in self.model_order if m in self.results]
        self.output_dir = latest_dir
        
        print(f"✅ Loaded results for models: {[self.model_display_names.get(m, m.upper()) for m in self.models]}")
        print(f"📁 Output directory: {self.output_dir}\n")
    
    def format_value(self, mean, std=None):
        """Format value to 2 significant digits"""
        if std is None:
            return f"{mean:.2f}"
        else:
            return f"{mean:.2f}±{std:.2f}"
    
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
            return ""
    
    def create_overall_comparison_enhanced(self):
        """Create enhanced overall comparison with better formatting"""
        # Optimized figure size: 20x24 (faster than 20x28, still plenty of room)
        fig, axes = plt.subplots(2, 2, figsize=(20, 24))
        fig.suptitle('5-Fold Cross-Validation: Overall Performance Comparison\nOCT vs BIO vs 3D Models', 
                     fontsize=5+20, fontweight='bold', y=0.995)
        
        # Get model pairs for comparison
        model_pairs = list(combinations(self.models, 2))
        
        for idx, (metric, title) in enumerate(zip(self.metrics, self.metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            # Collect data in model order
            model_names = []
            means = []
            stds = []
            colors = []
            
            for model in self.models:
                if model in self.results and metric in self.results[model] and 'Overall' in self.results[model][metric]:
                    model_names.append(self.model_display_names.get(model, model.upper()))
                    stats_data = self.results[model][metric]['Overall']
                    means.append(stats_data['mean'])
                    stds.append(stats_data['std'])
                    colors.append(self.model_colors.get(model, '#333333'))
            
            if len(model_names) >= 2:
                # Create bars with more space
                x_pos = np.arange(len(model_names))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=12, alpha=0.85, color=colors, 
                             edgecolor='black', linewidth=2.5, width=0.6)
                
                ax.set_title(f'{title}', fontsize=5+18, fontweight='bold', pad=15)
                ax.set_ylabel(f'{title} Score', fontsize=5+14, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(model_names, fontsize=5+13, fontweight='bold', rotation=0)
                ax.tick_params(axis='y', labelsize=14)
                
                # Add value labels on bars with 2 significant digits (NO BACKGROUND BOX)
                for bar, mean, std in zip(bars, means, stds):
                    height = bar.get_height()
                    value_text = self.format_value(mean, std)
                    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                           value_text, ha='center', va='bottom', 
                           fontsize=5+14, fontweight='bold')
                
                # Add significance bars with more room
                y_max = max([m + s for m, s in zip(means, stds)])
                y_offset = y_max + 0.15
                
                # Calculate maximum y position needed for p-values
                max_p_value_y = y_offset
                for i in enumerate(model_pairs):
                    max_p_value_y = max(max_p_value_y, y_offset + (i[0] * 0.10) + 0.08)
                
                for i, (model1, model2) in enumerate(model_pairs):
                    if model1 in self.models and model2 in self.models:
                        idx1 = self.models.index(model1)
                        idx2 = self.models.index(model2)
                        
                        if idx1 < len(x_pos) and idx2 < len(x_pos):
                            p_value = self.calculate_pvalue(model1, model2, metric)
                            
                            x1, x2 = x_pos[idx1], x_pos[idx2]
                            y = y_offset + (i * 0.10)
                            
                            # Draw significance line
                            ax.plot([x1, x2], [y, y], 'k-', linewidth=2)
                            ax.plot([x1, x1], [y - 0.025, y + 0.025], 'k-', linewidth=2)
                            ax.plot([x2, x2], [y - 0.025, y + 0.025], 'k-', linewidth=2)
                            
                            # Add p-value text (NO BACKGROUND BOX)
                            if p_value is not None:
                                significance = self.get_significance_symbol(p_value)
                                p_text = f"p={p_value:.3f} {significance}"
                                ax.text((x1 + x2) / 2, y + 0.035, p_text, 
                                       ha='center', va='bottom', fontsize=5+12, fontweight='bold')
                
                # Set y-limit with sufficient padding for p-values
                ax.set_ylim(0, max(1.2, max_p_value_y + 0.05))
                ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1.5)
                ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enhanced_overall_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Enhanced overall comparison saved (improved height and spacing)")
        plt.close()
    
    def create_perclass_detailed_heatmaps(self):
        """Create separate heatmaps for each metric with better spacing"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Per-Class Performance Comparison: All Models and Metrics', 
                     fontsize=5+20, fontweight='bold', y=0.995)
        
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
            
            # Create heatmap with better spacing
            im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels with larger fonts using proper model names
            ax.set_xticks(range(len(self.models)))
            ax.set_xticklabels([self.model_display_names.get(m, m.upper()) for m in self.models], 
                               fontweight='bold', fontsize=5+12, rotation=15, ha='right')
            ax.set_yticks(range(len(self.class_names)))
            ax.set_yticklabels(self.class_names, fontsize=5+14, fontweight='bold')
            ax.set_title(f'{metric_name}', fontsize=5+18, fontweight='bold', pad=15)
            
            # Add text annotations with 2 significant digits
            for i in range(len(self.class_names)):
                for j in range(len(self.models)):
                    if data_matrix[i][j] > 0:
                        text_color = 'white' if data_matrix[i][j] < 0.5 else 'black'
                        value_text = f"{data_matrix[i][j]:.2f}"
                        ax.text(j, i, value_text,
                               ha="center", va="center", color=text_color, 
                               fontsize=5+13, fontweight='bold')
            
            # Add colorbar with larger font
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Performance Score', rotation=270, labelpad=25, fontsize=5+14, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enhanced_perclass_heatmaps.png', dpi=300, bbox_inches='tight')
        print(f"✅ Enhanced per-class heatmaps saved")
        plt.close()
    
    def create_comprehensive_summary_table(self):
        """Create comprehensive summary table with 2 significant digits"""
        print("\n" + "="*120)
        print("ENHANCED COMPREHENSIVE THREE-MODEL COMPARISON TABLE")
        print("="*120)
        
        # Overall summary table
        print("\n📊 OVERALL PERFORMANCE (Mean ± Std, 2 significant digits):")
        print("-"*120)
        
        table_data = []
        for metric, metric_name in zip(self.metrics, self.metric_names):
            row = {'Metric': metric_name}
            for model in self.models:
                if metric in self.results[model] and 'Overall' in self.results[model][metric]:
                    stats_data = self.results[model][metric]['Overall']
                    row[self.model_display_names.get(model, model.upper())] = self.format_value(stats_data['mean'], stats_data['std'])
                else:
                    row[self.model_display_names.get(model, model.upper())] = 'N/A'
            table_data.append(row)
        
        df_overall = pd.DataFrame(table_data)
        print(df_overall.to_string(index=False))
        
        # Save to CSV with 2 significant digits
        csv_file = self.output_dir / 'enhanced_overall_summary.csv'
        df_overall.to_csv(csv_file, index=False)
        print(f"\n✅ Overall summary saved to: {csv_file}")
        
        # Per-class detailed tables
        print("\n\n📍 PER-CLASS PERFORMANCE SUMMARY (2 significant digits):")
        print("-"*120)
        
        all_class_data = []
        
        for class_name in self.class_names:
            print(f"\n{class_name.upper()}:")
            print("-" * 120)
            
            class_data = []
            for metric, metric_name in zip(self.metrics, self.metric_names):
                row = {'Metric': metric_name}
                for model in self.models:
                    if (metric in self.results[model] and 
                        class_name in self.results[model][metric]):
                        stats_data = self.results[model][metric][class_name]
                        row[self.model_display_names.get(model, model.upper())] = self.format_value(stats_data['mean'], stats_data['std'])
                    else:
                        row[self.model_display_names.get(model, model.upper())] = 'N/A'
                class_data.append(row)
            
            df_class = pd.DataFrame(class_data)
            print(df_class.to_string(index=False))
            
            # Track for comprehensive file - use ordered model display names
            for _, row in df_class.iterrows():
                all_class_data.append({
                    'Class': class_name,
                    'Metric': row['Metric'],
                    'Biomarker Model': row.get('Biomarker Model', 'N/A'),
                    '2D OCT/OCTA Model': row.get('2D OCT/OCTA Model', 'N/A'),
                    '3D OCT/OCTA Model': row.get('3D OCT/OCTA Model', 'N/A')
                })
        
        # Save comprehensive class data
        df_all_classes = pd.DataFrame(all_class_data)
        csv_class_file = self.output_dir / 'enhanced_perclass_summary.csv'
        df_all_classes.to_csv(csv_class_file, index=False)
        print(f"\n✅ Per-class summary saved to: {csv_class_file}")
        
        # Statistical significance table
        print("\n\n⚡ STATISTICAL SIGNIFICANCE TESTS (Paired t-test, 2 sig digits):")
        print("-"*120)
        
        sig_data = []
        for metric, metric_name in zip(self.metrics, self.metric_names):
            for model1, model2 in combinations(self.models, 2):
                p_value = self.calculate_pvalue(model1, model2, metric)
                if p_value is not None:
                    significance = self.get_significance_symbol(p_value)
                    mean1 = self.results[model1][metric]['Overall']['mean']
                    mean2 = self.results[model2][metric]['Overall']['mean']
                    diff = mean2 - mean1
                    
                    sig_data.append({
                        'Metric': metric_name,
                        'Comparison': f'{self.model_display_names.get(model1, model1.upper())} vs {self.model_display_names.get(model2, model2.upper())}',
                        'P-Value': f'{p_value:.3f}',
                        'Significance': significance,
                        'Difference': f'{diff:.2f}'
                    })
        
        df_sig = pd.DataFrame(sig_data)
        print(df_sig.to_string(index=False))
        
        csv_sig_file = self.output_dir / 'enhanced_statistical_tests.csv'
        df_sig.to_csv(csv_sig_file, index=False)
        print(f"\n✅ Statistical tests saved to: {csv_sig_file}")
        
        print("\n" + "="*120)
    
    def run_all_enhancements(self):
        """Run all enhanced visualizations"""
        try:
            self.load_results()
            
            print("📊 Creating enhanced overall comparison...")
            self.create_overall_comparison_enhanced()
            
            print("🗺️  Creating enhanced per-class heatmaps...")
            self.create_perclass_detailed_heatmaps()
            
            print("📋 Creating enhanced summary tables...")
            self.create_comprehensive_summary_table()
            
            print("\n" + "="*120)
            print("✅ ALL ENHANCED VISUALIZATIONS & TABLES COMPLETED!")
            print("="*120)
            print(f"\n📁 Results saved in: {self.output_dir}")
            print("\nGenerated files:")
            print("  • enhanced_overall_comparison.png (2 sig digits, larger fonts, spacious)")
            print("  • enhanced_perclass_heatmaps.png (per-metric heatmaps)")
            print("  • enhanced_overall_summary.csv (overall performance table)")
            print("  • enhanced_perclass_summary.csv (per-class performance table)")
            print("  • enhanced_statistical_tests.csv (p-values and significance)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    visualizer = EnhancedThreeModelVisualizer()
    visualizer.run_all_enhancements()