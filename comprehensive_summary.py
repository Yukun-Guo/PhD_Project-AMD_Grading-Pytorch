#!/usr/bin/env python3
"""
Comprehensive Results Summary with Enhanced Visualizations

This script provides a complete summary of all the enhanced visualizations
and key statistical findings from the 5-fold cross-validation analysis.
"""

import json
from pathlib import Path
from scipy import stats

def load_and_summarize_results():
    """Load results and provide comprehensive summary"""
    
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
    
    print("=" * 80)
    print("COMPREHENSIVE 5-FOLD CROSS-VALIDATION ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Analysis Directory: {latest_dir}")
    print(f"Models Compared: {', '.join([m.upper() for m in models])}")
    print(f"Generated on: {latest_dir.name}")
    print()
    
    # Overall Performance with Statistical Tests
    print("📊 OVERALL PERFORMANCE SUMMARY")
    print("-" * 50)
    
    for metric, metric_name in zip(metrics, metric_names):
        print(f"\n{metric_name}:")
        
        # Show means and std
        for model in models:
            if metric in results[model] and 'Overall' in results[model][metric]:
                stats_data = results[model][metric]['Overall']
                print(f"  {model.upper()}: {stats_data['mean']:.4f} ± {stats_data['std']:.4f}")
        
        # Statistical comparison
        if len(models) == 2:
            model1, model2 = models[0], models[1]
            if (metric in results[model1] and 'Overall' in results[model1][metric] and
                metric in results[model2] and 'Overall' in results[model2][metric]):
                
                values1 = results[model1][metric]['Overall']['values']
                values2 = results[model2][metric]['Overall']['values']
                
                try:
                    _, p_value = stats.ttest_rel(values1, values2)
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    
                    # Determine which model is better
                    mean1 = results[model1][metric]['Overall']['mean']
                    mean2 = results[model2][metric]['Overall']['mean']
                    better_model = model1.upper() if mean1 > mean2 else model2.upper()
                    
                    print(f"  → Statistical Test: {better_model} performs better (p={p_value:.4f} {significance})")
                except:
                    print(f"  → Statistical Test: Could not calculate")
    
    print("\n" + "=" * 80)
    print("🎯 KEY FINDINGS")
    print("=" * 80)
    
    # Determine overall winner
    if len(models) == 2:
        model1, model2 = models[0], models[1]
        significant_advantages = []
        
        for metric, metric_name in zip(metrics, metric_names):
            if (metric in results[model1] and 'Overall' in results[model1][metric] and
                metric in results[model2] and 'Overall' in results[model2][metric]):
                
                try:
                    values1 = results[model1][metric]['Overall']['values']
                    values2 = results[model2][metric]['Overall']['values']
                    _, p_value = stats.ttest_rel(values1, values2)
                    
                    if p_value < 0.05:
                        mean1 = results[model1][metric]['Overall']['mean']
                        mean2 = results[model2][metric]['Overall']['mean']
                        if mean1 > mean2:
                            significant_advantages.append(f"{model1.upper()} > {model2.upper()} in {metric_name}")
                        else:
                            significant_advantages.append(f"{model2.upper()} > {model1.upper()} in {metric_name}")
                except:
                    pass
        
        if significant_advantages:
            print("🏆 STATISTICALLY SIGNIFICANT ADVANTAGES:")
            for advantage in significant_advantages:
                print(f"  ✓ {advantage}")
        else:
            print("📊 No statistically significant differences found between models.")
    
    print("\n" + "=" * 80)
    print("📈 AVAILABLE VISUALIZATIONS")
    print("=" * 80)
    
    # List all available plots
    plot_files = [
        ("enhanced_overall_with_pvalues.png", "Enhanced overall performance comparison with p-values and significance bars"),
        ("simple_perclass_comparison.png", "Per-class performance comparison with statistical significance"),
        ("enhanced_overall_performance_comparison.png", "Detailed overall performance with comprehensive annotations"),
        ("perclass_performance_comparison.png", "Comprehensive per-class comparison across all metrics"),
        ("significance_matrix.png", "Statistical significance matrix showing all pairwise comparisons"),
        ("model_ranking_chart.png", "Model performance ranking chart across all metrics and classes"),
        ("comprehensive_analysis.png", "Radar chart and heatmap comprehensive analysis"),
        ("perclass_performance_heatmap.png", "Heatmap visualization of per-class performance")
    ]
    
    print("📁 Generated Visualization Files:")
    for filename, description in plot_files:
        filepath = latest_dir / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
            print(f"     └─ {description}")
        else:
            print(f"  ❌ {filename} (not found)")
    
    print(f"\n📊 Data Files:")
    data_files = [
        ("raw_analysis_results.json", "Raw statistical results and fold-by-fold data"),
        ("detailed_comparison.csv", "Detailed performance comparison table"),
        ("statistical_comparison.csv", "Statistical test results"),
        ("analysis_summary.txt", "Text summary of analysis")
    ]
    
    for filename, description in data_files:
        filepath = latest_dir / filename
        if filepath.exists():
            print(f"  ✅ {filename}")
            print(f"     └─ {description}")
    
    print("\n" + "=" * 80)
    print("🔍 PER-CLASS INSIGHTS")
    print("=" * 80)
    
    # Per-class analysis
    for class_name in class_names:
        print(f"\n{class_name}:")
        
        best_models = {}
        for metric, metric_name in zip(metrics, metric_names):
            best_score = -1
            best_model = None
            
            for model in models:
                if (model in results and metric in results[model] and 
                    class_name in results[model][metric]):
                    score = results[model][metric][class_name]['mean']
                    if score > best_score:
                        best_score = score
                        best_model = model.upper()
            
            if best_model:
                best_models[metric_name] = (best_model, best_score)
                print(f"  {metric_name}: {best_model} leads with {best_score:.3f}")
        
        # Summary for this class
        if best_models:
            model_counts = {}
            for model, _ in best_models.values():
                model_counts[model] = model_counts.get(model, 0) + 1
            
            overall_best = max(model_counts.items(), key=lambda x: x[1])
            print(f"  → Overall best for {class_name}: {overall_best[0]} (leads in {overall_best[1]}/4 metrics)")
    
    print("\n" + "=" * 80)
    print("💡 CLINICAL RECOMMENDATIONS")
    print("=" * 80)
    
    # Clinical insights based on sensitivity and specificity
    if len(models) == 2:
        model1, model2 = models[0], models[1]
        
        # Check sensitivity (most important for screening)
        if ('sensitivity' in results[model1] and 'Overall' in results[model1]['sensitivity'] and
            'sensitivity' in results[model2] and 'Overall' in results[model2]['sensitivity']):
            
            sens1 = results[model1]['sensitivity']['Overall']['mean']
            sens2 = results[model2]['sensitivity']['Overall']['mean']
            
            try:
                values1 = results[model1]['sensitivity']['Overall']['values']
                values2 = results[model2]['sensitivity']['Overall']['values']
                _, p_value = stats.ttest_rel(values1, values2)
                
                if p_value < 0.05:
                    better_model = model1.upper() if sens1 > sens2 else model2.upper()
                    better_sens = max(sens1, sens2)
                    print(f"🏥 For AMD Screening: {better_model} recommended")
                    print(f"   └─ Superior sensitivity: {better_sens:.3f} (p={p_value:.4f})")
                    print(f"   └─ Better at detecting early AMD cases")
            except:
                pass
    
    print(f"\n📂 All results saved in: {latest_dir}")
    print("=" * 80)

if __name__ == "__main__":
    load_and_summarize_results()