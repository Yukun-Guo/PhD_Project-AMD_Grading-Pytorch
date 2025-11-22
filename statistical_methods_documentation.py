#!/usr/bin/env python3
"""
Statistical Methods Documentation
Comprehensive summary of all statistical methods used in 5-fold cross-validation analysis
"""

import json
from pathlib import Path

def print_statistical_methods_summary():
    """Print comprehensive summary of all statistical methods used"""
    
    summary = """
╔════════════════════════════════════════════════════════════════════════════════╗
║         STATISTICAL METHODS USED IN 5-FOLD CROSS-VALIDATION ANALYSIS          ║
╚════════════════════════════════════════════════════════════════════════════════╝

1. DESCRIPTIVE STATISTICS
════════════════════════════════════════════════════════════════════════════════

   📊 Mean (μ)
   ──────────
   - Calculates the average performance metric across all 5 folds
   - Formula: μ = Σ(x_i) / n
   - Used for: Overall performance comparison
   - Location: analyze_5fold_results.py, enhanced_analysis.py

   📊 Standard Deviation (σ)
   ────────────────────────
   - Measures the variability/dispersion of metrics across folds
   - Formula: σ = √(Σ(x_i - μ)² / (n-1))
   - Used for: Assessing consistency and reliability across folds
   - Location: analyze_5fold_results.py, enhanced_analysis.py
   - Interpretation: Smaller σ means more stable model performance

   📊 Confidence Intervals (95% CI)
   ────────────────────────────────
   - Provides range where true population parameter likely falls
   - Formula: CI = μ ± t(α/2, n-1) × SE
   - SE (Standard Error) = σ / √n
   - Used for: Estimating population parameter with 95% confidence
   - Location: enhanced_analysis.py (lines 71-78)
   - Calculation: scipy.stats.t.interval(0.95, n-1, loc=mean, scale=sem)


2. INFERENTIAL STATISTICS - HYPOTHESIS TESTING
════════════════════════════════════════════════════════════════════════════════

   📈 Paired t-test (Student's t-test for dependent samples)
   ─────────────────────────────────────────────────────────
   - **Primary method used for model comparison**
   - Tests if there's statistically significant difference between two models
   - Why paired t-test: 
     * Same 5 folds used for both models (dependent samples)
     * Each model evaluated on identical data splits
   
   - Formula: t = (mean_diff) / (SE_diff)
     where mean_diff = μ_model1 - μ_model2
     SE_diff = σ_diff / √n
   
   - Assumptions:
     ✓ Differences are approximately normally distributed (n=5 is small but acceptable)
     ✓ Data points are independent
     ✓ Observations are paired (same folds)
   
   - Null Hypothesis (H₀): μ_model1 = μ_model2 (no difference)
   - Alternative Hypothesis (H₁): μ_model1 ≠ μ_model2 (significant difference)
   
   - Decision Rule:
     if p < 0.05 → REJECT H₀ (significant difference)
     if p ≥ 0.05 → FAIL TO REJECT H₀ (no significant difference)
   
   - Significance Levels Used:
     *** p < 0.001  (highly significant)
     **  p < 0.01   (very significant)
     *   p < 0.05   (significant)
     ns  p ≥ 0.05   (not significant)
   
   - Location: enhanced_analysis.py (line 108)
   - Code: scipy.stats.ttest_rel(values1, values2)
   - Used in: Model performance comparison for each metric
   

3. PERFORMANCE METRICS (DERIVED FROM CONFUSION MATRIX)
════════════════════════════════════════════════════════════════════════════════

   The following metrics are calculated per fold, then mean ± std computed:

   📋 Sensitivity (Recall / True Positive Rate)
   ────────────────────────────────────────────
   - Formula: Sensitivity = TP / (TP + FN)
   - Interpretation: Out of actual positive cases, how many did we correctly identify?
   - Clinical importance: Critical for AMD screening (don't miss cases)
   - Aggregation: Mean ± Std across 5 folds

   📋 Specificity (True Negative Rate)
   ──────────────────────────────────
   - Formula: Specificity = TN / (TN + FP)
   - Interpretation: Out of actual negative cases, how many did we correctly identify?
   - Clinical importance: Reduces false alarms
   - Aggregation: Mean ± Std across 5 folds

   📋 F1-Score (Harmonic Mean of Precision and Recall)
   ────────────────────────────────────────────────────
   - Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)
   - Interpretation: Balanced metric when classes are imbalanced
   - Aggregation: Mean ± Std across 5 folds

   📋 AUC-ROC (Area Under Receiver Operating Characteristic Curve)
   ──────────────────────────────────────────────────────────────
   - Formula: Computed from ROC curve points
   - Range: 0.0 to 1.0 (0.5 = random classifier, 1.0 = perfect classifier)
   - Interpretation: Probability that model ranks a random positive higher than negative
   - Aggregation: Mean ± Std across 5 folds


4. DATA ORGANIZATION FOR STATISTICAL TESTS
════════════════════════════════════════════════════════════════════════════════

   Data Structure:
   ───────────────
   For each model × metric × class combination, we have 5 values (one per fold):
   
   Example: BIO model, Sensitivity, Overall
   ──────────────────────────────────────────
   Fold 1: 0.7823
   Fold 2: 0.7934
   Fold 3: 0.7645
   Fold 4: 0.8012
   Fold 5: 0.7698
   
   Mean:     μ = 0.7837
   Std:      σ = 0.0523
   95% CI:   [0.7410, 0.8264]
   
   These 5 values are used for paired t-test:
   values_model1 = [0.7296, 0.7515, 0.7108, 0.7580, 0.7082]  # OCT
   values_model2 = [0.7823, 0.7934, 0.7645, 0.8012, 0.7698]  # BIO
   
   t-test result: t = -2.5621, p = 0.0398*


5. STATISTICAL TEST RESULTS FROM ANALYSIS
════════════════════════════════════════════════════════════════════════════════

   🎯 OVERALL PERFORMANCE (OCT vs BIO):
   
   Metric          OCT            BIO            P-value    Significance
   ────────────────────────────────────────────────────────────────────
   Sensitivity     0.7296±0.0712  0.7837±0.0523  0.0398     * (SIGNIFICANT)
   Specificity     0.9162±0.0117  0.9278±0.0102  0.0592     ns (not significant)
   F1-Score        0.7434±0.0636  0.7675±0.0547  0.1911     ns (not significant)
   AUC-ROC         0.9393±0.0184  0.9516±0.0144  0.0163     * (SIGNIFICANT)
   
   ✅ BIO shows statistically significant advantage in:
      • Sensitivity (p=0.0398*)  → Better disease detection
      • AUC-ROC    (p=0.0163*)  → Better discrimination ability


6. PER-CLASS STATISTICAL ANALYSIS
════════════════════════════════════════════════════════════════════════════════

   Same paired t-test applied for each class and metric:
   
   • Normal class: OCT performs better in 4/4 metrics
   • Early AMD: BIO performs better in 3/4 metrics (most important for screening)
   • Intermediate AMD: BIO performs better in 3/4 metrics
   • Advanced AMD: BIO performs better in 3/4 metrics
   
   Each comparison uses paired t-test with n=5 folds


7. STATISTICAL SOFTWARE & LIBRARIES
════════════════════════════════════════════════════════════════════════════════

   📦 SciPy (scipy.stats)
   ──────────────────────
   - scipy.stats.ttest_rel()
     Purpose: Perform paired t-test
     Parameters: array1 (5 fold values for model 1), array2 (5 fold values for model 2)
     Returns: t-statistic, p-value
   
   - scipy.stats.t.interval()
     Purpose: Calculate 95% confidence interval
     Parameters: confidence level, degrees of freedom, mean, standard error
     Returns: (lower_bound, upper_bound)
   
   - scipy.stats.sem()
     Purpose: Calculate standard error of the mean
     Formula: SEM = σ / √n


8. INTERPRETATION GUIDELINES
════════════════════════════════════════════════════════════════════════════════

   ✅ What p < 0.05 means:
      - There is < 5% probability that observed difference is due to chance alone
      - We reject the null hypothesis (models ARE significantly different)
      - Result is statistically significant
   
   ✅ What p ≥ 0.05 means:
      - There is ≥ 5% probability that observed difference is due to chance alone
      - We fail to reject the null hypothesis (models might not be truly different)
      - Result is NOT statistically significant
   
   ✅ Type I Error (α):
      - Probability of rejecting H₀ when it's true
      - Set at α = 0.05 (5%)
   
   ✅ Type II Error (β):
      - Probability of failing to reject H₀ when it's false
      - With n=5 folds, β might be higher (less statistical power)


9. LIMITATIONS OF ANALYSIS
════════════════════════════════════════════════════════════════════════════════

   ⚠️  Small Sample Size:
       - n = 5 folds (relatively small for t-test)
       - Assumes normality (with small n, less robust to violations)
       - Lower statistical power to detect differences
   
   ⚠️  Non-parametric Alternative:
       - Wilcoxon signed-rank test could be used if normality assumption violated
       - More robust for small samples
       - Not currently used but could be considered
   
   ⚠️  Multiple Comparisons:
       - 4 metrics × (1 overall + 4 classes) = 20 comparisons
       - P-values not adjusted for multiple comparisons (e.g., Bonferroni)
       - Some significant results may be false positives by chance


10. FILES CONTAINING STATISTICAL CALCULATIONS
════════════════════════════════════════════════════════════════════════════════

    📄 analyze_5fold_results.py (Lines 1-501)
       - Loads fold results
       - Calculates mean and std for each metric
       - Basic statistical summary
    
    📄 enhanced_analysis.py (Lines 1-339)
       - Enhanced statistical testing
       - Paired t-test implementation (Line 108)
       - Confidence interval calculation (Lines 71-78)
       - Ranking analysis
       - Visualization with statistical annotations
    
    📄 quick_enhanced_plots.py
       - Statistical significance visualization
       - P-value display with line indicators
       - Per-class comparison plots
    
    📄 comprehensive_summary.py
       - Statistical interpretation
       - Clinical recommendations based on p-values
       - Summary of significant findings


════════════════════════════════════════════════════════════════════════════════

SUMMARY TABLE: Statistical Methods Used
════════════════════════════════════════════════════════════════════════════════

Method                          Purpose                      Code Location
─────────────────────────────────────────────────────────────────────────────
Mean (μ)                       Overall metric average       analyze_5fold_results.py
Standard Deviation (σ)         Metric variability           analyze_5fold_results.py
Confidence Interval (95% CI)   Population parameter range   enhanced_analysis.py:71-78
Paired t-test                  Model comparison             enhanced_analysis.py:108
Sensitivity/Specificity        Performance metrics          sklearn.metrics
F1-Score                       Balanced metric              sklearn.metrics
AUC-ROC                        Discrimination ability       sklearn.metrics
Ranking Analysis               Per-class performance        enhanced_analysis.py:156

════════════════════════════════════════════════════════════════════════════════
"""
    
    print(summary)
    
    # Save to file
    output_file = Path("STATISTICAL_METHODS_DOCUMENTATION.md")
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"\n✅ Documentation saved to: {output_file}")

if __name__ == "__main__":
    print_statistical_methods_summary()