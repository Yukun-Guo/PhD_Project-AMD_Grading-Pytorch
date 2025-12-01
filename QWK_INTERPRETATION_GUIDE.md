# Quadratic Weighted Kappa (QWK) - Clinical Interpretation Guide
QWK is inherently a multi-class overall metric - it measures agreement across all classes simultaneously by using the confusion matrix. Unlike metrics such as Sensitivity, Specificity, or F1-Score, QWK cannot be meaningfully calculated per-class because:

QWK uses the full confusion matrix - it requires knowledge of how all classes relate to each other (ordinal distances)
Ordinal weighting is global - the quadratic penalty depends on distances between all class pairs
Binary QWK loses ordinal information - computing QWK for one class vs. rest would lose the ordinal relationships between the other classes

## Overview
Quadratic Weighted Kappa (QWK) has been successfully integrated into the 5-fold cross-validation analysis to assess the agreement between predicted and true AMD grades. Unlike standard accuracy, QWK accounts for the **ordinal nature** of AMD grading (Normal → Early → Intermediate → Advanced) and penalizes larger disagreements more heavily.

## Current Results

### Model Performance (QWK Scores)
| Model | QWK Score | Interpretation |
|-------|-----------|----------------|
| **BIO** | 0.85 ± 0.03 | Almost Perfect Agreement |
| **OCT** | 0.83 ± 0.04 | Substantial Agreement |
| **3D**  | 0.83 ± 0.09 | Substantial Agreement |

### QWK Interpretation Scale
| Range | Agreement Level | Clinical Meaning |
|-------|----------------|------------------|
| 0.81 - 1.00 | Almost Perfect | Excellent clinical reliability |
| 0.61 - 0.80 | Substantial | Good clinical reliability |
| 0.41 - 0.60 | Moderate | Fair clinical reliability |
| 0.21 - 0.40 | Fair | Limited clinical reliability |
| 0.00 - 0.20 | Slight | Poor clinical reliability |
| < 0.00 | Poor | No agreement (worse than chance) |

## Key Findings

### 1. All Models Achieve High Agreement
- All three models demonstrate **substantial to almost perfect agreement** (QWK > 0.80)
- This indicates the models are clinically reliable for AMD grading
- The BIO model shows slightly higher agreement (0.85) compared to OCT and 3D (0.83)

### 2. Consistency Across Folds
- **OCT**: Lowest variability (σ = 0.04) → Most consistent across folds
- **BIO**: Low variability (σ = 0.03) → Very consistent
- **3D**: Higher variability (σ = 0.09) → More sensitive to fold composition

### 3. QWK vs Other Metrics
QWK provides complementary information:
- **Accuracy/F1-Score**: Treat all misclassifications equally
- **QWK**: Penalizes severe misclassifications (e.g., Advanced predicted as Normal) more than adjacent errors (e.g., Intermediate predicted as Early)
- This makes QWK particularly valuable for **ordinal classification tasks** like AMD grading

## Clinical Implications

### Why QWK Matters for AMD Grading
1. **Ordinal Disease Progression**: AMD naturally progresses through stages (Normal → Early → Intermediate → Advanced)
2. **Clinical Impact**: Misclassifying Advanced AMD as Normal is far more serious than confusing Early with Intermediate
3. **Treatment Decisions**: QWK better reflects the clinical cost of different types of errors

### Model Selection Guidance
- **BIO Model**: Highest QWK (0.85) suggests best overall ordinal classification
- **OCT Model**: High QWK (0.83) with lowest variability → Most stable predictions
- **3D Model**: High QWK (0.83) but higher variability → May benefit from more data or regularization

## Technical Details

### How QWK is Calculated
1. **Data Source**: Uses existing `validation_results.json` files from each fold
2. **Method**: `cohen_kappa_score(y_true, y_pred, weights='quadratic')`
3. **Weighting**: Quadratic penalty matrix for ordinal classes:
   ```
   Weight = 1 - ((i - j)² / (k - 1)²)
   where i, j = predicted/true class indices
         k = number of classes (4 for AMD)
   ```

### Why QWK is Same Across Averaging Types
- QWK is computed from the **confusion matrix** (overall predictions)
- Unlike Sensitivity/Specificity/F1, it's not calculated per-class then averaged
- Similar to accuracy, QWK represents **global agreement** regardless of averaging type

## Recommendations

### For Clinical Deployment
1. ✅ All models have sufficient QWK (> 0.80) for clinical consideration
2. ✅ BIO model shows best ordinal classification performance
3. ⚠️ 3D model's higher variability suggests additional validation may be needed

### For Further Analysis
1. **Confusion Matrix Analysis**: Examine which adjacent grades are most confused
2. **Per-Severity Analysis**: Check if QWK varies for different disease severities
3. **Threshold Optimization**: Consider adjusting decision thresholds to improve QWK

## References
- Cohen, J. (1968). "Weighted kappa: Nominal scale agreement with provision for scaled disagreement or partial credit." *Psychological Bulletin*, 70(4), 213-220.
- Fleiss, J. L., & Cohen, J. (1973). "The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability." *Educational and Psychological Measurement*, 33(3), 613-619.

## File Locations
- **Analysis Results**: `analysis_results/20251201_135522/`
- **QWK Values**: Found in all summary files (TXT, CSV, PNG, JSON)
- **Visualizations**: `overall_performance_comparison_{micro|macro|weighted}.png` (2×3 grid with QWK panel)
- **Raw Data**: `raw_analysis_results.json` contains QWK per model and averaging type

---
*Generated: December 1, 2024*  
*Analysis Directory: analysis_results/20251201_135522*
