# Per-Class Metrics Explanation

## Why Per-Class Metrics Are Identical Across Averaging Types

### Summary
**Per-class metrics ARE THE SAME across micro, macro, and weighted averaging** - this is **CORRECT** behavior, not a bug!

### Understanding the Difference

#### Per-Class Metrics (Individual Class Performance)
Per-class metrics measure how well the model performs on **each individual class**:

- **Normal class**: Sensitivity = 0.92 ± 0.06
- **Early AMD**: Sensitivity = 0.38 ± 0.20
- **Intermediate AMD**: Sensitivity = 0.82 ± 0.02
- **Advanced AMD**: Sensitivity = 0.80 ± 0.04

These values represent the **true performance for that specific class** and do not depend on any averaging method.

#### Overall Metrics (Aggregated Performance)
Overall metrics **combine** the per-class metrics using different strategies:

| Averaging Type | How It Combines Per-Class Metrics | Example Calculation |
|----------------|-----------------------------------|---------------------|
| **Macro** | Simple average of all classes | `(0.92 + 0.38 + 0.82 + 0.80) / 4 = 0.73` |
| **Micro** | Weighted by total samples | `ΣTP / Σ(TP + FN)` across all samples |
| **Weighted** | Weighted by class frequency | `Σ(wi × metrici) / Σwi` |

### Real Example from Our Data

#### Overall Sensitivity (Differs by Averaging Type)
- **MACRO**: 0.73 ± 0.07 (treats all 4 classes equally)
- **MICRO**: 0.80 ± 0.03 (weighted by sample count)
- **WEIGHTED**: 0.80 ± 0.03 (weighted by class frequency)

**Why different?**
- Macro gives equal weight to "Early AMD" (rare, low sensitivity 0.38)
- Micro/Weighted give more weight to common classes (Normal, Intermediate)

#### Per-Class Sensitivity (Same Across All Types)
- **Normal**: 0.92 ± 0.06 (same in micro/macro/weighted)
- **Early AMD**: 0.38 ± 0.20 (same in micro/macro/weighted)
- **Intermediate AMD**: 0.82 ± 0.02 (same in micro/macro/weighted)
- **Advanced AMD**: 0.80 ± 0.04 (same in micro/macro/weighted)

**Why the same?**
These are the **actual class-specific performance values** - they don't get "averaged"!

### What This Means for Analysis

1. **When comparing models for a specific AMD stage:**
   - Use **per-class metrics** (same across all averaging types)
   - Example: "For Early AMD detection, BIO model has 0.65 sensitivity vs OCT's 0.38"

2. **When reporting overall model performance:**
   - Choose averaging type based on your goal:
     - **Macro**: All AMD stages equally important (research standard)
     - **Micro**: Overall diagnostic accuracy (clinical screening)
     - **Weighted**: Real-world population distribution

3. **When writing papers:**
   - Report **macro-averaged overall metrics** (standard in ML research)
   - Show **per-class metrics** in supplementary tables
   - Explain that per-class values are inherent to each class

### Analogy

Think of a student's grades in 4 subjects:
- Math: 92%
- History: 38%
- Science: 82%
- English: 80%

**Per-subject grades** (like per-class metrics):
- Always the same: Math=92%, History=38%, etc.
- Don't change based on how you calculate GPA

**Overall GPA** (like overall metrics):
- **Macro**: (92+38+82+80)/4 = 73% (all subjects equal weight)
- **Weighted**: If Math counts 2x, GPA changes
- **The individual subject grades stay 92%, 38%, etc.**

### Verification in Code

The code correctly handles this:

```python
# Lines 137-163: Per-class extraction (same for all averaging types)
for class_name in self.class_names:
    recall_row = df[(df['Metric'] == 'Recall') & 
                   (df['Type'] == 'Per-Class') &  # <-- Always 'Per-Class'
                   (df['Class'] == class_name)]
    # Same value stored for micro, macro, weighted
    for avg_type in ['micro', 'macro', 'weighted']:
        metrics_data[avg_type]['sensitivity'][fold_num][class_name] = recall_row['Value'].iloc[0]

# Lines 165-172: Overall extraction (different for each averaging type)
for avg_type in ['Micro', 'Macro', 'Weighted']:
    overall_row = df[(df['Metric'] == csv_metric) & 
                    (df['Type'] == avg_type) &  # <-- Type varies!
                    (df['Class'] == 'All')]
    # Different values for micro/macro/weighted
    metrics_data[avg_type.lower()][metric_name][fold_num]['Overall'] = overall_row['Value'].iloc[0]
```

### CSV File Structure

The source CSV files confirm this structure:

```csv
Metric,Type,Class,Value
# Overall metrics - DIFFER by averaging type
Recall,Macro,All,0.6510
Recall,Micro,All,0.7788
Recall,Weighted,All,0.7788

# Per-class metrics - SAME for all averaging types (no Type column variation)
Recall,Per-Class,Normal,0.84
Recall,Per-Class,Early AMD,0.1667
Recall,Per-Class,Intermediate AMD,0.8031
Recall,Per-Class,Advanced AMD,0.7943
```

Notice: Per-class rows have `Type='Per-Class'` (not Micro/Macro/Weighted).

### Conclusion

✅ **This is CORRECT behavior**
- Per-class metrics measure individual class performance
- They are inherently class-specific and don't change with averaging type
- Only the "Overall" aggregation differs between micro/macro/weighted

✅ **Updated output files now include clarification:**
```
PER-CLASS PERFORMANCE DETAILS
--------------------------------------------------
Note: Per-class metrics are identical across all averaging types.
      Averaging type only affects the 'Overall' aggregation above.
```

### For Your Paper

When reporting results, structure like this:

**Table 1: Overall Model Performance (Macro-Averaged)**
| Model | Sensitivity | Specificity | F1-Score | AUC-ROC |
|-------|-------------|-------------|----------|---------|
| OCT   | 0.73±0.07   | 0.92±0.01   | 0.74±0.06| 0.94±0.02|
| BIO   | 0.78±0.05   | 0.93±0.01   | 0.77±0.05| 0.95±0.01|
| 3D    | 0.71±0.14   | 0.93±0.03   | 0.72±0.14| 0.95±0.02|

**Table 2: Per-Stage AMD Detection Performance**
| AMD Stage | Best Model | Sensitivity | Specificity |
|-----------|------------|-------------|-------------|
| Normal    | OCT        | 0.92±0.06   | 0.99±0.01   |
| Early     | BIO        | 0.65±0.17   | 0.98±0.01   |
| Intermediate | 3D      | 0.83±0.06   | 0.84±0.08   |
| Advanced  | BIO        | 0.89±0.03   | 0.86±0.02   |

Note: Per-stage values are the same regardless of macro/micro/weighted averaging choice for overall metrics.
