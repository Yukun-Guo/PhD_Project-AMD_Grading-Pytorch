## Metric Equations and Aggregation Methods

This document lists the formulas used to compute the four primary metrics reported in the "OVERALL PERFORMANCE COMPARISON" section of the analysis summary: Sensitivity (Recall), Specificity, F1-Score, and AUC-ROC. It also explains how multiclass aggregation and fold-level aggregation (mean ± std) are calculated.

---

**Definitions (per class, from confusion matrix)**

- TP = True Positives (predicted positive and actually positive)
- TN = True Negatives (predicted negative and actually negative)
- FP = False Positives (predicted positive but actually negative)
- FN = False Negatives (predicted negative but actually positive)

All equations below can be applied per-class (one-vs-rest) in multiclass settings.

---

**1) Sensitivity (Recall, True Positive Rate)**

Equation:

$$\text{Sensitivity} = \dfrac{TP}{TP + FN}$$

Interpretation: fraction of actual positives correctly identified.

---

**2) Specificity (True Negative Rate)**

Equation:

$$\text{Specificity} = \dfrac{TN}{TN + FP}$$

Interpretation: fraction of actual negatives correctly identified.

---

**3) Precision (Positive Predictive Value)**

Equation:

$$\text{Precision} = \dfrac{TP}{TP + FP}$$

---

**4) F1-Score**

Harmonic mean of Precision and Recall (Sensitivity):

$$\text{F1} = 2 \cdot \dfrac{\text{Precision} \cdot \text{Sensitivity}}{\text{Precision} + \text{Sensitivity}}$$

Algebraically (in terms of TP, FP, FN):

$$\text{F1} = \dfrac{2\,TP}{2\,TP + FP + FN}$$

---

**5) AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**

Conceptual definition:

$$\text{AUC} = \int_{0}^{1} \text{TPR}(\text{FPR})\;d(\text{FPR})$$

Where TPR = True Positive Rate (Sensitivity) and FPR = False Positive Rate = FP / (FP + TN).

Practical computation: computed numerically from predicted scores using trapezoidal integration (e.g., `sklearn.metrics.roc_auc_score`). For multiclass problems a common approach is one-vs-rest AUC per class followed by averaging (see below).

---

**Multiclass aggregation methods**

Suppose there are C classes.

- Macro-average (unweighted average across classes):

  $$M_{macro} = \frac{1}{C} \sum_{c=1}^{C} M_c$$

  Here $M_c$ is the metric computed for class $c$ in a one-vs-rest manner.

- Micro-average (aggregate counts across classes then compute):

  Aggregate confusion counts across classes:

  $$TP_{micro} = \sum_{c} TP_c,\ \ FP_{micro} = \sum_{c} FP_c,\ \ FN_{micro} = \sum_{c} FN_c,\ \ TN_{micro} = \sum_{c} TN_c$$

  Then compute metrics from aggregates, e.g.:

  $$\text{Sensitivity}_{micro} = \dfrac{TP_{micro}}{TP_{micro} + FN_{micro}}$$

- Weighted-average (class-weighted by support):

  $$M_{weighted} = \sum_{c=1}^{C} w_c M_c \quad\text{with } w_c = \dfrac{N_c}{\sum_c N_c}$$

  where $N_c$ is the number of true examples of class $c$ (support).

Notes on AUC multiclass:
- One-vs-rest AUC per class then macro-average is common.
- `sklearn.metrics.roc_auc_score` supports `multi_class='ovr'` and different `average` options.

---

**Fold-level aggregation (5-fold cross-validation)**

Let the dataset be split into K folds (K=5). For each fold k, compute the metric m_k (this may be a scalar overall metric for the fold obtained by macro-averaging per-class values or by micro-aggregation; see repository code for which was used). The overall reported metric is:

- Mean across folds:

  $$\overline{m} = \dfrac{1}{K} \sum_{k=1}^{K} m_k$$

- Standard deviation across folds (sample std):

  $$s = \sqrt{\dfrac{1}{K-1} \sum_{k=1}^{K} (m_k - \overline{m})^2}$$

Reported value in the summary: `mean ± std` (for example `0.78±0.05`).

---

**One-vs-rest procedure for computing per-class metrics**

To compute per-class TP/FP/TN/FN in multiclass classification, treat the target class as "positive" and all other classes as "negative". Then compute TP/FP/TN/FN accordingly and apply the formulas above.

---

**Practical code snippets (scikit-learn)**

Compute per-fold overall F1 (macro) and AUC (macro) example:

```python
from sklearn.metrics import f1_score, roc_auc_score

# y_true: shape (N,) with class labels
# y_pred: shape (N,) with predicted class labels
# y_score: shape (N, C) with predicted scores/probabilities per class

# Macro F1 for the fold
f1_macro = f1_score(y_true, y_pred, average='macro')

# Macro AUC (one-vs-rest)
auc_macro = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
```

Compute sensitivity/recall and specificity per-class from a confusion matrix:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred, labels=class_labels)
# cm is CxC where cm[i,j] = count of true class i predicted as class j

# For class c (index i):
TP = cm[i, i]
FN = cm[i, :].sum() - TP
FP = cm[:, i].sum() - TP
TN = cm.sum() - (TP + FP + FN)

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
precision = TP / (TP + FP)
f1 = 2 * precision * sensitivity / (precision + sensitivity)
```

---

If you want, I can inspect the exact function in this repository that computed the "Overall" numbers (to confirm whether `macro` or `micro` aggregation was used) and add a one-line note to this file stating the repository's chosen aggregation. Do you want me to locate and confirm that now?

---

**Repository implementation note**

In this codebase, the reported "Overall" metrics are computed using per-fold *macro-averaging* (one-vs-rest per class, then averaged across classes) for Precision/Recall/F1 and AUC. Specifically:

- Per-fold: metrics are computed using `average='macro'` (see `compute_validation_metrics` in `TrainerFitKFold.py`).
- AUCs for multiclass are computed with one-vs-rest AUC per class and `average='macro'` (`roc_auc_score(..., multi_class='ovr', average='macro')`).
- Across folds: the final reported value is the mean ± sample standard deviation of the per-fold macro metric values (K=5 folds).

Files of interest:
- `TrainerFitKFold.py` and `TrainerFitKFold3D.py`: per-fold metric computation (macro, micro, weighted available)
- `analyze_5fold_results.py`: extraction of per-fold macro 'Overall' values and aggregation (mean ± std)

This note confirms the aggregation method used to produce the numbers in `analysis_summary.txt`.

---

## Requested specific formulas

Below are the explicit formulas you requested, matching the repository's settings:

1) Sensitivity with `average='macro'` (per-fold)

Compute per-class sensitivity for each class c:

$$\text{Sensitivity}_c = \dfrac{TP_c}{TP_c + FN_c}$$

Macro-averaged sensitivity for a fold (C classes):

$$\text{Sensitivity}_{macro}^{(fold)} = \dfrac{1}{C} \sum_{c=1}^{C} \text{Sensitivity}_c = \dfrac{1}{C} \sum_{c=1}^{C} \dfrac{TP_c}{TP_c + FN_c}$$

2) Specificity with `average='micro'` (per-fold)

First aggregate confusion counts across classes for the fold:

$$TP_{micro} = \sum_{c=1}^{C} TP_c,\quad FP_{micro} = \sum_{c=1}^{C} FP_c,\quad TN_{micro} = \sum_{c=1}^{C} TN_c,\quad FN_{micro} = \sum_{c=1}^{C} FN_c$$

Micro-averaged specificity for the fold is then computed from aggregated counts:

$$\text{Specificity}_{micro}^{(fold)} = \dfrac{TN_{micro}}{TN_{micro} + FP_{micro}}$$

3) F1 with `average='weighted'` (per-fold)

Compute per-class F1 as usual:

$$\text{F1}_c = \dfrac{2\,TP_c}{2\,TP_c + FP_c + FN_c}$$

Let the support (true example count) for class c be $N_c$. The weighted-average F1 for the fold is:

$$\text{F1}_{weighted}^{(fold)} = \sum_{c=1}^{C} w_c \; \text{F1}_c \quad\text{with}\quad w_c = \dfrac{N_c}{\sum_{j=1}^{C} N_j}$$

4) AUC with `multi_class='ovr'` and `average='macro'` (per-fold)

Compute one-vs-rest AUC for each class c using the class scores/probabilities:

$$\text{AUC}_c = \text{AUC}_{\text{one-vs-rest}}(c)$$

Macro-averaged multiclass AUC for the fold:

$$\text{AUC}_{macro}^{(fold)} = \dfrac{1}{C} \sum_{c=1}^{C} \text{AUC}_c$$

---

## Cross-fold aggregation (K-fold, here K=5)

Let the per-fold metric value be denoted by $m_k$ for fold $k$ (where $m_k$ is computed using the per-fold formulas above — e.g., Sensitivity_{macro}^{(fold k)}, Specificity_{micro}^{(fold k)}, F1_{weighted}^{(fold k)}, AUC_{macro}^{(fold k)}).

The cross-fold aggregated mean and sample standard deviation are:

- Mean across folds:

  $$\overline{m} = \dfrac{1}{K} \sum_{k=1}^{K} m_k$$

- Sample standard deviation across folds (ddof=1):

  $$s = \sqrt{\dfrac{1}{K-1} \sum_{k=1}^{K} (m_k - \overline{m})^2}$$

Reported value in summaries: `\overline{m} \pm s` (for example `0.78\pm0.05`).

Concretely, for each metric:

- Sensitivity (macro): compute Sensitivity_{macro}^{(fold k)} per fold, then

  $$\overline{\text{Sensitivity}}_{macro} = \dfrac{1}{K} \sum_{k=1}^{K} \text{Sensitivity}_{macro}^{(fold k)},\quad s_{sens} = \sqrt{\dfrac{1}{K-1} \sum_{k=1}^{K} (\text{Sensitivity}_{macro}^{(fold k)} - \overline{\text{Sensitivity}}_{macro})^2}$$

- Specificity (micro): compute Specificity_{micro}^{(fold k)} per fold from aggregated counts, then

  $$\overline{\text{Specificity}}_{micro} = \dfrac{1}{K} \sum_{k=1}^{K} \text{Specificity}_{micro}^{(fold k)},\quad s_{spec} = \sqrt{\dfrac{1}{K-1} \sum_{k=1}^{K} (\text{Specificity}_{micro}^{(fold k)} - \overline{\text{Specificity}}_{micro})^2}$$

- F1 (weighted): compute F1_{weighted}^{(fold k)} per fold, then

  $$\overline{\text{F1}}_{weighted} = \dfrac{1}{K} \sum_{k=1}^{K} \text{F1}_{weighted}^{(fold k)},\quad s_{f1} = \sqrt{\dfrac{1}{K-1} \sum_{k=1}^{K} (\text{F1}_{weighted}^{(fold k)} - \overline{\text{F1}}_{weighted})^2}$$

- AUC (macro, OVR): compute AUC_{macro}^{(fold k)} per fold, then

  $$\overline{\text{AUC}}_{macro} = \dfrac{1}{K} \sum_{k=1}^{K} \text{AUC}_{macro}^{(fold k)},\quad s_{auc} = \sqrt{\dfrac{1}{K-1} \sum_{k=1}^{K} (\text{AUC}_{macro}^{(fold k)} - \overline{\text{AUC}}_{macro})^2}$$

---

I have appended these explicit formulas to the `METRIC_EQUATIONS.md` file in the analysis results directory. Let me know if you want: (A) numeric verification by re-computing any of these aggregated values from `raw_analysis_results.json`, or (B) a short runnable script that reads `raw_analysis_results.json` and prints the per-fold and aggregated values using these exact formulas.
