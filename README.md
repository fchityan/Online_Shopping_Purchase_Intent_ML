## Pipeline Flow and Design Logic
### Flow
```text
Load raw data
-> Apply domain-aware cleaning
-> Split data (train/validation/test)
-> Build preprocessing (impute + scale + one-hot)
-> Train baseline models
-> Select best model on validation AUC
-> Tune threshold on validation F1
-> Retrain on train+validation
-> Evaluate on untouched test set
-> Save metrics/artifacts
```

### Design rationale
- Used a modular design to separate concerns: ingestion, cleaning, modeling, and orchestration.
- Used `ColumnTransformer + Pipeline` to prevent train-test leakage and keep preprocessing consistent.
- Used both ranking metric (AUC) and operational classification metrics (F1, precision, recall).
- Included threshold tuning because the target is imbalanced and default 0.5 may not be optimal.

## Task 1 EDA Findings and Impact on Pipeline Choices
Quick EDA summary (full details in notebook):
- Target is imbalanced (~15.5% purchase, ~84.5% non-purchase).
- Missing values exist in several columns (~5% in multiple features).
- Domain constraints apply to bounded features: `BounceRate`, `ExitRate`, `SpecialDayProximity` in [0, 1].
- `TrafficSource` and `GeographicRegion` are coded categorical features.
- Engagement/value-related features (`PageValue`, `ProductPageTime`) are strong purchase signals.

Pipeline decisions based on EDA:
- Invalid bounded values are set to missing and imputed.
- Coded columns are treated as categories and one-hot encoded.
- Stratified splitting preserves class distribution.
- Threshold tuning is included to improve positive-class capture.

## Feature Processing Summary
| Feature | Type | Processing | Why |
|---|---|---|---|
| `CustomerType` | Categorical | Strip, lowercase, impute mode, one-hot encode | Normalize inconsistent labels and model category effects |
| `SpecialDayProximity` | Numeric (bounded) | Set out-of-range to missing, median impute, standardize | Preserve domain constraints and robust scaling |
| `ExitRate` | Numeric (bounded) | Set out-of-range to missing, median impute, standardize | Avoid invalid rates affecting model |
| `PageValue` | Numeric | Median impute, standardize | Handle missingness and scale differences |
| `TrafficSource` | Coded categorical | Convert to category label, impute mode, one-hot encode | Treat code values as categories, not magnitudes |
| `GeographicRegion` | Coded categorical | Convert to category label, impute mode, one-hot encode | Avoid false ordinal interpretation |
| `BounceRate` | Numeric (bounded) | Set out-of-range to missing, median impute, standardize | Enforce valid rate domain |
| `ProductPageTime` | Numeric | Median impute, standardize | Stabilize variance and handle missingness |
| `PurchaseCompleted` | Target | Binary classification target | Predicted output |

## Choice of Models
- **Logistic Regression**: strong linear baseline, interpretable, fast to train.
- **Gradient Boosting**: strong non-linear learner with robust tabular performance.
- **Random Forest**: captures non-linear relationships and interactions without heavy feature engineering.

Final selection from current run:
- Best model: `gb_baseline` (Gradient Boosting)
- Best threshold by validation F1: `0.25`

Validation metrics (current run):
- `gb_baseline`: AUC 0.9014, F1 0.6118, Precision 0.6957, Recall 0.5459, Accuracy 0.8929
- `rf_baseline`: AUC 0.8977, F1 0.5994, Precision 0.7179, Recall 0.5144, Accuracy 0.8938
- `logreg_baseline`: AUC 0.8686, F1 0.4912, Precision 0.7407, Recall 0.3675, Accuracy 0.8824

## Model Evaluation and Metrics
### Metrics used
- **AUC-ROC**: ranking quality independent of threshold.
- **F1-score**: balance of precision and recall for imbalanced binary classification.
- **Precision**: proportion of predicted positives that are true positives.
- **Recall**: proportion of true positives correctly identified.
- **Accuracy**: overall correctness.
- **Confusion matrix**: detailed error breakdown (TN, FP, FN, TP).

### Evaluation protocol
- Split into train/validation/test with stratification.
- Compare baseline models on validation set.
- Tune probability threshold on validation F1.
- Evaluate selected configuration on untouched test set.

Current test performance (`gb_baseline`):
- Default threshold 0.50: AUC 0.8891, F1 0.5982, Precision 0.7071, Recall 0.5183, Accuracy 0.8921
- Tuned threshold 0.25: AUC 0.8891, F1 0.6277, Precision 0.5580, Recall 0.7173, Accuracy 0.8682

Output files generated in `outputs/`:
- `validation_metrics.csv`
- `threshold_tuning.csv`
- `feature_importance.csv`
- `summary.json`

## Deployment Considerations
- Data validation checks should run before scoring (schema, bounds, null rates).
- Monitor drift in high-impact features (`PageValue`, `ProductPageTime`, source mix).
- Retrain periodically or when performance degradation is detected.
- Threshold should be tuned to business objective (recall-oriented vs precision-oriented campaigns).
- Log predictions and outcomes to support model monitoring and recalibration.