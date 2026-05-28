### Executive Summary
This project focuses on predicting purchase intent using a machine learning pipeline. It employs modular design principles, robust preprocessing, and evaluation protocols to address challenges such as imbalanced data and operational constraints. The pipeline is optimized for both ranking metrics (AUC) and operational metrics (F1, precision, recall).

### Project Objectives
- Develop a machine learning pipeline to predict purchase intent.
- Ensure preprocessing consistency and prevent train-test leakage.
- Optimize the pipeline for imbalanced classification using threshold tuning.
- Provide actionable insights for deployment and monitoring.

### Analytical Scope
- **Data Characteristics**: Imbalanced target (~15.5% positive class), missing values (~5% in multiple features), and bounded domain constraints.
- **Key Features**: Engagement/value-related metrics (`PageValue`, `ProductPageTime`) and categorical features (`TrafficSource`, `GeographicRegion`).
- **Evaluation Metrics**: AUC, F1, precision, recall, accuracy, and confusion matrix.

### Methodology
1. **Data Preparation**: Domain-aware cleaning, handling missing values, and stratified data splitting.
2. **Preprocessing**: Imputation, scaling, and one-hot encoding using `ColumnTransformer + Pipeline`.
3. **Modeling**: Training baseline models, selecting the best model based on validation AUC, and tuning thresholds for F1 optimization.
4. **Evaluation**: Retraining on combined train+validation data and evaluating on an untouched test set.
5. **Outputs**: Metrics and artifacts saved for reproducibility.

### Outputs
- **Metrics**: Validation AUC, F1, precision, recall, and test set evaluation metrics.
- **Artifacts**: Preprocessing pipelines, trained models, and threshold tuning results.
- **Files**: `validation_metrics.csv`, `threshold_tuning.csv`, `feature_importance.csv`, `summary.json`.

### Limitations
- Imbalanced data may still pose challenges despite threshold tuning.
- Missing value imputation assumes missingness at random, which may not always hold.
- Domain constraints and bounded features require careful validation to avoid introducing bias.
- The pipeline assumes the provided features are sufficient for robust predictions.

### Test Performance
The pipeline's test performance highlights the impact of threshold tuning on key metrics:

- **Default threshold (0.50)**:
  - AUC: 0.8891
  - F1: 0.5982
  - Precision: 0.7071
  - Recall: 0.5183
  - Accuracy: 0.8921
- **Tuned threshold (0.25)**:
  - AUC: 0.8891
  - F1: 0.6277
  - Precision: 0.5580
  - Recall: 0.7173
  - Accuracy: 0.8682

The tuned threshold improves recall significantly, aligning with objectives for imbalanced classification, though at the cost of precision and accuracy.

### Deployment Considerations
- **Data Validation**: Ensure schema consistency, enforce bounds, and monitor null rates before scoring.
- **Feature Drift Monitoring**: Track high-impact features like `PageValue`, `ProductPageTime`, and source mix for drift.
- **Retraining**: Retrain the model periodically or when performance degradation is detected.
- **Threshold Tuning**: Adjust the probability threshold based on the business objective (e.g., recall-oriented for high-risk scenarios or precision-oriented for cost-sensitive campaigns).
- **Logging and Monitoring**: Log predictions and outcomes to enable model monitoring, recalibration, and compliance with governance requirements.
