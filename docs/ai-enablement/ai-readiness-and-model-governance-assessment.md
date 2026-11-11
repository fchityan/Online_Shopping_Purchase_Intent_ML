# AI Readiness & Model Governance Assessment — Online Shopping Purchase Intent ML

## Model evidence

Logistic Regression, Random Forest and LightGBM are compared for session-level purchase intent. Model family and operating threshold are chosen on validation data before locked-test assessment.

**Recorded LightGBM locked-test ROC-AUC: 0.9253.** At threshold 0.50, precision was **0.7986** and recall **0.6126**; threshold 0.35 increased recall to **0.6702** while reducing precision to **0.6900**.

These measures describe the evaluated dataset, **not incremental campaign conversion**.

## AI enablement controls

| Control | Present in the project | Needed in operational use |
|---|---|---|
| Data quality | Preprocessing and data-quality checks | Event provenance, consent and schema contracts |
| Evaluation | Stratified validation and held-out test | Temporal holdout, generalization and calibration tests |
| Experiment tracking | MLflow runs | Model-approval procedure and reproducible releases |
| Decision governance | Versioned model and threshold | Agreed targeting capacity and false-positive cost |
| Business validation | Model metrics | Randomized controlled marketing experiment and ROI |
| Operational readiness | API and monitoring reference | Production access control, alerting and accountable owners |

**Boundary:** A validated test AUC does not demonstrate that targeting policies improve business results.

See [requirements and solution design](../business-analysis/business-requirements-and-solution-design.md).
