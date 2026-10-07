# Online Shopping Purchase Intent ML — Production Service

An end-to-end imbalanced-classification system for predicting online purchase intent, with reproducible model selection, threshold tuning, experiment tracking, deployable model packaging, API serving, monitoring, and containerization.

## Portfolio Snapshot

**Problem:** Purchase sessions are imbalanced, so a useful decision system must balance missed purchasers against unnecessary targeting.

**Method:** Stratified train/validation/test splitting, shared preprocessing, Logistic Regression / Random Forest / LightGBM comparison, validation-only threshold tuning, and MLflow experiment tracking.

**Production output:** A versioned model bundle that contains preprocessing + classifier + operating threshold + schema, a FastAPI service with liveness/readiness/metadata endpoints, bounded batch inference, Docker deployment, and lightweight drift/data-quality/business monitoring.

**Evidence:** The existing locked test evaluation reports AUC **0.9253** for the selected LightGBM model. Threshold choice changes the precision/recall trade-off without changing ranking AUC.

**Stack:** Python · scikit-learn · LightGBM · MLflow · FastAPI · Docker

## Production architecture

```text
raw session data
      │
      ▼
domain validation / cleaning
      │
      ▼
stratified train / validation / test
      │
      ├── Logistic Regression
      ├── Random Forest
      └── LightGBM
      │
      ▼
validation model selection + threshold tuning
      │
      ▼
locked holdout evaluation
      │
      ├── MLflow run + metrics
      ├── monitoring outputs
      └── outputs/model_bundle.joblib
                    │
                    ▼
             FastAPI / Docker
```

The serving artifact is no longer just a classifier. `model_bundle.joblib` stores the fitted preprocessing/model pipeline together with:

- selected model name and human-readable label
- model version and creation timestamp
- tuned operating threshold
- expected feature columns
- target column and random state
- locked test metrics

This prevents the API from silently reverting to a hard-coded `0.50` threshold after training.

## Train

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train_model \
  --data-path online_shopping.csv \
  --output-dir outputs \
  --random-state 42 \
  --mlflow-tracking-uri file:./mlruns_purchase_intent \
  --experiment-name purchase-intent-lightgbm
```

The pipeline writes:

```text
outputs/
├── model_bundle.joblib
├── summary.json
├── validation_metrics.csv
├── threshold_tuning.csv
├── feature_importance.csv
├── weekly_performance_summary.csv
├── data_quality_alerts.json
├── drift_report.json
└── business_impact.csv
```

## Model selection and operating threshold

Candidate models:

- Logistic Regression
- Random Forest
- LightGBM

The best model is selected by validation AUC. The operating threshold is then tuned only on the validation split for F1. The untouched test split is evaluated after model and threshold selection are locked.

Existing recorded validation results at threshold `0.50`:

| Model | AUC | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.9230 | 0.6922 | 0.7886 | 0.6168 | 0.9152 |
| Random Forest | 0.9203 | 0.6842 | 0.8340 | 0.5801 | 0.9173 |
| Logistic Regression | 0.8599 | 0.5361 | 0.7761 | 0.4094 | 0.8905 |

Existing locked test results:

**Threshold 0.50**
- AUC: 0.9253
- F1: 0.6933
- Precision: 0.7986
- Recall: 0.6126
- Accuracy: 0.9161

**Threshold 0.35**
- AUC: 0.9253
- F1: 0.6799
- Precision: 0.6900
- Recall: 0.6702
- Accuracy: 0.9023

The lower threshold captures more potential purchasers at the cost of more false positives. The serving API uses the threshold stored with the promoted artifact rather than a hard-coded value.

## Serve

After training:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoints:

```text
GET  /
GET  /live
GET  /health      # backward-compatible liveness alias
GET  /ready
GET  /metadata
POST /predict
```

`/live` answers whether the service process is alive. `/ready` answers whether the model artifact has loaded successfully. This keeps process health separate from model readiness.

`MODEL_PATH` defaults to `outputs/model_bundle.joblib`. `MAX_BATCH_SIZE` defaults to 100.

Example scoring payload:

```json
[
  {
    "CustomerType": "returning_visitor",
    "SpecialDayProximity": 0.0,
    "ExitRate": 0.12,
    "PageValue": 8.4,
    "TrafficSource": 2,
    "GeographicRegion": 1,
    "BounceRate": 0.04,
    "ProductPageTime": 520.0
  }
]
```

The response includes score, binary decision, threshold, model name, and model version.

## Docker

```bash
docker build -t purchase-intent-api .
docker run --rm -p 8000:8000 -v "$PWD/outputs:/app/outputs:ro" purchase-intent-api
```

The container runs as a non-root user and includes a liveness health check.

## Monitoring

The project writes lightweight production monitoring artifacts for:

- missingness and invalid-range data-quality alerts
- numeric and categorical drift
- weekly accuracy / precision / recall / F1 summaries
- targeted vs non-targeted conversion behavior

These are local reference implementations rather than a replacement for managed observability, alert routing, or feature-store monitoring.

## Tests

```bash
pytest -q
```

The existing suite covers data loading, preprocessing, evaluation, inference, monitoring, and training helpers.

## Repository structure

```text
.
├── app.py
├── Dockerfile
├── README.md
├── requirements.txt
├── eda.ipynb
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── monitoring.py
│   ├── preprocess.py
│   └── train_model.py
├── tests/
└── outputs/
```

## Production hardening layer

The service now adds API-key authentication, SHA-256 model verification, JSON request logs, request IDs, Prometheus metrics, environment-controlled documentation, immutable deployment manifests, CI container builds, Kubernetes probes/autoscaling/disruption controls, and production runbook/security guidance.

See `docs/production_runbook.md`, `deploy/kubernetes.yaml`, `.env.example`, and `SECURITY.md`.

## Production boundary

This is intentionally close to the **application and MLOps layer** of a real production purchase-intent service. It is still not a live enterprise production system until it runs on real traffic with real campaign decisions, delayed conversion labels, cloud IAM/secrets, centralized telemetry, a managed model registry, alert routing, canary/rollback execution, and operational ownership.
