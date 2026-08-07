# Online Shopping Purchase Intent ML

An end-to-end machine learning pipeline for predicting customer purchase intent from online session behavior data.

## Overview

This project builds and evaluates a binary classification pipeline for purchase intent prediction. The workflow covers data loading, domain-aware cleaning, preprocessing, model selection, threshold tuning, and final evaluation on a held-out test set.

The pipeline is designed for imbalanced classification, so it tracks both ranking quality and business-facing classification metrics:

- AUC
- F1 score
- Precision
- Recall
- Accuracy
- Confusion matrix

It also now includes a lightweight monitoring layer for drift, data quality, weekly performance reporting, and a simple business-impact summary.

## Project Structure

```text
online_shopping_purchase_intent_ml/
|-- README.md
|-- requirements.txt
|-- online_shopping_purchase_intent_ml.ipynb
|-- outputs/
|   |-- feature_importance.csv
|   |-- summary.json
|   |-- threshold_tuning.csv
|   `-- validation_metrics.csv
|-- tests/
|   |-- conftest.py
|   |-- test_data_loader.py
|   |-- test_evaluate.py
|   |-- test_preprocess.py
|   `-- test_train_model.py
`-- src/
    |-- data_loader.py
    |-- evaluate.py
    |-- preprocess.py
    `-- train_model.py
```

## Pipeline Flow

1. Load the raw dataset.
2. Apply domain cleaning and prepare the target column.
3. Split data into train, validation, and test sets with stratification.
4. Build a preprocessing pipeline for numeric and categorical features.
5. Train multiple baseline models.
6. Select the best model by validation AUC.
7. Tune the classification threshold for the best validation F1 score.
8. Retrain on train+validation data and evaluate on the untouched test set.
9. Save metrics and artifacts into the `outputs/` directory.

## Models

The training script currently compares these classifiers:

- Logistic Regression
- LightGBM Classifier
- Random Forest Classifier

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full training and evaluation pipeline:

```bash
python -m src.train_model --data-path online_shopping.csv --output-dir outputs --random-state 42 --mlflow-tracking-uri file:./mlruns --experiment-name purchase-intent-lightgbm
```

Arguments:

- `--data-path`: path to the input CSV dataset
- `--output-dir`: directory for generated metrics and artifacts
- `--random-state`: random seed for reproducibility
- `--mlflow-tracking-uri`: location for MLflow experiment tracking
- `--experiment-name`: MLflow experiment name

## Tests

Run the test suite with:

```bash
pytest
```

The tests cover data loading, preprocessing, evaluation, selected training helpers, and the new inference API scaffolding.

## Production-Ready Starter Components

This repository now includes a lightweight production-ready foundation:

- A prediction API entry point in [app.py](app.py)
- A reusable inference module in [src/inference.py](src/inference.py)
- A Docker image in [Dockerfile](Dockerfile)
- A CI workflow in [.github/workflows/ci.yml](.github/workflows/ci.yml)

### Run the API locally

```bash
uvicorn app:app --reload
```

Then open the Swagger docs at:

```text
http://127.0.0.1:8000/docs
```

### Build the Docker image

```bash
docker build -t purchase-intent-api .
```

### Run the Docker container

```bash
docker run -p 8000:8000 purchase-intent-api
```

`tests/conftest.py` is included so `pytest` can import the top-level `src` package reliably in this repo layout.

`__pycache__/` directories are generated automatically by Python. They are not source files and should not be committed.

## Outputs

After a successful run, the pipeline writes these files to `outputs/`:

- `validation_metrics.csv`: validation performance for each candidate model
- `threshold_tuning.csv`: threshold search results for the selected model
- `feature_importance.csv`: feature importances when supported by the final model
- `summary.json`: run summary, split sizes, selected model, threshold, and final test metrics
- `weekly_performance_summary.csv`: simple weekly performance summary table
- `data_quality_alerts.json`: basic missingness and invalid-range alerts
- `drift_report.json`: simple drift summary for numeric and categorical features
- `business_impact.csv`: simple business-impact view showing targeted vs non-targeted conversion behavior

## Reported Performance

The pipeline now uses LightGBM as the boosted-tree baseline. The latest run is summarized below.

### Compared with the previous Gradient Boosting baseline

- Previous best Gradient Boosting baseline: AUC 0.8891, F1 0.5982, Precision 0.7071, Recall 0.5183, Accuracy 0.8921
- New LightGBM-based pipeline: AUC 0.9253, F1 0.6933, Precision 0.7986, Recall 0.6126, Accuracy 0.9161 at the default threshold of 0.50

### Default threshold: 0.50

- AUC: 0.9253
- F1: 0.6933
- Precision: 0.7986
- Recall: 0.6126
- Accuracy: 0.9161

### Tuned threshold: 0.35

- AUC: 0.9253
- F1: 0.6799
- Precision: 0.6900
- Recall: 0.6702
- Accuracy: 0.9023

The tuned threshold improves recall and F1, which is often preferable in imbalanced purchase-intent classification.

## Monitoring and Business Evaluation

The project now includes a lightweight monitoring layer with the following first-version checks:

- Drift checks for key numerical and categorical features such as `PageValue`, `BounceRate`, `ExitRate`, and `CustomerType`
- Basic data quality alerts for missing values and out-of-range bounded features
- A simple weekly performance summary table for tracking accuracy, precision, recall, and F1
- A business-impact table that summarizes targeted versus non-targeted conversion rates

These outputs are designed as a practical starting point for ongoing monitoring and business evaluation rather than a full production observability stack.

## Deployment Considerations

- Validate the incoming scoring schema so feature names, data types, and categorical formats match the training pipeline.
- Enforce domain checks on bounded fields such as `BounceRate`, `ExitRate`, and `SpecialDayProximity` before inference.
- Reuse the same preprocessing logic in production to avoid training-serving skew.
- Monitor null rates and unexpected category values for `CustomerType`, `TrafficSource`, and `GeographicRegion`.
- Track drift in high-impact numerical features such as `PageValue` and `ProductPageTime`.
- Choose the operating threshold based on business goals: the default threshold favors precision, while the tuned threshold improves recall.
- Log prediction scores, final class decisions, and downstream outcomes so model quality can be recalibrated over time.
- Retrain and retune the threshold periodically when traffic patterns, customer behavior, or campaign sources change.

## Notes

- Missing values are handled during loading and preprocessing.
- Threshold tuning is based on validation-set F1.
- The final evaluation is performed on a held-out test split.
- `online_shopping_purchase_intent_ml.ipynb` contains exploratory analysis and supporting observations.
