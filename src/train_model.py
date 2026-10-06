"""End-to-end ML pipeline for purchase-intent prediction and deployable artifact export."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import load_raw_data
from src.evaluate import evaluate_predictions
from src.monitoring import (
    build_business_impact_table,
    build_data_quality_alerts,
    build_drift_report,
    build_weekly_performance_summary,
)
from src.preprocess import build_preprocessor, domain_clean, get_feature_types, split_features_target


@dataclass
class PipelineConfig:
    data_path: Path = Path("online_shopping.csv")
    target_col: str = "PurchaseCompleted"
    test_size: float = 0.2
    val_size_within_trainval: float = 0.25
    random_state: int = 42
    threshold_grid_start: float = 0.10
    threshold_grid_stop: float = 0.91
    threshold_grid_step: float = 0.05
    output_dir: Path = Path("outputs")
    mlflow_tracking_uri: str = "file:./mlruns_purchase_intent"
    experiment_name: str = "purchase-intent-lightgbm"


def get_models(random_state: int) -> Dict[str, object]:
    return {
        "logreg_baseline": LogisticRegression(max_iter=2000, random_state=random_state),
        "lgbm_baseline": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
        "rf_baseline": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def get_model_display_name(model_name: str) -> str:
    return {
        "logreg_baseline": "Logistic Regression",
        "lgbm_baseline": "LightGBM",
        "rf_baseline": "Random Forest",
    }.get(model_name, model_name)


def build_model_selection_reason(best_model_name: str, val_df: pd.DataFrame) -> str:
    best_row = val_df[val_df["model"] == best_model_name].iloc[0]
    return (
        f"{get_model_display_name(best_model_name)} was selected by validation AUC "
        f"({best_row['auc']:.4f}); the operating threshold was tuned separately on the validation split."
    )


def fit_and_score_model(name: str, model, preprocessor, X_fit, y_fit, X_eval, y_eval, threshold: float = 0.5):
    clf = Pipeline([("preprocess", preprocessor), ("model", model)])
    clf.fit(X_fit, y_fit)
    proba = clf.predict_proba(X_eval)[:, 1]
    metrics = evaluate_predictions(y_eval, proba, threshold=threshold)
    metrics.update({"model": name, "threshold": threshold})
    return clf, metrics, proba


def tune_threshold_for_f1(y_true, proba, start: float, stop: float, step: float):
    rows = []
    for threshold in np.arange(start, stop, step):
        metrics = evaluate_predictions(y_true, proba, threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "accuracy": metrics["accuracy"],
            }
        )
    threshold_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return float(threshold_df.iloc[0]["threshold"]), threshold_df


def get_feature_importance(final_clf: Pipeline, num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    model = final_clf.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    preprocessor = final_clf.named_steps["preprocess"]
    categorical_names: list[str] = []
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        categorical_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + categorical_names
    return pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)


def make_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def make_readable_run_name(model_name: str, threshold: float, random_state: int) -> str:
    return f"{model_name}-threshold-{threshold:.2f}-seed-{random_state}"


def resolve_artifact_location(tracking_uri: str, experiment_name: str) -> str:
    tracking_path = Path(tracking_uri.replace("file:", "", 1)).resolve()
    artifact_dir = tracking_path / "artifacts" / make_slug(experiment_name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return f"file:{artifact_dir}"


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run end-to-end purchase intent ML pipeline.")
    parser.add_argument("--data-path", default="online_shopping.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns_purchase_intent")
    parser.add_argument("--experiment-name", default="purchase-intent-lightgbm")
    return parser.parse_args(args)


def _save_serving_bundle(
    final_clf: Pipeline,
    best_model_name: str,
    best_threshold: float,
    feature_columns: list[str],
    config: PipelineConfig,
    test_metrics: dict,
) -> dict:
    created_at = datetime.now(timezone.utc)
    model_version = f"{best_model_name}-{created_at.strftime('%Y%m%dT%H%M%SZ')}"
    bundle = {
        "model": final_clf,
        "model_name": best_model_name,
        "model_display_name": get_model_display_name(best_model_name),
        "model_version": model_version,
        "created_at_utc": created_at.isoformat(),
        "threshold": float(best_threshold),
        "feature_columns": feature_columns,
        "target_column": config.target_col,
        "random_state": config.random_state,
        "test_metrics": test_metrics,
    }
    joblib.dump(bundle, config.output_dir / "model_bundle.joblib")
    return bundle


def run(config: PipelineConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            config.experiment_name,
            artifact_location=resolve_artifact_location(config.mlflow_tracking_uri, config.experiment_name),
        )
    mlflow.set_experiment(config.experiment_name)

    raw = load_raw_data(config.data_path)
    data = domain_clean(raw)
    X, y = split_features_target(data, config.target_col)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=config.val_size_within_trainval,
        random_state=config.random_state,
        stratify=y_trainval,
    )

    cat_cols, num_cols = get_feature_types(X_train)
    preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
    models = get_models(config.random_state)

    validation_rows = []
    validation_probabilities = {}
    for name, model in models.items():
        _, metrics, probabilities = fit_and_score_model(
            name, model, preprocessor, X_train, y_train, X_val, y_val, threshold=0.5
        )
        validation_rows.append(metrics)
        validation_probabilities[name] = probabilities

    val_df = pd.DataFrame(validation_rows).sort_values("auc", ascending=False).reset_index(drop=True)
    best_model_name = str(val_df.iloc[0]["model"])
    best_threshold, threshold_df = tune_threshold_for_f1(
        y_val,
        validation_probabilities[best_model_name],
        config.threshold_grid_start,
        config.threshold_grid_stop,
        config.threshold_grid_step,
    )

    final_clf = Pipeline(
        [("preprocess", preprocessor), ("model", models[best_model_name])]
    )
    final_clf.fit(X_trainval, y_trainval)
    test_proba = final_clf.predict_proba(X_test)[:, 1]
    test_default = evaluate_predictions(y_test, test_proba, threshold=0.5)
    test_tuned = evaluate_predictions(y_test, test_proba, threshold=best_threshold)

    val_df.to_csv(config.output_dir / "validation_metrics.csv", index=False)
    threshold_df.to_csv(config.output_dir / "threshold_tuning.csv", index=False)
    get_feature_importance(final_clf, num_cols, cat_cols).to_csv(
        config.output_dir / "feature_importance.csv", index=False
    )

    bundle = _save_serving_bundle(
        final_clf,
        best_model_name,
        best_threshold,
        list(X.columns),
        config,
        test_tuned,
    )

    summary = {
        "data_shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "split_shape": {
            "train_rows": int(X_train.shape[0]),
            "val_rows": int(X_val.shape[0]),
            "test_rows": int(X_test.shape[0]),
        },
        "feature_types": {"categorical": cat_cols, "numerical": num_cols},
        "best_model": best_model_name,
        "best_model_name": get_model_display_name(best_model_name),
        "model_selection_reason": build_model_selection_reason(best_model_name, val_df),
        "best_threshold_by_f1": best_threshold,
        "test_default_0.50": test_default,
        "test_tuned": test_tuned,
        "serving_artifact": "model_bundle.joblib",
        "model_version": bundle["model_version"],
    }
    with (config.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    weekly_metrics = pd.DataFrame(
        [{"week": "current", **{key: float(test_tuned[key]) for key in ["accuracy", "precision", "recall", "f1"]}}]
    )
    build_weekly_performance_summary(weekly_metrics).to_csv(
        config.output_dir / "weekly_performance_summary.csv", index=False
    )
    quality_alerts = build_data_quality_alerts(data)
    pd.DataFrame(quality_alerts).to_json(
        config.output_dir / "data_quality_alerts.json", orient="records", indent=2
    )
    drift_report = build_drift_report(
        reference=data.iloc[: max(1, len(data) // 2)],
        current=data.iloc[max(1, len(data) // 2) :],
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )
    with (config.output_dir / "drift_report.json").open("w", encoding="utf-8") as handle:
        json.dump(drift_report, handle, indent=2)
    build_business_impact_table(
        predictions=pd.DataFrame(
            {"score": test_proba, "predicted_positive": test_proba >= best_threshold}
        ),
        labels=pd.Series(y_test.to_numpy(), index=y_test.index),
        threshold=best_threshold,
    ).to_csv(config.output_dir / "business_impact.csv", index=False)

    run_name = make_readable_run_name(best_model_name, best_threshold, config.random_state)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "random_state": config.random_state,
                "best_model": best_model_name,
                "threshold": best_threshold,
            }
        )
        mlflow.log_metrics(
            {
                "test_auc": float(test_default["auc"]),
                "test_f1_default": float(test_default["f1"]),
                "test_f1_tuned": float(test_tuned["f1"]),
            }
        )
        mlflow.log_artifact(str(config.output_dir / "summary.json"))
        mlflow.log_artifact(str(config.output_dir / "model_bundle.joblib"))
        mlflow.sklearn.log_model(final_clf, artifact_path="model")

    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    run(
        PipelineConfig(
            data_path=Path(args.data_path),
            output_dir=Path(args.output_dir),
            random_state=args.random_state,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.experiment_name,
        )
    )


if __name__ == "__main__":
    main()
