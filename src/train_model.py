"""End-to-end ML pipeline for purchase intent prediction."""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import load_raw_data
from src.preprocess import build_preprocessor, domain_clean, get_feature_types, split_features_target
from src.evaluate import evaluate_predictions


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline."""
    data_path: Path = Path('online_shopping')
    target_col: str = 'PurchaseCompleted'
    test_size: float = 0.2
    val_size_within_trainval: float = 0.25
    random_state: int = 42
    threshold_grid_start: float = 0.10
    threshold_grid_stop: float = 0.91
    threshold_grid_step: float = 0.05
    output_dir: Path = Path('outputs')


def get_models(random_state: int) -> Dict[str, object]:
    """Initialize ML models."""
    return {
        'logreg_baseline': LogisticRegression(max_iter=2000, random_state=random_state),
        'gb_baseline': GradientBoostingClassifier(random_state=random_state),
        'rf_baseline': RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def fit_and_score_model(name: str, model, preprocessor, X_fit, y_fit, X_eval, y_eval, threshold: float = 0.5):
    """Train model and evaluate on validation set."""
    clf = Pipeline([
        ('preprocess', preprocessor),
        ('model', model),
    ])
    clf.fit(X_fit, y_fit)
    proba = clf.predict_proba(X_eval)[:, 1]
    metrics = evaluate_predictions(y_eval, proba, threshold=threshold)
    metrics.update({'model': name, 'threshold': threshold})
    return clf, metrics, proba


def tune_threshold_for_f1(y_true, proba, start: float, stop: float, step: float):
    """Search for best threshold that maximizes F1 on validation set."""
    rows = []
    threshold_grid = np.arange(start, stop, step)
    for t in threshold_grid:
        m = evaluate_predictions(y_true, proba, threshold=float(t))
        rows.append({
            'threshold': float(t),
            'f1': m['f1'],
            'precision': m['precision'],
            'recall': m['recall'],
            'accuracy': m['accuracy'],
        })

    th_df = pd.DataFrame(rows).sort_values('f1', ascending=False).reset_index(drop=True)
    best_t = float(th_df.iloc[0]['threshold'])
    return best_t, th_df


def get_feature_importance(final_clf: Pipeline, num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """Extract feature importance from trained model."""
    model = final_clf.named_steps['model']
    if not hasattr(model, 'feature_importances_'):
        return pd.DataFrame(columns=['feature', 'importance'])

    pre = final_clf.named_steps['preprocess']
    ohe = pre.named_transformers_['cat'].named_steps['ohe']
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist() if len(cat_cols) else []
    feature_names = num_cols + cat_feature_names

    return pd.DataFrame(
        {'feature': feature_names, 'importance': model.feature_importances_}
    ).sort_values('importance', ascending=False).reset_index(drop=True)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run end-to-end purchase intent ML pipeline.')
    parser.add_argument('--data-path', type=str, default='online_shopping', help='Path to CSV-formatted dataset file.')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save pipeline artifacts.')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for split/model reproducibility.')
    return parser.parse_args()


def run(config: PipelineConfig):
    """Execute the full ML pipeline."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(config.data_path)
    data = domain_clean(raw)
    X, y = split_features_target(data, config.target_col)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
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

    val_records = []
    trained_models = {}
    validation_probabilities = {}

    for name, model in models.items():
        clf, metrics, val_proba = fit_and_score_model(
            name=name,
            model=model,
            preprocessor=preprocessor,
            X_fit=X_train,
            y_fit=y_train,
            X_eval=X_val,
            y_eval=y_val,
            threshold=0.5,
        )
        trained_models[name] = clf
        validation_probabilities[name] = val_proba
        val_records.append(metrics)

    val_df = pd.DataFrame(val_records).sort_values('auc', ascending=False).reset_index(drop=True)
    best_model_name = val_df.iloc[0]['model']

    best_threshold, threshold_df = tune_threshold_for_f1(
        y_true=y_val,
        proba=validation_probabilities[best_model_name],
        start=config.threshold_grid_start,
        stop=config.threshold_grid_stop,
        step=config.threshold_grid_step,
    )

    final_model = models[best_model_name]
    final_clf = Pipeline([
        ('preprocess', preprocessor),
        ('model', final_model),
    ])
    final_clf.fit(X_trainval, y_trainval)
    test_proba = final_clf.predict_proba(X_test)[:, 1]

    test_default = evaluate_predictions(y_test, test_proba, threshold=0.5)
    test_tuned = evaluate_predictions(y_test, test_proba, threshold=best_threshold)

    feature_importance_df = get_feature_importance(final_clf, num_cols=num_cols, cat_cols=cat_cols)

    val_df.to_csv(config.output_dir / 'validation_metrics.csv', index=False)
    threshold_df.to_csv(config.output_dir / 'threshold_tuning.csv', index=False)
    feature_importance_df.to_csv(config.output_dir / 'feature_importance.csv', index=False)

    summary = {
        'data_shape': {'rows': int(data.shape[0]), 'columns': int(data.shape[1])},
        'split_shape': {
            'train_rows': int(X_train.shape[0]),
            'val_rows': int(X_val.shape[0]),
            'test_rows': int(X_test.shape[0]),
        },
        'feature_types': {'categorical': cat_cols, 'numerical': num_cols},
        'best_model': best_model_name,
        'best_threshold_by_f1': best_threshold,
        'test_default_0.50': test_default,
        'test_tuned': test_tuned,
    }

    with open(config.output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Pipeline completed successfully.')
    print(f'Best model: {best_model_name}')
    print(f'Best threshold: {best_threshold:.2f}')
    print('Artifacts written to:', config.output_dir)


def main():
    """Main entry point."""
    args = parse_args()
    cfg = PipelineConfig(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
    )
    run(cfg)


if __name__ == '__main__':
    main()
