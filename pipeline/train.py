"""
Reproducible LUAD/LUSC classifier.

Trains logistic regression and XGBoost on the processed expression matrix.
Feature selection uses either the limma DE results ranking (adj.P.Val ascending)
or the top-N genes by cross-sample variance.
Evaluates with stratified CV, computes SHAP values, and persists all artifacts.

Usage:
    python pipeline/train.py [--n-features N] [--feature-selection {de,variance}]
                             [--seed SEED]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier

from app.core.config import settings

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DE_RESULTS_PATH = PROJECT_ROOT / "r_analysis" / "output" / "de_results.csv"
MODELS_DIR = PROJECT_ROOT / "models"

N_FEATURES_DEFAULT = 500
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# LUAD=0, LUSC=1 (alphabetical; fixed across every run)
_LABEL_MAP = {"LUAD": 0, "LUSC": 1}


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def select_features(
    expr: pd.DataFrame,
    n_features: int,
    method: str,
    de_results: pd.DataFrame | None = None,
) -> list[str]:
    """Return up to n_features gene names selected by method.

    method="de":       genes ranked by adj.P.Val ascending (most significant first),
                       restricted to genes present in expr columns.
    method="variance": genes ranked by cross-sample variance descending.
    """
    if method == "de":
        if de_results is None:
            raise ValueError("de_results DataFrame required for method='de'")
        ranked = de_results.loc[de_results["gene_symbol"].isin(expr.columns), "gene_symbol"]
        return ranked.tolist()[:n_features]
    if method == "variance":
        gene_var = expr.var(axis=0).sort_values(ascending=False)
        return gene_var.index.tolist()[:n_features]
    raise ValueError(f"Unknown feature_selection method: {method!r}; expected 'de' or 'variance'")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _cv_metrics(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> dict:
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }
    raw = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return {
        metric: {
            "mean": float(np.mean(raw[f"test_{metric}"])),
            "std": float(np.std(raw[f"test_{metric}"])),
        }
        for metric in scoring
    }


def _test_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def _shap_ranking(
    xgb_model: XGBClassifier,
    X_test: np.ndarray,
    gene_names: list[str],
    model_version: str,
) -> pd.DataFrame:
    """Return genes ranked by mean absolute SHAP value (rank 1 = most discriminative)."""
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_test, check_additivity=False)
    # Normalize across SHAP API versions: list -> use positive class; 3-D -> slice last axis
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-mean_abs)
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(gene_names) + 1),
            "gene_symbol": np.array(gene_names)[order],
            "mean_shap": mean_abs[order],
            "model_version": model_version,
        }
    )


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


def _persist_top_genes_db(engine, shap_df: pd.DataFrame) -> None:
    rows = shap_df[["rank", "gene_symbol", "mean_shap", "model_version"]].to_dict("records")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE top_genes"))
        conn.execute(
            text(
                "INSERT INTO top_genes (rank, gene_symbol, mean_shap, model_version)"
                " VALUES (:rank, :gene_symbol, :mean_shap, :model_version)"
            ),
            rows,
        )
    log.info("Loaded %d rows into top_genes", len(rows))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_train(
    processed_dir: Path = PROCESSED_DIR,
    de_results_path: Path = DE_RESULTS_PATH,
    models_dir: Path = MODELS_DIR,
    n_features: int = N_FEATURES_DEFAULT,
    feature_selection: str = "de",
    random_seed: int = RANDOM_SEED,
    cv_folds: int = CV_FOLDS,
    skip_db: bool = False,
    database_url: str | None = None,
) -> str:
    """Run the full training pipeline. Returns the model version string."""
    random.seed(random_seed)
    np.random.seed(random_seed)

    version = f"v{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    version_dir = models_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    log.info("Model version: %s", version)

    # 1. Load data
    log.info("Loading expression matrix...")
    expr = pd.read_parquet(processed_dir / "expression_matrix.parquet")
    log.info("  %d samples x %d genes", *expr.shape)

    meta = pd.read_parquet(processed_dir / "sample_metadata.parquet")
    common = expr.index.intersection(meta.index)
    expr = expr.loc[common]
    meta = meta.loc[common]

    y = np.array([_LABEL_MAP[c] for c in meta["cohort"].values])
    log.info("Labels: %d LUAD, %d LUSC", (y == 0).sum(), (y == 1).sum())

    # 2. Feature selection (before split; uses DE results from separate analysis)
    de_results: pd.DataFrame | None = None
    if feature_selection == "de":
        de_results = pd.read_csv(de_results_path)
    genes = select_features(expr, n_features, feature_selection, de_results)
    log.info("Selected %d features via %s", len(genes), feature_selection)

    X = expr[genes].values.astype(np.float64)
    # Impute residual NaN with column mean (rare after ETL variance filter)
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # 3. Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_seed
    )
    log.info("Split: %d train / %d test", len(y_train), len(y_test))

    # 4. Standardize; scaler fitted on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5. Estimators
    lr = LogisticRegression(max_iter=1000, random_state=random_seed)
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        eval_metric="logloss",
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    metrics: dict = {
        "version": version,
        "n_features": len(genes),
        "feature_selection": feature_selection,
        "random_seed": random_seed,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "label_map": _LABEL_MAP,
    }

    # 6. CV + final fit for each model
    for name, model in [("logistic_regression", lr), ("xgboost", xgb)]:
        log.info("Cross-validating %s (%d folds)...", name, cv_folds)
        cv_summary = _cv_metrics(model, X_train_s, y_train, cv)
        log.info(
            "  CV  accuracy=%.4f+/-%.4f  roc_auc=%.4f+/-%.4f",
            cv_summary["accuracy"]["mean"],
            cv_summary["accuracy"]["std"],
            cv_summary["roc_auc"]["mean"],
            cv_summary["roc_auc"]["std"],
        )
        model.fit(X_train_s, y_train)
        test_m = _test_metrics(model, X_test_s, y_test)
        log.info(
            "  Test accuracy=%.4f  roc_auc=%.4f  f1=%.4f",
            test_m["accuracy"],
            test_m["roc_auc"],
            test_m["f1"],
        )
        metrics[name] = {"cv": cv_summary, "test": test_m}

    # 7. SHAP values on XGBoost (test set, scaled space)
    log.info("Computing SHAP values...")
    shap_df = _shap_ranking(xgb, X_test_s, genes, version)
    log.info("Top 5 genes by SHAP: %s", shap_df["gene_symbol"].head(5).tolist())

    # 8. Persist artifacts
    joblib.dump(scaler, version_dir / "scaler.joblib")
    joblib.dump(lr, version_dir / "lr.joblib")
    joblib.dump(xgb, version_dir / "xgb.joblib")
    (version_dir / "feature_names.json").write_text(json.dumps(genes))
    (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shap_df.to_csv(version_dir / "top_genes.csv", index=False)
    log.info("Artifacts saved to %s", version_dir)

    # 9. DB: persist top_genes (skipped in unit tests via skip_db=True)
    if not skip_db:
        url = database_url or settings.database_url
        engine = create_engine(url, pool_pre_ping=True)
        _persist_top_genes_db(engine, shap_df)

    return version


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LUAD/LUSC classifier")
    p.add_argument("--n-features", type=int, default=N_FEATURES_DEFAULT, metavar="N")
    p.add_argument(
        "--feature-selection",
        choices=["de", "variance"],
        default="de",
        metavar="METHOD",
    )
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run_train(
        n_features=args.n_features,
        feature_selection=args.feature_selection,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
