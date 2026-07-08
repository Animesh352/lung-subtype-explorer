"""
Tests for pipeline/train.py.

All tests run on a tiny synthetic dataset (60 samples x 100 genes) so they
complete in seconds without touching the real data or Postgres.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.train import run_train, select_features

# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> dict:
    rng = np.random.default_rng(0)
    n_luad, n_lusc = 30, 30
    n_genes = 100
    genes = [f"GENE_{i:04d}" for i in range(n_genes)]
    sample_ids = [f"TCGA-XX-{i:04d}-01" for i in range(n_luad + n_lusc)]

    # Clearly separable classes: LUAD ~ N(0, 0.5), LUSC ~ N(3, 0.5)
    data = np.vstack(
        [
            rng.normal(0.0, 0.5, (n_luad, n_genes)),
            rng.normal(3.0, 0.5, (n_lusc, n_genes)),
        ]
    ).astype(np.float32)

    expr_df = pd.DataFrame(data, columns=genes, index=pd.Index(sample_ids, name="sample_id"))
    meta_df = pd.DataFrame(
        {
            "cohort": ["LUAD"] * n_luad + ["LUSC"] * n_lusc,
            "sample_type": ["primary_tumor"] * (n_luad + n_lusc),
        },
        index=pd.Index(sample_ids, name="sample_id"),
    )
    de_df = pd.DataFrame(
        {
            "gene_symbol": genes,
            "logFC": rng.normal(0, 1, n_genes),
            "AveExpr": rng.normal(5, 1, n_genes),
            "t": rng.normal(0, 5, n_genes),
            "P.Value": rng.uniform(0, 1, n_genes),
            # adj.P.Val sorted ascending so DE ranking is well-defined
            "adj.P.Val": np.sort(rng.uniform(0, 1, n_genes)),
        }
    )

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    expr_df.to_parquet(processed_dir / "expression_matrix.parquet")
    meta_df.to_parquet(processed_dir / "sample_metadata.parquet")

    de_path = tmp_path / "de_results.csv"
    de_df.to_csv(de_path, index=False)

    return {
        "processed_dir": processed_dir,
        "de_results_path": de_path,
        "models_dir": tmp_path / "models",
    }


# Shared helper to call run_train with test-friendly defaults
def _run(d: dict, n_features: int = 50, feature_selection: str = "de", seed: int = 42) -> str:
    return run_train(
        processed_dir=d["processed_dir"],
        de_results_path=d["de_results_path"],
        models_dir=d["models_dir"],
        n_features=n_features,
        feature_selection=feature_selection,
        random_seed=seed,
        cv_folds=3,
        skip_db=True,
    )


# ---------------------------------------------------------------------------
# Artifact tests
# ---------------------------------------------------------------------------


def test_all_artifacts_produced(tiny_dataset: dict) -> None:
    version = _run(tiny_dataset)
    vdir = tiny_dataset["models_dir"] / version
    for fname in (
        "scaler.joblib",
        "lr.joblib",
        "xgb.joblib",
        "feature_names.json",
        "metrics.json",
        "top_genes.csv",
    ):
        assert (vdir / fname).exists(), f"{fname} missing from {vdir}"


def test_feature_names_length(tiny_dataset: dict) -> None:
    n = 30
    version = _run(tiny_dataset, n_features=n)
    vdir = tiny_dataset["models_dir"] / version
    names = json.loads((vdir / "feature_names.json").read_text())
    assert len(names) == n


def test_variance_selection_length(tiny_dataset: dict) -> None:
    n = 40
    version = _run(tiny_dataset, n_features=n, feature_selection="variance")
    vdir = tiny_dataset["models_dir"] / version
    names = json.loads((vdir / "feature_names.json").read_text())
    assert len(names) == n


# ---------------------------------------------------------------------------
# Metrics structure tests
# ---------------------------------------------------------------------------


def test_metrics_keys_present(tiny_dataset: dict) -> None:
    version = _run(tiny_dataset)
    metrics = json.loads((tiny_dataset["models_dir"] / version / "metrics.json").read_text())

    assert metrics["n_features"] == 50
    assert metrics["feature_selection"] == "de"

    for model_key in ("logistic_regression", "xgboost"):
        assert model_key in metrics
        cv = metrics[model_key]["cv"]
        test = metrics[model_key]["test"]
        for metric in ("accuracy", "roc_auc", "f1", "precision", "recall"):
            assert metric in cv, f"cv.{metric} missing for {model_key}"
            assert "mean" in cv[metric] and "std" in cv[metric]
            assert metric in test, f"test.{metric} missing for {model_key}"
        assert "confusion_matrix" in test


def test_metrics_reasonable(tiny_dataset: dict) -> None:
    version = _run(tiny_dataset)
    metrics = json.loads((tiny_dataset["models_dir"] / version / "metrics.json").read_text())
    # Classes are clearly separable -> expect high accuracy
    for model_key in ("logistic_regression", "xgboost"):
        assert metrics[model_key]["test"]["accuracy"] >= 0.8
        auc = metrics[model_key]["cv"]["roc_auc"]["mean"]
        assert 0.0 <= auc <= 1.0


def test_confusion_matrix_shape(tiny_dataset: dict) -> None:
    version = _run(tiny_dataset)
    metrics = json.loads((tiny_dataset["models_dir"] / version / "metrics.json").read_text())
    cm = metrics["xgboost"]["test"]["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2


def test_n_train_n_test_sum_to_total(tiny_dataset: dict) -> None:
    version = _run(tiny_dataset)
    metrics = json.loads((tiny_dataset["models_dir"] / version / "metrics.json").read_text())
    assert metrics["n_train"] + metrics["n_test"] == 60


# ---------------------------------------------------------------------------
# SHAP / top_genes tests
# ---------------------------------------------------------------------------


def test_top_genes_columns(tiny_dataset: dict) -> None:
    version = _run(tiny_dataset)
    tg = pd.read_csv(tiny_dataset["models_dir"] / version / "top_genes.csv")
    assert {"rank", "gene_symbol", "mean_shap", "model_version"}.issubset(tg.columns)


def test_top_genes_ranks_contiguous(tiny_dataset: dict) -> None:
    version = _run(tiny_dataset)
    tg = pd.read_csv(tiny_dataset["models_dir"] / version / "top_genes.csv")
    assert list(tg["rank"]) == list(range(1, len(tg) + 1))


def test_top_genes_shap_descending(tiny_dataset: dict) -> None:
    version = _run(tiny_dataset)
    tg = pd.read_csv(tiny_dataset["models_dir"] / version / "top_genes.csv")
    assert tg["mean_shap"].is_monotonic_decreasing


def test_top_genes_version_matches(tiny_dataset: dict) -> None:
    version = _run(tiny_dataset)
    tg = pd.read_csv(tiny_dataset["models_dir"] / version / "top_genes.csv")
    assert (tg["model_version"] == version).all()


# ---------------------------------------------------------------------------
# select_features unit tests
# ---------------------------------------------------------------------------


def test_select_features_de(tiny_dataset: dict) -> None:
    expr = pd.read_parquet(tiny_dataset["processed_dir"] / "expression_matrix.parquet")
    de = pd.read_csv(tiny_dataset["de_results_path"])
    genes = select_features(expr, 20, "de", de)
    assert len(genes) == 20
    assert all(g in expr.columns for g in genes)


def test_select_features_variance(tiny_dataset: dict) -> None:
    expr = pd.read_parquet(tiny_dataset["processed_dir"] / "expression_matrix.parquet")
    genes = select_features(expr, 25, "variance")
    assert len(genes) == 25
    assert all(g in expr.columns for g in genes)
    # Verify ordering: each selected gene has variance >= the next
    variances = expr[genes].var(axis=0).values
    assert all(variances[i] >= variances[i + 1] for i in range(len(variances) - 1))


def test_select_features_de_requires_de_results(tiny_dataset: dict) -> None:
    expr = pd.read_parquet(tiny_dataset["processed_dir"] / "expression_matrix.parquet")
    with pytest.raises(ValueError, match="de_results"):
        select_features(expr, 10, "de", de_results=None)


def test_select_features_unknown_method(tiny_dataset: dict) -> None:
    expr = pd.read_parquet(tiny_dataset["processed_dir"] / "expression_matrix.parquet")
    with pytest.raises(ValueError, match="Unknown"):
        select_features(expr, 10, "pca")


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


def test_deterministic_across_runs(tiny_dataset: dict, tmp_path: Path) -> None:
    models_a = tmp_path / "models_a"
    models_b = tmp_path / "models_b"
    v1 = run_train(
        processed_dir=tiny_dataset["processed_dir"],
        de_results_path=tiny_dataset["de_results_path"],
        models_dir=models_a,
        n_features=30,
        random_seed=7,
        cv_folds=3,
        skip_db=True,
    )
    v2 = run_train(
        processed_dir=tiny_dataset["processed_dir"],
        de_results_path=tiny_dataset["de_results_path"],
        models_dir=models_b,
        n_features=30,
        random_seed=7,
        cv_folds=3,
        skip_db=True,
    )
    m1 = json.loads((models_a / v1 / "metrics.json").read_text())
    m2 = json.loads((models_b / v2 / "metrics.json").read_text())
    assert m1["xgboost"]["test"]["accuracy"] == m2["xgboost"]["test"]["accuracy"]
    assert (
        m1["logistic_regression"]["test"]["roc_auc"] == m2["logistic_regression"]["test"]["roc_auc"]
    )
