from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.services.model import load_latest_model


def _write_artifacts(models_dir: Path, version: str = "v1", n_features: int = 3) -> list[str]:
    genes = [f"G{i}" for i in range(n_features)]
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal(0.0, 0.5, (20, n_features)),
            rng.normal(3.0, 0.5, (20, n_features)),
        ]
    )
    y = np.array([0] * 20 + [1] * 20)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    xgb = XGBClassifier(n_estimators=3, random_state=0, eval_metric="logloss", verbosity=0)
    xgb.fit(X_s, y)

    vdir = models_dir / version
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "feature_names.json").write_text(json.dumps(genes))
    joblib.dump(scaler, vdir / "scaler.joblib")
    joblib.dump(xgb, vdir / "xgb.joblib")
    return genes


# ---------------------------------------------------------------------------
# load_latest_model
# ---------------------------------------------------------------------------


def test_load_missing_dir(tmp_path: Path) -> None:
    result = load_latest_model(tmp_path / "does_not_exist")
    assert result is None


def test_load_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("placeholder")
    result = load_latest_model(tmp_path)
    assert result is None


def test_load_valid_model(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, version="v1")
    store = load_latest_model(tmp_path)
    assert store is not None
    assert store.version == "v1"
    assert len(store.feature_names) == 3


def test_load_picks_newest_version(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, version="v1")
    _write_artifacts(tmp_path, version="v2")
    store = load_latest_model(tmp_path)
    assert store is not None
    assert store.version == "v2"


def test_load_broken_joblib(tmp_path: Path) -> None:
    vdir = tmp_path / "v1"
    vdir.mkdir()
    (vdir / "feature_names.json").write_text('["G0"]')
    (vdir / "scaler.joblib").write_bytes(b"not a real file")
    (vdir / "xgb.joblib").write_bytes(b"not a real file")
    result = load_latest_model(tmp_path)
    assert result is None


def test_load_missing_feature_names(tmp_path: Path) -> None:
    _write_artifacts(tmp_path, version="v1")
    (tmp_path / "v1" / "feature_names.json").unlink()
    result = load_latest_model(tmp_path)
    assert result is None


def test_load_feature_names_match(tmp_path: Path) -> None:
    genes = _write_artifacts(tmp_path, version="v1", n_features=5)
    store = load_latest_model(tmp_path)
    assert store is not None
    assert store.feature_names == genes


# ---------------------------------------------------------------------------
# ModelStore.predict
# ---------------------------------------------------------------------------


def test_predict_returns_subtype_and_probs(tmp_path: Path) -> None:
    genes = _write_artifacts(tmp_path)
    store = load_latest_model(tmp_path)
    assert store is not None
    features = {g: 0.0 for g in genes}
    subtype, probs = store.predict(features)
    assert subtype in ("LUAD", "LUSC")
    assert set(probs.keys()) == {"LUAD", "LUSC"}


def test_predict_probs_sum_to_one(tmp_path: Path) -> None:
    genes = _write_artifacts(tmp_path)
    store = load_latest_model(tmp_path)
    assert store is not None
    _, probs = store.predict({g: 1.5 for g in genes})
    assert abs(probs["LUAD"] + probs["LUSC"] - 1.0) < 1e-6


def test_predict_luad_cluster(tmp_path: Path) -> None:
    genes = _write_artifacts(tmp_path)
    store = load_latest_model(tmp_path)
    assert store is not None
    # LUAD cluster was trained near 0
    subtype, _ = store.predict({g: 0.0 for g in genes})
    assert subtype == "LUAD"


def test_predict_lusc_cluster(tmp_path: Path) -> None:
    genes = _write_artifacts(tmp_path)
    store = load_latest_model(tmp_path)
    assert store is not None
    # LUSC cluster was trained near 3
    subtype, _ = store.predict({g: 3.0 for g in genes})
    assert subtype == "LUSC"


def test_predict_lusc_has_higher_lusc_prob(tmp_path: Path) -> None:
    genes = _write_artifacts(tmp_path)
    store = load_latest_model(tmp_path)
    assert store is not None
    subtype, probs = store.predict({g: 3.0 for g in genes})
    assert subtype == "LUSC"
    assert probs["LUSC"] > probs["LUAD"]


def test_predict_ignores_dict_ordering(tmp_path: Path) -> None:
    genes = _write_artifacts(tmp_path)
    store = load_latest_model(tmp_path)
    assert store is not None
    # predict() always reads features in feature_names order,
    # so dict insertion order does not change the result
    vals = {g: float(i) for i, g in enumerate(genes)}
    vals_rev = dict(reversed(list(vals.items())))
    _, probs1 = store.predict(vals)
    _, probs2 = store.predict(vals_rev)
    assert abs(probs1["LUAD"] - probs2["LUAD"]) < 1e-9
