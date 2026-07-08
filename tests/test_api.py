"""
Integration tests for the REST API.

Uses an in-memory SQLite database seeded with deterministic test data.
The trained model and httpx annotation client are replaced with lightweight
fakes so tests run fully offline and independently of the real model files.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from xgboost import XGBClassifier

from app.api.deps import get_http_client
from app.db.session import Base, get_db
from app.main import app
from app.models.genomics import DEResult, Expression, Gene, GeneAnnotation, Sample, TopGene
from app.services.model import ModelStore

# ---------------------------------------------------------------------------
# Test gene/feature set (must match fake ModelStore)
# ---------------------------------------------------------------------------

_GENES = ["GENE_A", "GENE_B", "GENE_C"]


# ---------------------------------------------------------------------------
# Fake httpx client (returns empty MyGene.info hits for all requests)
# ---------------------------------------------------------------------------


class _FakeHttpxResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"hits": []}


class _FakeHttpxClient:
    def get(self, url: str, **kwargs) -> _FakeHttpxResponse:
        return _FakeHttpxResponse()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fake ModelStore
# ---------------------------------------------------------------------------


def _make_test_model() -> ModelStore:
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.normal([0.0, 0.1, 0.2], 0.3, size=(20, 3)),  # LUAD (label 0)
            rng.normal([2.0, 1.8, 1.5], 0.3, size=(20, 3)),  # LUSC (label 1)
        ]
    )
    y = np.array([0] * 20 + [1] * 20)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    xgb = XGBClassifier(n_estimators=5, random_state=0, eval_metric="logloss", verbosity=0)
    xgb.fit(X_s, y)

    return ModelStore(version="v_test", feature_names=_GENES, scaler=scaler, xgb=xgb)


# ---------------------------------------------------------------------------
# DB seeding
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _seed(session: Session) -> None:
    session.add_all(
        [
            Sample(sample_id="LUAD_T1", cohort="LUAD", sample_type="tumor"),
            Sample(sample_id="LUAD_T2", cohort="LUAD", sample_type="tumor"),
            Sample(sample_id="LUAD_N1", cohort="LUAD", sample_type="normal"),
            Sample(sample_id="LUSC_T1", cohort="LUSC", sample_type="tumor"),
        ]
    )
    session.add_all(
        [
            Gene(gene_id="g_A", gene_symbol="GENE_A"),
            Gene(gene_id="g_B", gene_symbol="GENE_B"),
            Gene(gene_id="g_C", gene_symbol="GENE_C"),
        ]
    )
    session.add_all(
        [
            Expression(sample_id="LUAD_T1", gene_id="g_A", value=1.1),
            Expression(sample_id="LUAD_T2", gene_id="g_A", value=1.3),
            Expression(sample_id="LUAD_N1", gene_id="g_A", value=0.6),
            Expression(sample_id="LUSC_T1", gene_id="g_A", value=3.5),
        ]
    )
    session.add_all(
        [
            DEResult(
                gene_symbol="GENE_A",
                log_fc=2.1,
                ave_expr=1.5,
                t_stat=8.3,
                p_value=0.0001,
                adj_p_val=0.005,
            ),
            DEResult(
                gene_symbol="GENE_B",
                log_fc=1.2,
                ave_expr=2.0,
                t_stat=4.1,
                p_value=0.02,
                adj_p_val=0.08,
            ),
            DEResult(
                gene_symbol="GENE_C",
                log_fc=0.4,
                ave_expr=1.0,
                t_stat=2.0,
                p_value=0.15,
                adj_p_val=0.50,
            ),
        ]
    )
    session.add_all(
        [
            TopGene(rank=1, gene_symbol="GENE_A", mean_shap=0.90, model_version="v_test"),
            TopGene(rank=2, gene_symbol="GENE_B", mean_shap=0.55, model_version="v_test"),
            TopGene(rank=3, gene_symbol="GENE_C", mean_shap=0.20, model_version="v_test"),
        ]
    )
    session.add(
        GeneAnnotation(
            symbol="GENE_A",
            found=True,
            entrez_id=100,
            ensembl_id="ENSG000001",
            name="Gene Alpha",
            gene_type="protein-coding",
            summary="Alpha function.",
            uniprot_id="P001",
            fetched_at=_utcnow(),
        )
    )
    session.commit()


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine)

    with SessionFactory() as sess:
        _seed(sess)

    def override_db():
        with SessionFactory() as db:
            yield db

    def override_http_client():
        return _FakeHttpxClient()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_http_client] = override_http_client

    with TestClient(app) as client:
        # Override model store after lifespan startup
        app.state.model_store = _make_test_model()
        yield client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /genes/top
# ---------------------------------------------------------------------------


def test_top_genes_returns_list(api_client):
    r = api_client.get("/genes/top")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) == 3


def test_top_genes_order_by_rank(api_client):
    r = api_client.get("/genes/top")
    ranks = [g["rank"] for g in r.json()]
    assert ranks == sorted(ranks)


def test_top_genes_joined_fields(api_client):
    r = api_client.get("/genes/top")
    by_symbol = {g["gene_symbol"]: g for g in r.json()}

    gene_a = by_symbol["GENE_A"]
    assert gene_a["adj_p_val"] == pytest.approx(0.005)
    assert gene_a["log_fc"] == pytest.approx(2.1)
    assert gene_a["annotation"] is not None
    assert gene_a["annotation"]["found"] is True
    assert gene_a["annotation"]["entrez_id"] == 100

    gene_b = by_symbol["GENE_B"]
    assert gene_b["adj_p_val"] == pytest.approx(0.08)
    assert gene_b["annotation"] is None  # not seeded in gene_annotations

    gene_c = by_symbol["GENE_C"]
    assert gene_c["annotation"] is None


def test_top_genes_limit(api_client):
    r = api_client.get("/genes/top?limit=1")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["rank"] == 1


def test_top_genes_limit_validation(api_client):
    assert api_client.get("/genes/top?limit=0").status_code == 422
    assert api_client.get("/genes/top?limit=501").status_code == 422


# ---------------------------------------------------------------------------
# /genes/{symbol}  (annotation)
# ---------------------------------------------------------------------------


def test_gene_annotation_cached(api_client):
    r = api_client.get("/genes/GENE_A")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "GENE_A"
    assert data["found"] is True
    assert data["entrez_id"] == 100
    assert data["name"] == "Gene Alpha"


def test_gene_annotation_lowercase_normalized(api_client):
    r = api_client.get("/genes/gene_a")
    assert r.status_code == 200
    assert r.json()["symbol"] == "GENE_A"


def test_gene_annotation_unknown_returns_not_found(api_client):
    # Fake httpx client returns empty hits; service caches found=False
    r = api_client.get("/genes/COMPLETELY_UNKNOWN_GENE")
    assert r.status_code == 200
    data = r.json()
    assert data["found"] is False
    assert data["symbol"] == "COMPLETELY_UNKNOWN_GENE"


# ---------------------------------------------------------------------------
# /genes/{symbol}/expression
# ---------------------------------------------------------------------------


def test_expression_groups_present(api_client):
    r = api_client.get("/genes/GENE_A/expression")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "GENE_A"
    group_keys = {g["group"] for g in data["groups"]}
    assert "LUAD_tumor" in group_keys
    assert "LUAD_normal" in group_keys
    assert "LUSC_tumor" in group_keys


def test_expression_summary_stats(api_client):
    r = api_client.get("/genes/GENE_A/expression")
    groups = {g["group"]: g for g in r.json()["groups"]}

    luad_tumor = groups["LUAD_tumor"]
    assert luad_tumor["n"] == 2
    assert luad_tumor["mean"] == pytest.approx((1.1 + 1.3) / 2)
    assert luad_tumor["min"] == pytest.approx(1.1)
    assert luad_tumor["max"] == pytest.approx(1.3)
    assert len(luad_tumor["values"]) == 2

    luad_normal = groups["LUAD_normal"]
    assert luad_normal["n"] == 1
    assert luad_normal["mean"] == pytest.approx(0.6)


def test_expression_unknown_gene_404(api_client):
    r = api_client.get("/genes/NONEXISTENT_XYZ/expression")
    assert r.status_code == 404


def test_expression_lowercase_normalized(api_client):
    r = api_client.get("/genes/gene_a/expression")
    assert r.status_code == 200
    assert r.json()["symbol"] == "GENE_A"


# ---------------------------------------------------------------------------
# /de
# ---------------------------------------------------------------------------


def test_de_returns_all_by_default(api_client):
    r = api_client.get("/de")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 3
    assert data["page"] == 1
    assert len(data["results"]) == 3


def test_de_sorted_by_adj_p_val(api_client):
    r = api_client.get("/de")
    vals = [res["adj_p_val"] for res in r.json()["results"]]
    assert vals == sorted(vals)


def test_de_filter_by_threshold(api_client):
    r = api_client.get("/de?adj_p_val=0.01")
    data = r.json()
    assert data["total"] == 1
    assert data["results"][0]["gene_symbol"] == "GENE_A"
    assert data["results"][0]["adj_p_val"] == pytest.approx(0.005)


def test_de_filter_strict(api_client):
    r = api_client.get("/de?adj_p_val=0.005")
    data = r.json()
    assert data["total"] == 1


def test_de_pagination(api_client):
    r1 = api_client.get("/de?page=1&page_size=2")
    r2 = api_client.get("/de?page=2&page_size=2")
    d1, d2 = r1.json(), r2.json()

    assert d1["total"] == 3
    assert d2["total"] == 3
    assert len(d1["results"]) == 2
    assert len(d2["results"]) == 1

    syms1 = {res["gene_symbol"] for res in d1["results"]}
    syms2 = {res["gene_symbol"] for res in d2["results"]}
    assert syms1.isdisjoint(syms2)


def test_de_page_size_validation(api_client):
    assert api_client.get("/de?page_size=0").status_code == 422
    assert api_client.get("/de?page_size=501").status_code == 422


def test_de_result_fields(api_client):
    r = api_client.get("/de?adj_p_val=0.01")
    item = r.json()["results"][0]
    for field in ("gene_symbol", "log_fc", "ave_expr", "t_stat", "p_value", "adj_p_val"):
        assert field in item


# ---------------------------------------------------------------------------
# POST /predict  (JSON body)
# ---------------------------------------------------------------------------


def _full_features(lusc: bool = True) -> dict[str, float]:
    if lusc:
        return {"GENE_A": 2.0, "GENE_B": 1.8, "GENE_C": 1.5}
    return {"GENE_A": 0.0, "GENE_B": 0.1, "GENE_C": 0.2}


def test_predict_valid_json(api_client):
    r = api_client.post("/predict", json={"features": _full_features()})
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_subtype"] in ("LUAD", "LUSC")
    assert abs(data["probabilities"]["LUAD"] + data["probabilities"]["LUSC"] - 1.0) < 1e-6
    assert data["model_version"] == "v_test"


def test_predict_lusc_features(api_client):
    r = api_client.post("/predict", json={"features": _full_features(lusc=True)})
    assert r.status_code == 200
    assert r.json()["predicted_subtype"] == "LUSC"


def test_predict_luad_features(api_client):
    r = api_client.post("/predict", json={"features": _full_features(lusc=False)})
    assert r.status_code == 200
    assert r.json()["predicted_subtype"] == "LUAD"


def test_predict_missing_genes_returns_422(api_client):
    r = api_client.post("/predict", json={"features": {"GENE_A": 1.0}})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "missing" in detail["message"].lower()
    assert detail["total_missing"] == 2
    assert "GENE_B" in detail["missing_genes"] or "GENE_C" in detail["missing_genes"]


def test_predict_extra_genes_allowed(api_client):
    features = {**_full_features(), "EXTRA_GENE": 99.9}
    r = api_client.post("/predict", json={"features": features})
    assert r.status_code == 200


def test_predict_empty_features_returns_422(api_client):
    r = api_client.post("/predict", json={"features": {}})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/upload  (CSV file)
# ---------------------------------------------------------------------------


def _csv_bytes(features: dict[str, float], header: bool = True) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    if header:
        w.writerow(["gene_symbol", "value"])
    for gene, val in features.items():
        w.writerow([gene, val])
    return buf.getvalue().encode()


def test_predict_upload_valid(api_client):
    csv_data = _csv_bytes(_full_features())
    r = api_client.post(
        "/predict/upload",
        files={"file": ("expr.csv", csv_data, "text/csv")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_subtype"] in ("LUAD", "LUSC")
    assert data["model_version"] == "v_test"


def test_predict_upload_no_header(api_client):
    csv_data = _csv_bytes(_full_features(), header=False)
    r = api_client.post(
        "/predict/upload",
        files={"file": ("expr.csv", csv_data, "text/csv")},
    )
    assert r.status_code == 200


def test_predict_upload_missing_genes_returns_422(api_client):
    csv_data = _csv_bytes({"GENE_A": 1.0})
    r = api_client.post(
        "/predict/upload",
        files={"file": ("expr.csv", csv_data, "text/csv")},
    )
    assert r.status_code == 422


def test_predict_upload_empty_file_returns_400(api_client):
    r = api_client.post(
        "/predict/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Model unavailable
# ---------------------------------------------------------------------------


def test_predict_no_model_returns_503():
    """When model_store is None, /predict must return 503."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        saved = app.state.model_store
        app.state.model_store = None
        try:
            r = client.post("/predict", json={"features": _full_features()})
            assert r.status_code == 503
        finally:
            app.state.model_store = saved


# ---------------------------------------------------------------------------
# UI smoke tests (each page must return 200 with expected HTML landmarks)
# ---------------------------------------------------------------------------


def test_ui_home_page(api_client):
    r = api_client.get("/")
    assert r.status_code == 200
    assert b"LungTX" in r.content
    assert b"Gene Expression Explorer" in r.content


def test_ui_home_page_with_query(api_client):
    r = api_client.get("/?q=GENE_A")
    assert r.status_code == 200
    # Input is pre-filled with the queried symbol
    assert b"GENE_A" in r.content


def test_ui_top_genes_page(api_client):
    r = api_client.get("/top-genes")
    assert r.status_code == 200
    assert b"Top Discriminative Genes" in r.content
    # Seeded genes appear in the table
    assert b"GENE_A" in r.content
    assert b"GENE_B" in r.content


def test_ui_predict_page(api_client):
    r = api_client.get("/predict")
    assert r.status_code == 200
    assert b"Predict LUAD" in r.content
    # Feature count from fake model (3 genes)
    assert b"3" in r.content


def test_ui_pages_have_nav(api_client):
    for path in ("/", "/top-genes", "/predict"):
        r = api_client.get(path)
        assert b"nav" in r.content, f"Missing nav on {path}"
        assert b"Top Genes" in r.content, f"Missing nav link on {path}"
