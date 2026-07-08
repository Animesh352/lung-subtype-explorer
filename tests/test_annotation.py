"""
Tests for app/services/annotation.py.

All MyGene.info HTTP calls are intercepted by a fake httpx client so these
tests run fully offline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.session import Base
from app.models.genomics import GeneAnnotation
from app.services.annotation import (
    AnnotationFetchError,
    batch_annotate,
    fetch_annotation,
    get_or_fetch,
)

# ---------------------------------------------------------------------------
# Fake httpx helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        import httpx

        if self.status_code >= 400:
            request = MagicMock()
            response = MagicMock()
            response.status_code = self.status_code
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self._status_code = status_code
        self.call_count = 0

    def get(self, url: str, **kwargs) -> _FakeResponse:
        self.call_count += 1
        return _FakeResponse(self._payload, self._status_code)


class _NetworkErrorClient:
    def get(self, url: str, **kwargs):
        import httpx

        raise httpx.RequestError("connection refused")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TP53_HIT = {
    "entrezgene": 7157,
    "ensembl": {"gene": "ENSG00000141510"},
    "name": "tumor protein p53",
    "type_of_gene": "protein-coding",
    "summary": "This gene encodes a tumor suppressor protein.",
    "uniprot": {"Swiss-Prot": "P04637"},
}

_TP53_PAYLOAD = {"hits": [_TP53_HIT]}
_EMPTY_PAYLOAD = {"hits": []}


@pytest.fixture
def sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# fetch_annotation tests
# ---------------------------------------------------------------------------


def test_fetch_found_gene():
    client = _FakeClient(_TP53_PAYLOAD)
    result = fetch_annotation("TP53", client)

    assert result.found is True
    assert result.symbol == "TP53"
    assert result.entrez_id == 7157
    assert result.ensembl_id == "ENSG00000141510"
    assert result.name == "tumor protein p53"
    assert result.gene_type == "protein-coding"
    assert "tumor suppressor" in result.summary
    assert result.uniprot_id == "P04637"


def test_fetch_unknown_symbol():
    client = _FakeClient(_EMPTY_PAYLOAD)
    result = fetch_annotation("NOTAREALGENEXYZ", client)

    assert result.found is False
    assert result.symbol == "NOTAREALGENEXYZ"
    assert result.entrez_id is None
    assert result.ensembl_id is None


def test_fetch_http_error_raises():
    client = _FakeClient({}, status_code=500)
    with pytest.raises(AnnotationFetchError, match="HTTP 500"):
        fetch_annotation("TP53", client)


def test_fetch_network_error_raises():
    client = _NetworkErrorClient()
    with pytest.raises(AnnotationFetchError, match="connection refused"):
        fetch_annotation("TP53", client)


# ---------------------------------------------------------------------------
# _parse_hit edge cases
# ---------------------------------------------------------------------------


def test_ensembl_as_list():
    hit = dict(_TP53_HIT)
    hit["ensembl"] = [{"gene": "ENSG00000141510"}, {"gene": "ENSG99999999999"}]
    client = _FakeClient({"hits": [hit]})
    result = fetch_annotation("TP53", client)
    assert result.ensembl_id == "ENSG00000141510"


def test_uniprot_swiss_prot_as_list():
    hit = dict(_TP53_HIT)
    hit["uniprot"] = {"Swiss-Prot": ["P04637", "P99999"]}
    client = _FakeClient({"hits": [hit]})
    result = fetch_annotation("TP53", client)
    assert result.uniprot_id == "P04637"


def test_missing_optional_fields():
    hit = {"entrezgene": 7157}
    client = _FakeClient({"hits": [hit]})
    result = fetch_annotation("TP53", client)
    assert result.found is True
    assert result.ensembl_id is None
    assert result.uniprot_id is None
    assert result.summary is None


# ---------------------------------------------------------------------------
# get_or_fetch tests
# ---------------------------------------------------------------------------


def test_get_or_fetch_cache_miss_then_hit(sqlite_session):
    client = _FakeClient(_TP53_PAYLOAD)

    result1 = get_or_fetch("TP53", sqlite_session, client)
    assert result1.found is True
    assert client.call_count == 1

    result2 = get_or_fetch("TP53", sqlite_session, client)
    assert result2.found is True
    assert client.call_count == 1  # served from cache


def test_get_or_fetch_api_error_not_cached(sqlite_session):
    bad_client = _FakeClient({}, status_code=503)

    result = get_or_fetch("TP53", sqlite_session, bad_client)
    assert result.found is False

    row = sqlite_session.get(GeneAnnotation, "TP53")
    assert row is None  # not cached on failure


def test_get_or_fetch_not_found_is_cached(sqlite_session):
    client = _FakeClient(_EMPTY_PAYLOAD)

    result = get_or_fetch("FAKE999", sqlite_session, client)
    assert result.found is False
    assert client.call_count == 1

    get_or_fetch("FAKE999", sqlite_session, client)
    assert client.call_count == 1  # not-found result was cached


# ---------------------------------------------------------------------------
# batch_annotate tests
# ---------------------------------------------------------------------------


def test_batch_annotate_empty(sqlite_session):
    client = _FakeClient(_TP53_PAYLOAD)
    results = batch_annotate([], sqlite_session, client)
    assert results == []
    assert client.call_count == 0


def test_batch_annotate_all_new(sqlite_session):
    egfr_hit = {
        "entrezgene": 1956,
        "ensembl": {"gene": "ENSG00000146648"},
        "name": "epidermal growth factor receptor",
        "type_of_gene": "protein-coding",
        "summary": "EGFR summary.",
        "uniprot": {"Swiss-Prot": "P00533"},
    }

    responses = {
        "TP53": {"hits": [_TP53_HIT]},
        "EGFR": {"hits": [egfr_hit]},
    }

    call_log: list[str] = []

    class _MultiClient:
        call_count = 0

        def get(self, url: str, params=None, **kwargs) -> _FakeResponse:
            _MultiClient.call_count += 1
            symbol = (params or {}).get("q", "").replace("symbol:", "")
            call_log.append(symbol)
            return _FakeResponse(responses.get(symbol, _EMPTY_PAYLOAD))

    client = _MultiClient()
    results = batch_annotate(["TP53", "EGFR"], sqlite_session, client)

    assert len(results) == 2
    assert {r.symbol for r in results} == {"TP53", "EGFR"}
    assert all(r.found for r in results)
    assert _MultiClient.call_count == 2


def test_batch_annotate_partial_cache(sqlite_session):
    # Pre-cache TP53
    sqlite_session.add(
        GeneAnnotation(
            symbol="TP53",
            found=True,
            entrez_id=7157,
            ensembl_id="ENSG00000141510",
            name="tumor protein p53",
            gene_type="protein-coding",
            summary=None,
            uniprot_id="P04637",
            fetched_at=datetime.now(UTC).replace(tzinfo=None),
        )
    )
    sqlite_session.flush()

    client = _FakeClient(_EMPTY_PAYLOAD)
    results = batch_annotate(["TP53", "BRAND_NEW"], sqlite_session, client)

    assert len(results) == 2
    tp53 = next(r for r in results if r.symbol == "TP53")
    assert tp53.found is True
    assert client.call_count == 1  # only BRAND_NEW hit the API
