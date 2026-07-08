"""
Gene annotation service backed by MyGene.info.

Public API
----------
fetch_annotation(symbol, client)
    Query MyGene.info for a single gene symbol. Returns an AnnotationResult.
    Raises AnnotationFetchError on network / server errors.

get_or_fetch(symbol, session, client)
    Return a cached annotation from gene_annotations or fetch and cache it.
    API failures are logged and return a not-found result without caching,
    so the next call will retry the API.

batch_annotate(symbols, session, client)
    Annotate a list of symbols, skipping those already cached.
    Designed for pre-populating annotations for top DE and SHAP genes.
    The caller is responsible for committing the session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.genomics import GeneAnnotation as _Row

log = logging.getLogger(__name__)

MYGENE_URL = "https://mygene.info/v3/query"
_FIELDS = "entrezgene,ensembl.gene,name,type_of_gene,summary,uniprot"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnotationResult:
    symbol: str
    found: bool
    entrez_id: int | None = None
    ensembl_id: str | None = None
    name: str | None = None
    gene_type: str | None = None
    summary: str | None = None
    uniprot_id: str | None = None


class AnnotationFetchError(Exception):
    """Raised when MyGene.info is unreachable or returns a server error."""


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------


def fetch_annotation(symbol: str, client: httpx.Client) -> AnnotationResult:
    """Query MyGene.info for symbol and return an AnnotationResult.

    Returns AnnotationResult(found=False) when the symbol is unknown.
    Raises AnnotationFetchError on HTTP 4xx/5xx or network failures.
    """
    try:
        resp = client.get(
            MYGENE_URL,
            params={
                "q": f"symbol:{symbol}",
                "fields": _FIELDS,
                "species": "human",
                "size": 1,
            },
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise AnnotationFetchError(
            f"HTTP {exc.response.status_code} from MyGene.info for {symbol!r}"
        ) from exc
    except httpx.RequestError as exc:
        raise AnnotationFetchError(
            f"Request error from MyGene.info for {symbol!r}: {exc}"
        ) from exc

    hits = resp.json().get("hits", [])
    if not hits:
        log.debug("[annotation] %r not found in MyGene.info", symbol)
        return AnnotationResult(symbol=symbol, found=False)

    return _parse_hit(hits[0], symbol)


def _parse_hit(hit: dict, symbol: str) -> AnnotationResult:
    entrez_raw = hit.get("entrezgene")
    entrez_id = int(entrez_raw) if entrez_raw is not None else None

    ensembl = hit.get("ensembl")
    ensembl_id: str | None = None
    if isinstance(ensembl, dict):
        ensembl_id = ensembl.get("gene")
    elif isinstance(ensembl, list) and ensembl:
        first = ensembl[0]
        ensembl_id = first.get("gene") if isinstance(first, dict) else None

    uniprot = hit.get("uniprot") or {}
    uniprot_id: str | None = None
    if isinstance(uniprot, dict):
        sp = uniprot.get("Swiss-Prot")
        if isinstance(sp, list):
            uniprot_id = sp[0] if sp else None
        elif isinstance(sp, str):
            uniprot_id = sp

    return AnnotationResult(
        symbol=symbol,
        found=True,
        entrez_id=entrez_id,
        ensembl_id=ensembl_id,
        name=hit.get("name"),
        gene_type=hit.get("type_of_gene"),
        summary=hit.get("summary"),
        uniprot_id=uniprot_id,
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _row_to_result(row: _Row) -> AnnotationResult:
    return AnnotationResult(
        symbol=row.symbol,
        found=row.found,
        entrez_id=row.entrez_id,
        ensembl_id=row.ensembl_id,
        name=row.name,
        gene_type=row.gene_type,
        summary=row.summary,
        uniprot_id=row.uniprot_id,
    )


def _result_to_row(result: AnnotationResult) -> _Row:
    return _Row(
        symbol=result.symbol,
        found=result.found,
        entrez_id=result.entrez_id,
        ensembl_id=result.ensembl_id,
        name=result.name,
        gene_type=result.gene_type,
        summary=result.summary,
        uniprot_id=result.uniprot_id,
        fetched_at=datetime.now(UTC).replace(tzinfo=None),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_or_fetch(
    symbol: str,
    session: Session,
    client: httpx.Client,
) -> AnnotationResult:
    """Return a cached annotation from gene_annotations, or fetch and cache it.

    API errors are logged and return AnnotationResult(found=False) without
    caching, so the next call will retry.
    """
    cached = session.get(_Row, symbol)
    if cached is not None:
        return _row_to_result(cached)

    try:
        result = fetch_annotation(symbol, client)
    except AnnotationFetchError as exc:
        log.warning("[annotation] fetch failed for %r: %s", symbol, exc)
        return AnnotationResult(symbol=symbol, found=False)

    session.add(_result_to_row(result))
    session.flush()

    log.info(
        "[annotation] %r: found=%s entrez=%s ensembl=%s",
        symbol,
        result.found,
        result.entrez_id,
        result.ensembl_id,
    )
    return result


def batch_annotate(
    symbols: list[str],
    session: Session,
    client: httpx.Client,
) -> list[AnnotationResult]:
    """Fetch and cache annotations for multiple gene symbols.

    Symbols already in gene_annotations are returned from cache without hitting
    the API. Designed to pre-populate annotations for the top DE and SHAP genes.
    The caller is responsible for committing the session after this call.
    """
    if not symbols:
        return []

    cached_symbols: set[str] = set(
        session.scalars(
            select(_Row.symbol).where(_Row.symbol.in_(symbols))
        )
    )
    n_cached = len(cached_symbols)
    n_fetch = len(symbols) - n_cached
    log.info(
        "batch_annotate: %d symbols (%d cached, %d to fetch)",
        len(symbols),
        n_cached,
        n_fetch,
    )

    results: list[AnnotationResult] = []
    for symbol in symbols:
        if symbol in cached_symbols:
            row = session.get(_Row, symbol)
            results.append(_row_to_result(row))  # type: ignore[arg-type]
        else:
            results.append(get_or_fetch(symbol, session, client))

    found = sum(1 for r in results if r.found)
    log.info("batch_annotate: %d/%d symbols resolved", found, len(symbols))
    return results
