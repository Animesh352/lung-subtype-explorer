"""
Pre-populate gene_annotations for top DE genes and top SHAP genes.

Reads the latest model version's top_genes.csv from models/ and the
de_results table from Postgres, then calls batch_annotate to fetch and cache
annotations from MyGene.info.

Usage:
    python pipeline/annotate.py [--top-de N] [--top-shap N]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import httpx
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.genomics import DEResult
from app.services.annotation import batch_annotate

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

_TOP_DE_DEFAULT = 100
_TOP_SHAP_DEFAULT = 50


def _latest_model_dir() -> Path | None:
    versions = sorted(
        (d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")),
        key=lambda d: d.name,
        reverse=True,
    )
    return versions[0] if versions else None


def _shap_symbols_from_disk(n: int) -> list[str]:
    model_dir = _latest_model_dir()
    if model_dir is None:
        log.warning("No model version found in %s; skipping SHAP genes", MODELS_DIR)
        return []
    csv_path = model_dir / "top_genes.csv"
    if not csv_path.exists():
        log.warning("top_genes.csv not found in %s; skipping SHAP genes", model_dir)
        return []
    df = pd.read_csv(csv_path)
    return df.sort_values("rank")["gene_symbol"].head(n).tolist()


def _de_symbols_from_db(session: Session, n: int) -> list[str]:
    rows = session.scalars(select(DEResult.gene_symbol).order_by(DEResult.adj_p_val).limit(n)).all()
    return list(rows)


def run_annotate(
    top_de: int = _TOP_DE_DEFAULT,
    top_shap: int = _TOP_SHAP_DEFAULT,
    database_url: str | None = None,
) -> None:
    url = database_url or settings.database_url
    engine = create_engine(url, pool_pre_ping=True)

    with Session(engine) as session, httpx.Client(timeout=30) as client:
        de_symbols = _de_symbols_from_db(session, top_de)
        shap_symbols = _shap_symbols_from_disk(top_shap)

        combined = list(dict.fromkeys(de_symbols + shap_symbols))
        log.info(
            "Annotating %d unique symbols (%d DE + %d SHAP, deduplicated)",
            len(combined),
            len(de_symbols),
            len(shap_symbols),
        )

        results = batch_annotate(combined, session, client)
        session.commit()

    found = sum(1 for r in results if r.found)
    log.info("Done: %d/%d symbols resolved", found, len(results))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-populate gene annotations")
    p.add_argument(
        "--top-de",
        type=int,
        default=_TOP_DE_DEFAULT,
        metavar="N",
        help="Number of top DE genes to annotate (default: %(default)s)",
    )
    p.add_argument(
        "--top-shap",
        type=int,
        default=_TOP_SHAP_DEFAULT,
        metavar="N",
        help="Number of top SHAP genes to annotate (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run_annotate(top_de=args.top_de, top_shap=args.top_shap)


if __name__ == "__main__":
    main()
