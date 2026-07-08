"""
ETL pipeline for TCGA lung transcriptomics.

Reads from data/raw/ (produced by pipeline/download_data.py):
  luad_expression.tsv.gz   -- HiSeqV2 gene-x-sample matrix (LUAD cohort)
  lusc_expression.tsv.gz   -- HiSeqV2 gene-x-sample matrix (LUSC cohort)
  tcga_phenotype.tsv.gz    -- PanCanAtlas phenotype table (all TCGA cohorts)

When cohort-specific clinical matrices are present (i.e. XENA_TOKEN was used):
  luad_clinical.tsv.gz     -- LUAD clinical matrix with age/gender/stage
  lusc_clinical.tsv.gz     -- LUSC clinical matrix with age/gender/stage

Writes to data/processed/:
  expression_matrix.parquet  -- samples x genes (post-variance-filter, float32)
  sample_metadata.parquet    -- per-sample clinical and cohort fields

Loads into Postgres:
  samples, genes, expression  (idempotent: truncates before each run)

TCGA barcode format (1-indexed):
  TCGA-{TSS}-{Participant}-{SampleType}{Vial}-...
  Positions 14-15 (characters 14-15 of the full barcode, 1-indexed) are the
  two-digit sample type code embedded in the 4th hyphen-separated segment:
    01 -> primary tumor
    02 -> recurrent tumor
    06 -> metastatic
    11 -> solid tissue normal
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app.core.config import settings

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Bottom-N percentile of per-gene variance to discard before DB load.
VAR_PERCENTILE = 10.0

# Rows per COPY buffer flush (expression table only).
COPY_CHUNK = 500_000

# ---------------------------------------------------------------------------
# PanCanAtlas phenotype column -> sample metadata field mapping.
# TCGA hub clinical matrices add age/gender/stage when present.
# ---------------------------------------------------------------------------
_PHENOTYPE_COL_MAP = {
    "_primary_disease": "subtype",
    "sample_type": "sample_type_clinical",
}

_CLINICAL_COL_MAP = {
    "age_at_initial_pathologic_diagnosis": "age",
    "gender": "gender",
    "pathologic_stage": "stage",
}

# Two-digit TCGA sample type codes -> canonical label.
_SAMPLE_TYPE_CODES = {
    "01": "primary_tumor",
    "02": "recurrent_tumor",
    "06": "metastatic",
    "07": "additional_metastatic",
    "10": "blood_derived_normal",
    "11": "solid_tissue_normal",
    "12": "buccal_cell_normal",
}


# ---------------------------------------------------------------------------
# Pure functions (no I/O, fully testable)
# ---------------------------------------------------------------------------


def parse_sample_type(barcode: str) -> str:
    """Return the sample type label from a TCGA barcode.

    Extracts positions 14-15 (1-indexed) of the barcode, which correspond to
    the first two characters of the 4th hyphen-separated segment:

        TCGA-05-4244-01A-... -> segment[3][:2] == '01' -> 'primary_tumor'
        TCGA-05-4244-11A-... -> segment[3][:2] == '11' -> 'solid_tissue_normal'

    Returns 'unknown' for malformed barcodes.
    """
    parts = barcode.split("-")
    if len(parts) < 4:
        return "unknown"
    code = parts[3][:2]
    return _SAMPLE_TYPE_CODES.get(code, f"other_{code}")


def filter_low_variance(
    expr: pd.DataFrame,
    percentile: float = VAR_PERCENTILE,
) -> pd.DataFrame:
    """Drop genes whose variance across samples falls below *percentile*.

    Args:
        expr: samples x genes DataFrame of log2-normalised expression values.
        percentile: genes below this variance percentile are dropped (0-100).

    Returns:
        Filtered DataFrame with the same row index.
    """
    gene_var = expr.var(axis=0)
    threshold = float(np.percentile(gene_var, percentile))
    keep_mask = gene_var >= threshold
    n_total = len(gene_var)
    n_dropped = int((~keep_mask).sum())
    log.info(
        "Variance filter (p%.0f): dropping %d / %d genes (var < %.4f); keeping %d",
        percentile,
        n_dropped,
        n_total,
        threshold,
        n_total - n_dropped,
    )
    return expr.loc[:, keep_mask]


def build_sample_meta(
    luad_samples: list[str],
    lusc_samples: list[str],
    phenotype: pd.DataFrame,
    luad_clinical: pd.DataFrame | None = None,
    lusc_clinical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine sample lists, barcodes, phenotype, and optional clinical data.

    Returns a DataFrame with sample_id as the index and columns:
    cohort, sample_type, subtype, age, gender, stage.
    """
    meta = pd.DataFrame(
        {
            "sample_id": luad_samples + lusc_samples,
            "cohort": ["LUAD"] * len(luad_samples) + ["LUSC"] * len(lusc_samples),
        }
    )
    meta["sample_type"] = meta["sample_id"].map(parse_sample_type)

    # Subtype from phenotype (_primary_disease)
    if "_primary_disease" in phenotype.columns:
        subtype_map = phenotype["_primary_disease"].to_dict()
        meta["subtype"] = meta["sample_id"].map(subtype_map)
    else:
        meta["subtype"] = None

    # age / gender / stage from cohort-specific clinical matrices (may be None)
    for col in ("age", "gender", "stage"):
        meta[col] = None

    for clin, cohort in ((luad_clinical, "LUAD"), (lusc_clinical, "LUSC")):
        if clin is None:
            continue
        mask = meta["cohort"] == cohort
        for src, dst in _CLINICAL_COL_MAP.items():
            if src in clin.columns:
                meta.loc[mask, dst] = meta.loc[mask, "sample_id"].map(clin[src].to_dict())

    meta = meta.set_index("sample_id")
    return meta


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_expression(path: Path, cohort: str) -> pd.DataFrame:
    """Read a HiSeqV2 gzipped TSV and return a samples-x-genes DataFrame."""
    log.info("[%s] loading expression from %s", cohort, path.name)
    df = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    df = df.T  # genes x samples -> samples x genes
    df.index.name = "sample_id"
    df = df.astype("float32")
    log.info("[%s] %d samples x %d genes", cohort, *df.shape)
    return df


def _load_phenotype(path: Path) -> pd.DataFrame:
    """Read a phenotype/clinical file (gzipped or plain TSV) indexed by sample_id."""
    log.info("Loading phenotype from %s", path.name)
    df = pd.read_csv(path, sep="\t", index_col=0, compression="infer")
    df.index.name = "sample_id"
    log.info("  %d rows x %d cols", *df.shape)
    return df


def _try_load_clinical(path: Path, cohort: str) -> pd.DataFrame | None:
    """Load a cohort-specific clinical matrix if it is a real cohort file.

    Returns None when the file is absent or is a pan-TCGA phenotype file
    (detected by the absence of clinical columns like age/gender/stage).
    """
    if not path.exists():
        return None
    df = _load_phenotype(path)
    has_clinical = any(c in df.columns for c in _CLINICAL_COL_MAP)
    if not has_clinical:
        log.info("[%s] clinical file is pan-TCGA phenotype; age/gender/stage will be NULL", cohort)
        return None
    return df


# ---------------------------------------------------------------------------
# Database loaders
# ---------------------------------------------------------------------------


def _truncate_tables(engine) -> None:
    # Single statement required: Postgres rejects truncating a referenced table
    # unless all tables sharing FK relationships are named together.
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE expression, samples, genes"))
    log.info("Tables truncated")


def _load_samples(engine, meta: pd.DataFrame) -> None:
    log.info("Loading %d samples...", len(meta))
    rows = meta.reset_index().rename(columns={"index": "sample_id"})
    rows = rows.where(pd.notna(rows), other=None)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO samples (sample_id, cohort, sample_type, subtype, age, gender, stage)"
                " VALUES (:sample_id, :cohort, :sample_type, :subtype, :age, :gender, :stage)"
            ),
            rows[
                ["sample_id", "cohort", "sample_type", "subtype", "age", "gender", "stage"]
            ].to_dict("records"),
        )
    log.info("  samples done")


def _load_genes(engine, gene_symbols: list[str]) -> None:
    log.info("Loading %d genes...", len(gene_symbols))
    rows = [{"gene_id": g, "gene_symbol": g} for g in gene_symbols]
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO genes (gene_id, gene_symbol) VALUES (:gene_id, :gene_symbol)"),
            rows,
        )
    log.info("  genes done")


def _copy_expression(raw_conn, buf: io.StringIO) -> None:
    buf.seek(0)
    with raw_conn.cursor() as cur:
        cur.copy_expert(
            "COPY expression (sample_id, gene_id, value) FROM STDIN WITH (FORMAT TEXT)",
            buf,
        )


def _load_expression_copy(engine, expr: pd.DataFrame) -> None:
    """Bulk-load expression table via PostgreSQL COPY, chunked by gene batches."""
    genes = expr.columns.tolist()
    n_total = len(expr) * len(genes)
    log.info("Loading %d expression rows (COPY, %d-row chunks)...", n_total, COPY_CHUNK)

    raw_conn = engine.raw_connection()
    try:
        buf = io.StringIO()
        written = 0

        for sample_id, row in expr.iterrows():
            for gene_id, val in row.items():
                if np.isnan(val):
                    continue
                buf.write(f"{sample_id}\t{gene_id}\t{val:.6f}\n")
                written += 1
                if written % COPY_CHUNK == 0:
                    _copy_expression(raw_conn, buf)
                    buf = io.StringIO()
                    log.info("  copied %d / %d rows", written, n_total)

        if buf.tell():
            _copy_expression(raw_conn, buf)

        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise
    finally:
        raw_conn.close()

    log.info("  expression done: %d rows", written)


# ---------------------------------------------------------------------------
# Clinical backfill (updates age/gender/stage without reloading expression)
# ---------------------------------------------------------------------------


def _backfill_cohort(engine, clin: pd.DataFrame | None, cohort: str) -> None:
    if clin is None:
        log.warning("[%s] no clinical data; skipping backfill", cohort)
        return

    col_map = {src: dst for src, dst in _CLINICAL_COL_MAP.items() if src in clin.columns}
    if not col_map:
        log.warning("[%s] no mapped columns found in clinical file; skipping", cohort)
        return

    rows = []
    for sample_id in clin.index:
        row: dict = {"sample_id": sample_id, "age": None, "gender": None, "stage": None}
        for src, dst in col_map.items():
            val = clin.at[sample_id, src]
            if pd.isna(val):
                val = None
            elif dst == "age":
                try:
                    val = int(float(val))
                except (ValueError, TypeError):
                    val = None
            row[dst] = val
        rows.append(row)

    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE samples SET age = :age, gender = :gender, stage = :stage"
                " WHERE sample_id = :sample_id"
            ),
            rows,
        )
    log.info("[%s] backfilled %d samples", cohort, len(rows))


def run_backfill(
    raw_dir: Path = RAW_DIR,
    database_url: str | None = None,
) -> None:
    """Update age, gender, stage in the samples table from cohort clinical matrices.

    Reads luad_clinical.tsv and lusc_clinical.tsv from raw_dir.
    Does not touch the expression table.
    """
    url = database_url or settings.database_url
    engine = create_engine(url, pool_pre_ping=True)

    luad_clin = _try_load_clinical(raw_dir / "luad_clinical.tsv", "LUAD")
    lusc_clin = _try_load_clinical(raw_dir / "lusc_clinical.tsv", "LUSC")

    _backfill_cohort(engine, luad_clin, "LUAD")
    _backfill_cohort(engine, lusc_clin, "LUSC")

    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT"
                " COUNT(*) AS total,"
                " COUNT(age) AS age_count,"
                " COUNT(gender) AS gender_count,"
                " COUNT(stage) AS stage_count"
                " FROM samples"
            )
        ).one()
    log.info(
        "Non-null counts (total %d): age=%d, gender=%d, stage=%d",
        result.total,
        result.age_count,
        result.gender_count,
        result.stage_count,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_etl(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    database_url: str | None = None,
    var_percentile: float = VAR_PERCENTILE,
) -> None:
    # 1. Load expression matrices
    luad_expr = _load_expression(raw_dir / "luad_expression.tsv.gz", "LUAD")
    lusc_expr = _load_expression(raw_dir / "lusc_expression.tsv.gz", "LUSC")

    # 2. Load clinical/phenotype data
    phenotype = _load_phenotype(raw_dir / "tcga_phenotype.tsv.gz")
    luad_clinical = _try_load_clinical(raw_dir / "luad_clinical.tsv.gz", "LUAD")
    lusc_clinical = _try_load_clinical(raw_dir / "lusc_clinical.tsv.gz", "LUSC")

    # 3. Build sample metadata
    meta = build_sample_meta(
        luad_samples=luad_expr.index.tolist(),
        lusc_samples=lusc_expr.index.tolist(),
        phenotype=phenotype,
        luad_clinical=luad_clinical,
        lusc_clinical=lusc_clinical,
    )
    log.info(
        "Sample metadata: %d total (%d LUAD, %d LUSC)",
        len(meta),
        (meta["cohort"] == "LUAD").sum(),
        (meta["cohort"] == "LUSC").sum(),
    )

    # 4. Concatenate expression and apply variance filter
    expr = pd.concat([luad_expr, lusc_expr])
    expr = filter_low_variance(expr, percentile=var_percentile)

    # 5. Persist processed artefacts
    processed_dir.mkdir(parents=True, exist_ok=True)
    expr_path = processed_dir / "expression_matrix.parquet"
    meta_path = processed_dir / "sample_metadata.parquet"
    expr.to_parquet(expr_path)
    meta.to_parquet(meta_path)
    log.info("Saved %s (%s)", expr_path.name, _fmt_size(expr_path))
    log.info("Saved %s (%s)", meta_path.name, _fmt_size(meta_path))

    # 6. Load into Postgres
    url = database_url or settings.database_url
    engine = create_engine(url, pool_pre_ping=True)

    _truncate_tables(engine)
    _load_samples(engine, meta)
    _load_genes(engine, expr.columns.tolist())
    _load_expression_copy(engine, expr)

    log.info(
        "ETL complete: %d samples, %d genes, %d expression rows",
        len(meta),
        len(expr.columns),
        len(meta) * len(expr.columns),
    )


def _fmt_size(p: Path) -> str:
    mb = p.stat().st_size / 1e6
    return f"{mb:.1f} MB"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_etl()


def main_backfill() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_backfill()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        main_backfill()
    else:
        main()
