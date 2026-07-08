"""Load DE results from r_analysis/output/de_results.csv into the de_results table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from app.core.config import settings

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[1]
DE_RESULTS_PATH = PROJECT_ROOT / "r_analysis" / "output" / "de_results.csv"


def run_load_de(
    csv_path: Path = DE_RESULTS_PATH,
    database_url: str | None = None,
) -> None:
    """Truncate de_results and reload from csv_path.

    Column mapping (CSV -> DB):
      logFC       -> log_fc
      AveExpr     -> ave_expr
      t           -> t_stat
      P.Value     -> p_value
      adj.P.Val   -> adj_p_val
    """
    df = pd.read_csv(csv_path)
    log.info("Read %d rows from %s", len(df), csv_path.name)

    rows = df.rename(
        columns={
            "logFC": "log_fc",
            "AveExpr": "ave_expr",
            "t": "t_stat",
            "P.Value": "p_value",
            "adj.P.Val": "adj_p_val",
        }
    )[["gene_symbol", "log_fc", "ave_expr", "t_stat", "p_value", "adj_p_val"]].to_dict("records")

    url = database_url or settings.database_url
    engine = create_engine(url, pool_pre_ping=True)

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE de_results"))
        conn.execute(
            text(
                "INSERT INTO de_results"
                " (gene_symbol, log_fc, ave_expr, t_stat, p_value, adj_p_val)"
                " VALUES (:gene_symbol, :log_fc, :ave_expr, :t_stat, :p_value, :adj_p_val)"
            ),
            rows,
        )
    log.info("Loaded %d rows into de_results", len(rows))

    n_sig = (df["adj.P.Val"] < 0.05).sum()
    log.info("Significant genes (adj.P.Val < 0.05): %d / %d", n_sig, len(df))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_load_de()


if __name__ == "__main__":
    main()
