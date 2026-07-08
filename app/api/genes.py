from __future__ import annotations

from collections import defaultdict

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_http_client
from app.db.session import get_db
from app.models.genomics import DEResult, Expression, Gene, GeneAnnotation, Sample, TopGene
from app.schemas.genomics import AnnotationOut, ExpressionGroup, ExpressionOut, TopGeneOut
from app.services.annotation import AnnotationResult, get_or_fetch

router = APIRouter(prefix="/genes", tags=["genes"])


def _annotation_result_to_out(r: AnnotationResult) -> AnnotationOut:
    return AnnotationOut(
        symbol=r.symbol,
        found=r.found,
        entrez_id=r.entrez_id,
        ensembl_id=r.ensembl_id,
        name=r.name,
        gene_type=r.gene_type,
        summary=r.summary,
        uniprot_id=r.uniprot_id,
    )


# ---------------------------------------------------------------------------
# GET /genes/top  -- must be defined before /{symbol}
# ---------------------------------------------------------------------------


@router.get("/top", response_model=list[TopGeneOut])
def get_top_genes(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of genes to return"),
    db: Session = Depends(get_db),
) -> list[TopGeneOut]:
    rows = db.execute(
        select(
            TopGene.rank,
            TopGene.gene_symbol,
            TopGene.mean_shap,
            TopGene.model_version,
            DEResult.adj_p_val,
            DEResult.log_fc,
            GeneAnnotation.found,
            GeneAnnotation.entrez_id,
            GeneAnnotation.ensembl_id,
            GeneAnnotation.name,
            GeneAnnotation.gene_type,
            GeneAnnotation.summary,
            GeneAnnotation.uniprot_id,
        )
        .join(DEResult, DEResult.gene_symbol == TopGene.gene_symbol, isouter=True)
        .join(GeneAnnotation, GeneAnnotation.symbol == TopGene.gene_symbol, isouter=True)
        .order_by(TopGene.rank)
        .limit(limit)
    ).all()

    result: list[TopGeneOut] = []
    for row in rows:
        annotation: AnnotationOut | None = None
        if row.found is not None:
            annotation = AnnotationOut(
                symbol=row.gene_symbol,
                found=row.found,
                entrez_id=row.entrez_id,
                ensembl_id=row.ensembl_id,
                name=row.name,
                gene_type=row.gene_type,
                summary=row.summary,
                uniprot_id=row.uniprot_id,
            )
        result.append(
            TopGeneOut(
                rank=row.rank,
                gene_symbol=row.gene_symbol,
                mean_shap=row.mean_shap,
                model_version=row.model_version,
                adj_p_val=row.adj_p_val,
                log_fc=row.log_fc,
                annotation=annotation,
            )
        )
    return result


# ---------------------------------------------------------------------------
# GET /genes/{symbol}/expression
# ---------------------------------------------------------------------------


@router.get("/{symbol}/expression", response_model=ExpressionOut)
def get_expression(
    symbol: str,
    db: Session = Depends(get_db),
) -> ExpressionOut:
    symbol = symbol.upper()

    gene_row = db.scalar(select(Gene).where(Gene.gene_symbol == symbol))
    if gene_row is None:
        raise HTTPException(status_code=404, detail=f"Gene {symbol!r} not found")

    rows = db.execute(
        select(Sample.cohort, Sample.sample_type, Expression.value)
        .select_from(Expression)
        .join(Sample, Expression.sample_id == Sample.sample_id)
        .where(Expression.gene_id == gene_row.gene_id)
    ).all()

    groups_data: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cohort, sample_type, value in rows:
        groups_data[(cohort, sample_type)].append(value)

    groups: list[ExpressionGroup] = []
    for (cohort, sample_type), vals in sorted(groups_data.items()):
        arr = np.array(vals, dtype=np.float64)
        groups.append(
            ExpressionGroup(
                group=f"{cohort}_{sample_type}",
                subtype=cohort,
                sample_type=sample_type,
                n=len(arr),
                mean=float(arr.mean()),
                median=float(np.median(arr)),
                q1=float(np.percentile(arr, 25)),
                q3=float(np.percentile(arr, 75)),
                min=float(arr.min()),
                max=float(arr.max()),
                values=arr.tolist(),
            )
        )

    return ExpressionOut(symbol=symbol, gene_id=gene_row.gene_id, groups=groups)


# ---------------------------------------------------------------------------
# GET /genes/{symbol}
# ---------------------------------------------------------------------------


@router.get("/{symbol}", response_model=AnnotationOut)
def get_gene_annotation(
    symbol: str,
    db: Session = Depends(get_db),
    http_client: httpx.Client = Depends(get_http_client),
) -> AnnotationOut:
    symbol = symbol.upper()
    result = get_or_fetch(symbol, db, http_client)
    db.commit()
    return _annotation_result_to_out(result)
