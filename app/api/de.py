from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.genomics import DEResult
from app.schemas.genomics import DEPage, DEResultOut

router = APIRouter(tags=["de"])


@router.get("/de", response_model=DEPage)
def get_de_results(
    adj_p_val: float = Query(
        1.0, ge=0.0, le=1.0, description="Return only genes with adj.P.Val <= this threshold"
    ),
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(50, ge=1, le=500, description="Results per page"),
    db: Session = Depends(get_db),
) -> DEPage:
    where_clause = DEResult.adj_p_val <= adj_p_val

    total: int = db.scalar(select(func.count(DEResult.gene_symbol)).where(where_clause)) or 0

    rows = db.scalars(
        select(DEResult)
        .where(where_clause)
        .order_by(DEResult.adj_p_val)
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()

    return DEPage(
        total=total,
        page=page,
        page_size=page_size,
        results=[
            DEResultOut(
                gene_symbol=r.gene_symbol,
                log_fc=r.log_fc,
                ave_expr=r.ave_expr,
                t_stat=r.t_stat,
                p_value=r.p_value,
                adj_p_val=r.adj_p_val,
            )
            for r in rows
        ],
    )
