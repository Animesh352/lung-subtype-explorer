from __future__ import annotations

from pydantic import BaseModel, Field


class AnnotationOut(BaseModel):
    symbol: str
    found: bool
    entrez_id: int | None = None
    ensembl_id: str | None = None
    name: str | None = None
    gene_type: str | None = None
    summary: str | None = None
    uniprot_id: str | None = None


class ExpressionGroup(BaseModel):
    group: str
    subtype: str
    sample_type: str
    n: int
    mean: float
    median: float
    q1: float
    q3: float
    min: float
    max: float
    values: list[float]


class ExpressionOut(BaseModel):
    symbol: str
    gene_id: str
    groups: list[ExpressionGroup]


class TopGeneOut(BaseModel):
    rank: int
    gene_symbol: str
    mean_shap: float
    model_version: str
    adj_p_val: float | None = None
    log_fc: float | None = None
    annotation: AnnotationOut | None = None


class DEResultOut(BaseModel):
    gene_symbol: str
    log_fc: float
    ave_expr: float
    t_stat: float
    p_value: float
    adj_p_val: float


class DEPage(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[DEResultOut]


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        description="Map of gene symbol to log2-normalized expression value"
    )


class PredictResponse(BaseModel):
    predicted_subtype: str
    probabilities: dict[str, float]
    model_version: str
