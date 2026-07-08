from __future__ import annotations

import csv
import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_model
from app.schemas.genomics import PredictRequest, PredictResponse
from app.services.model import ModelStore

log = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])

_LABEL_DISPLAY = {"LUAD": "LUAD (lung adenocarcinoma)", "LUSC": "LUSC (lung squamous cell)"}


def _run_predict(features: dict[str, float], model: ModelStore) -> PredictResponse:
    missing = [g for g in model.feature_names if g not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"{len(missing)} required gene(s) missing from feature vector",
                "missing_genes": missing[:10],
                "total_missing": len(missing),
                "required_count": len(model.feature_names),
                "provided_count": len(features),
            },
        )

    subtype, probabilities = model.predict(features)
    log.info(
        "predict: %s (LUAD=%.3f LUSC=%.3f) version=%s",
        subtype,
        probabilities["LUAD"],
        probabilities["LUSC"],
        model.version,
    )
    return PredictResponse(
        predicted_subtype=subtype,
        probabilities=probabilities,
        model_version=model.version,
    )


@router.post("/predict", response_model=PredictResponse)
def predict_json(
    body: PredictRequest,
    model: ModelStore = Depends(get_model),
) -> PredictResponse:
    """Predict LUAD/LUSC subtype from a JSON feature vector.

    All genes required by the model must be present in `features`.
    Returns 422 with a list of the first 10 missing gene names on mismatch.
    """
    return _run_predict(body.features, model)


@router.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    file: UploadFile = File(..., description="CSV file with gene_symbol,value columns"),
    model: ModelStore = Depends(get_model),
) -> PredictResponse:
    """Predict LUAD/LUSC subtype from an uploaded CSV expression file.

    Expected CSV format (header optional):
        gene_symbol,value
        TP53,3.14
        EGFR,1.20
        ...
    """
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    features: dict[str, float] = {}
    reader = csv.reader(io.StringIO(text))
    for line_no, row in enumerate(reader, start=1):
        if len(row) < 2:
            continue
        gene, val_str = row[0].strip(), row[1].strip()
        if gene.lower() in ("gene", "gene_symbol", "symbol"):
            continue  # skip header
        try:
            features[gene] = float(val_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Non-numeric value on line {line_no}: {val_str!r}",
            )

    if not features:
        raise HTTPException(status_code=400, detail="No valid gene-value pairs found in file")

    return _run_predict(features, model)
