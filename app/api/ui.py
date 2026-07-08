from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.api.genes import get_top_genes
from app.db.session import get_db

router = APIRouter(tags=["ui"], include_in_schema=False)

_TEMPLATES_DIR = Path(__file__).parents[1] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _ctx(request: Request, active_page: str = "") -> dict:
    model = request.app.state.model_store
    return {
        "model_version": model.version if model else None,
        "active_page": active_page,
    }


@router.get("/", response_class=HTMLResponse)
def home(request: Request, q: str = "") -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {**_ctx(request, "home"), "query": q},
    )


@router.get("/top-genes", response_class=HTMLResponse)
def top_genes_page(
    request: Request,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    genes = get_top_genes(limit=100, db=db)
    return templates.TemplateResponse(
        request,
        "top_genes.html",
        {**_ctx(request, "top_genes"), "genes": genes},
    )


@router.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request) -> HTMLResponse:
    model = request.app.state.model_store
    feature_count = len(model.feature_names) if model else 0
    return templates.TemplateResponse(
        request,
        "predict.html",
        {**_ctx(request, "predict"), "feature_count": feature_count},
    )
