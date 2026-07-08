from __future__ import annotations

import httpx
from fastapi import HTTPException, Request

from app.services.model import ModelStore


def get_http_client(request: Request) -> httpx.Client:
    return request.app.state.http_client


def get_model(request: Request) -> ModelStore:
    store: ModelStore | None = request.app.state.model_store
    if store is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return store
