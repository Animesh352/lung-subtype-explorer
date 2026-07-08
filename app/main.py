from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.de import router as de_router
from app.api.genes import router as genes_router
from app.api.health import router as health_router
from app.api.predict import router as predict_router
from app.api.ui import router as ui_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.model import load_latest_model

setup_logging(level=settings.log_level)

_APP_DIR = Path(__file__).parent
_MODELS_DIR = _APP_DIR.parent / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_store = load_latest_model(_MODELS_DIR)
    app.state.http_client = httpx.Client(timeout=30)
    yield
    app.state.http_client.close()


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(_APP_DIR / "static")), name="static")

# UI routes registered before API routes so / takes precedence over any catch-all
app.include_router(ui_router)

app.include_router(health_router)
app.include_router(genes_router)
app.include_router(de_router)
app.include_router(predict_router)
