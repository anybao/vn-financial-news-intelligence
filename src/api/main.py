import uvicorn
import logging
import os
from logging.handlers import RotatingFileHandler

# ── Configure logging BEFORE any app imports ──────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")

_log_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_root = logging.getLogger()
_root.setLevel(logging.INFO)

# Console handler
_console = logging.StreamHandler()
_console.setFormatter(_log_fmt)
_root.addHandler(_console)

# File handler – rotates at 5 MB, keeps 3 backups
_file = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
_file.setFormatter(_log_fmt)
_root.addHandler(_file)

# ── Now import app modules ────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api.routes import router as api_router
from mlops.prometheus_metrics import prometheus_middleware
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import make_asgi_app

def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial News Intelligence API",
        description=(
            "## NLP501 Final Project — Vietnamese Financial News Intelligence\n\n"
            "End-to-end NLP pipeline for Vietnamese financial news articles:\n\n"
            "- **Summarization** — 3 switchable models: ViT5 (pre-trained T5), "
            "Seq2Seq (BiLSTM + Bahdanau Attention + Beam Search), Extractive (TF scoring)\n"
            "- **Sentiment Analysis** — PhoBERT 3-class classifier (Positive / Neutral / Negative)\n"
            "- **Named Entity Recognition** — Hybrid VN30 stock extraction (dictionary + PhoBERT NER)\n\n"
            "### Summarization Model Switch\n\n"
            "Pass `\"model\": \"vit5\"` / `\"seq2seq\"` / `\"extractive\"` in the request body "
            "to choose the summarization engine.\n\n"
            "### Quick Start\n\n"
            "```bash\n"
            'curl -X POST http://localhost:8000/api/v1/summarize \\\n'
            '  -H "Content-Type: application/json" \\\n'
            '  -d \'{"text": "Hòa Phát công bố...", "model": "vit5"}\'\n'
            "```"
        ),
        version="1.0.0",
        contact={"name": "NLP501 Team", "url": "https://github.com"},
        license_info={"name": "MIT"},
    )
    
    # Add prometheus middleware
    app.add_middleware(BaseHTTPMiddleware, dispatch=prometheus_middleware)
    
    # Include application routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Mount prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirects root to the Dashboard"""
        return RedirectResponse(url="/api/v1/dashboard")

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
