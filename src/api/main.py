import uvicorn
from fastapi import FastAPI
from src.api.routes import router as api_router
from mlops.prometheus_metrics import prometheus_middleware
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import make_asgi_app

def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial News Intelligence API",
        description="NLP501 Final Project API exposing End-to-End NLP components.",
        version="1.0.0"
    )
    
    # Add prometheus middleware
    app.add_middleware(BaseHTTPMiddleware, dispatch=prometheus_middleware)
    
    # Include application routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Mount prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
