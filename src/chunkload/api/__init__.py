from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import chunk as chunk_router
from .routes import composite as composite_router

def create_app(ui_dir: str | None = None, allow_origins: list[str] | None = None):
    app = FastAPI(
        title="chunkload API",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(chunk_router.router, prefix="/api")
    app.include_router(composite_router.router, prefix="/api")

    if ui_dir:
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    return app

# expose a default app for uvicorn: `uvicorn chunkload.api:app --reload`
app = create_app()