from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autoslo.api.routers import (
    classifier_router,
    composite_router,
    chunk_router,
    strat_router,
)

app = FastAPI(title="AutoSLO API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(composite_router.router, prefix="/api", tags=["composite"])
app.include_router(chunk_router.router, prefix="/api", tags=["chunk"])
app.include_router(strat_router.router, prefix="/api", tags=["strat"])
app.include_router(classifier_router.router, prefix="/api", tags=["classifier"])


@app.get("/")
def root():
    return {
        "message": "AutoSLO API is running",
        "docs": "/docs",
        "redoc": "/redoc",
    }
