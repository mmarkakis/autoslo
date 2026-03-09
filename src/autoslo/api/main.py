from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autoslo.api.routers import (
    runner_router,
    simulator_router,
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
app.include_router(simulator_router.router, prefix="/api", tags=["simulator"])
app.include_router(runner_router.router, prefix="/api", tags=["runner"])


@app.get("/")
def root():
    return {
        "message": "AutoSLO API is running",
        "docs": "/docs",
        "redoc": "/redoc",
    }
