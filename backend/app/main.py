import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_menu, routes_predict, routes_health
from app.core.config import settings 
from app.ml.inference import preload_models

#to reduce noisy tokenizers fork warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = FastAPI(title="RacerPlates API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router, prefix="/health", tags=["health"])
app.include_router(routes_menu.router, prefix="/menu", tags=["menu"])
app.include_router(routes_predict.router, prefix="/predict", tags=["predict"])


@app.on_event("startup")
def _warm_models():
    preload_models()
