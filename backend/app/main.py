from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes_menu, routes_predict, routes_health
from app.core.config import settings 

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