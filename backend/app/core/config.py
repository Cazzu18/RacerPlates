from pydantic import BaseSettings
from typing import List
from pathlib import Path

#Project root directory (three levels above `backend/app/core`)
BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SQLITE_PATH = BASE_DIR / "backend" / "app" / "db" / "db.sqlite3"


class Settings(BaseSettings):
    DB_URL: str = f"sqlite:///{DEFAULT_SQLITE_PATH}"
    MODEL_PATH: str = str(
        BASE_DIR / "backend" / "app" / "ml" / "artifacts" / "model.pk1"
    )
    SBERT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"


# Singleton settings instance used across the app
settings = Settings()
