from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
from types import ModuleType
import logging

import joblib
import numpy as np
import pandas as pd

from app.ml.features import (
    compute_allergen_count,
    compute_diet_features,
    compute_numeric_features,
    build_embedding_text,
    embed_ingredients,
)
from app.ml.train_models import WeightedProbEnsemble as _WeightedProbEnsemble

#ensuring the class is discoverable under legacy pickle module paths
for module_name in ("__main__", "__mp_main__"):
    module = sys.modules.setdefault(module_name, ModuleType(module_name))
    setattr(module, "WeightedProbEnsemble", _WeightedProbEnsemble)
del module, module_name


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
FUSION_MODEL_PATH = MODELS_DIR / "sbert_fusion_mlp.joblib"
ORACLE_MODEL_PATH = MODELS_DIR / "oracle_knn_embeddings.joblib"

DEFAULT_MODEL_NAME = "sbert_fusion_mlp"
MODEL_PATHS: Dict[str, Path] = {
    "sbert_fusion_mlp": FUSION_MODEL_PATH,
    "oracle_knn_embeddings": ORACLE_MODEL_PATH,
}

# supporting a few friendly aliases for the frontend fallbacks
MODEL_ALIASES: Dict[str, str] = {
    "sbert_fusion_mlp": "sbert_fusion_mlp",
    "fusion": "sbert_fusion_mlp",
    "fusion_mlp": "sbert_fusion_mlp",
    "oracle_knn_embeddings": "oracle_knn_embeddings",
    "oracle-knn": "oracle_knn_embeddings",
    "oracle_knn": "oracle_knn_embeddings",
    "knn": "oracle_knn_embeddings",
}

_model_cache: Dict[str, Any] = {}
logger = logging.getLogger(__name__)


def _resolve_model_name(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_MODEL_NAME
    slug = name.strip().lower()
    canonical = MODEL_ALIASES.get(slug)
    if not canonical:
        raise ValueError(f"Unknown model '{name}'. Supported: {list(MODEL_PATHS)}")
    if canonical not in MODEL_PATHS:
        raise ValueError(f"Model '{name}' not available")
    return canonical


def _load_model(name: str):
    if name in _model_cache:
        return _model_cache[name]
    model_path = MODEL_PATHS.get(name)
    if not model_path or not model_path.exists():
        raise ValueError(f"Model artifact for '{name}' is missing")
    _model_cache[name] = joblib.load(model_path)
    return _model_cache[name]


def preload_models() -> None:
    """
    Warm up the embedder and model artifacts once at startup to avoid
    first-request latency or device init issues.
    """
    try:
        embed_ingredients("warmup")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Embedder warmup failed: %s", exc)

    for name in MODEL_PATHS:
        try:
            _load_model(name)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Skipping preload for %s: %s", name, exc)


class _MealLike:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.name = payload.get("name") or payload.get("meal_name") or ""
        self.ingredients = payload.get("ingredients") or ""
        self.calories = payload.get("calories")
        self.fat = payload.get("fat")
        self.sugar = payload.get("sugar")
        self.protein = payload.get("protein")
        self.carbohydrates = payload.get("carbohydrates")
        self.sodium = payload.get("sodium")
        self.fiber = payload.get("fiber")
        self.cholesterol = payload.get("cholesterol")
        self.iron = payload.get("iron")
        self.calcium = payload.get("calcium")
        self.potassium = payload.get("potassium")
        self.diet_key = payload.get("diet_key") or ""
        self.allergens = payload.get("allergens") or ""
        self.is_vegan = bool(payload.get("is_vegan"))
        self.is_vegetarian = bool(payload.get("is_vegetarian"))
        self.is_mindful = bool(payload.get("is_mindful"))
        self.is_plant_based = bool(payload.get("is_plant_based"))


NUMERIC_COLS: List[str] = [
    "calories",
    "fat",
    "sugar",
    "protein",
    "carbohydrates",
    "sodium",
    "fiber",
    "cholesterol",
    "iron",
    "calcium",
    "potassium",
    "allergen_count",
    "protein_per_calorie",
    "sugar_to_carb_ratio",
    "protein_to_carb_ratio",
    "fat_to_carb_ratio",
    "fiber_to_carb_ratio",
    "sodium_per_calorie",
    "sugar_per_calorie",
    "macro_density",
    "log_sodium",
    "log_sugar",
    "log_calories",
    "missing_nutrition",
    "calories_missing",
    "fat_missing",
    "sugar_missing",
    "protein_missing",
    "carbohydrates_missing",
    "sodium_missing",
    "fiber_missing",
    "cholesterol_missing",
    "iron_missing",
    "calcium_missing",
    "potassium_missing",
]

DIET_FLAG_COLS: List[str] = [
    "is_vegan",
    "is_vegetarian",
    "is_mindful",
    "is_plant_based",
    "is_standard",
]


def _build_feature_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    meal = _MealLike(payload)

    numeric = compute_numeric_features(meal)
    diet_flags = compute_diet_features(meal)
    allergen_count = compute_allergen_count(meal)
    numeric["allergen_count"] = allergen_count

    text = build_embedding_text(
        name=meal.name,
        ingredients=meal.ingredients,
        diet_key=meal.diet_key,
        allergens=meal.allergens,
    )
    emb_vec = embed_ingredients(text)
    embed_cols = [f"emb_{i}" for i in range(len(emb_vec))]
    embed_dict = {f"emb_{i}": float(v) for i, v in enumerate(emb_vec)}

    feature_cols = NUMERIC_COLS + DIET_FLAG_COLS + embed_cols

    row: Dict[str, Any] = {}
    for col in NUMERIC_COLS:
        row[col] = numeric.get(col, 0.0)
    for col in DIET_FLAG_COLS:
        row[col] = diet_flags.get(col, False)
    for col in embed_cols:
        row[col] = embed_dict[col]

    return pd.DataFrame([row], columns=feature_cols)


def predict_from_payload(payload: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
    canonical_model = _resolve_model_name(model_name)
    model = _load_model(canonical_model)
    X = _build_feature_frame(payload)

    proba = model.predict_proba(X)[0]
    classes = getattr(model, "classes_", np.arange(len(proba)))
    best_idx = int(np.argmax(proba))
    label = int(classes[best_idx])

    return {
        "model": canonical_model,
        "label": label,
        "proba": float(proba[best_idx]),
        "proba_per_class": proba.tolist(),
        "classes": [int(c) for c in classes],
    }
