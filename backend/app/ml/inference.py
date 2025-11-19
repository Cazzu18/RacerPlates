from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from app.ml.features import (
    compute_allergen_count,
    compute_diet_features,
    compute_numeric_features,
    embed_ingredients,
)


THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
FUSION_MODEL_PATH = MODELS_DIR / "sbert_fusion_mlp.joblib"

_fusion_model = None


def _load_fusion_model():
    global _fusion_model
    if _fusion_model is None:
        _fusion_model = joblib.load(FUSION_MODEL_PATH)
    return _fusion_model


class _MealLike:
    def __init__(self, payload: Dict[str, Any]) -> None:
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

    text = payload.get("ingredients") or ""
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


def predict_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    model = _load_fusion_model()
    X = _build_feature_frame(payload)

    proba = model.predict_proba(X)[0]
    classes = getattr(model, "classes_", np.arange(len(proba)))
    best_idx = int(np.argmax(proba))
    label = int(classes[best_idx])

    return {
        "label": label,
        "proba": float(proba[best_idx]),
        "proba_per_class": proba.tolist(),
        "classes": [int(c) for c in classes],
    }

