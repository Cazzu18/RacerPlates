from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np
from app.db.models import Meal

_model = None


NUMERIC_BASE_FIELDS = [
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
]


def _safe_number(value):
    if value is None:
        return 0.0, True
    try:
        return float(value), False
    except (TypeError, ValueError):
        return 0.0, True


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-8:
        return 0.0
    return float(numerator) / float(denominator)

def get_embedder():
    global _model
    if _model is None:
        # Force CPU to avoid meta-tensor issues on MPS/accelerator backends.
        _model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return _model


def build_diet_tags(diet_flags: dict, raw_diet_key: str | None = None) -> list[str]:
    tags = []

    raw_tags = [
        tag.strip().replace("_", " ")
        for tag in (raw_diet_key or "").split(",")
        if tag.strip()
    ]
    tags.extend(raw_tags)

    flag_map = [
        ("is_vegan", "vegan"),
        ("is_vegetarian", "vegetarian"),
        ("is_plant_based", "plant based"),
        ("is_mindful", "mindful"),
        ("is_standard", "standard"),
    ]
    for key, label in flag_map:
        if diet_flags.get(key):
            tags.append(label)

    #preserve insertion order, drop dupes
    seen = set()
    deduped = []
    for tag in tags:
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            deduped.append(tag.lower())

    return deduped


def build_embedding_text(
    name: str = "",
    ingredients: str = "",
    diet_key: str = "",
    allergens: str = "",
    nutrition_tags: list | None = None,
    diet_tags: list | None = None,
) -> str:
    """Composing a richer text snippet for SBERT from available meal metadata."""
    parts = []

    name_part = name.strip() if name else ""
    if name_part:
        parts.append(f"Name: {name_part}")

    diet_part = ", ".join(diet_tags or []) if diet_tags else ""
    if diet_part:
        parts.append(f"Diet: {diet_part}")
    else:
        parts.append("Diet: none")

    allergen_part = allergens.strip() if allergens else ""
    if allergen_part:
        parts.append(f"Allergens: {allergen_part}")
    else:
        parts.append("Allergens: none")

    nutri_tags = [t.strip() for t in (nutrition_tags or []) if t]
    if nutri_tags:
        nutri_str = ", ".join(sorted(set(nutri_tags)))
        parts.append(f"Nutrition tags: {nutri_str}")
    else:
        parts.append("Nutrition tags: none")

    ingredients_part = ingredients.strip() if ingredients else ""
    if ingredients_part:
        parts.append(f"Ingredients: {ingredients_part}")

    return ". ".join(parts).strip()

def compute_numeric_features(meal: Meal) -> dict:
    features = {}
    missing_flags = {}
    missing_any = False

    for field in NUMERIC_BASE_FIELDS:
        val = getattr(meal, field, None)
        numeric_val, is_missing = _safe_number(val)
        features[field] = numeric_val
        missing_flags[f"{field}_missing"] = is_missing
        missing_any = missing_any or is_missing

    calories = features["calories"]
    protein = features["protein"]
    carbs = features["carbohydrates"]
    fat = features["fat"]
    sugar = features["sugar"]
    sodium = features["sodium"]
    fiber = features["fiber"]

    features.update(
        {
            "protein_per_calorie": _safe_divide(protein, calories),
            "sugar_to_carb_ratio": _safe_divide(sugar, carbs),
            "protein_to_carb_ratio": _safe_divide(protein, carbs),
            "fat_to_carb_ratio": _safe_divide(fat, carbs),
            "fiber_to_carb_ratio": _safe_divide(fiber, carbs),
            "sodium_per_calorie": _safe_divide(sodium, calories),
            "sugar_per_calorie": _safe_divide(sugar, calories),
            "macro_density": _safe_divide(protein + fat + carbs, calories),
            "log_sodium": np.log1p(max(sodium, 0.0)),
            "log_sugar": np.log1p(max(sugar, 0.0)),
            "log_calories": np.log1p(max(calories, 0.0)),
        }
    )

    features["missing_nutrition"] = missing_any
    features.update(missing_flags)

    return features
    
def _bool_attr(obj, attr: str) -> bool:
    return bool(getattr(obj, attr, False))


def compute_diet_features(meal: Meal) -> dict:

    raw = (meal.diet_key or "").lower()

    #splitting comma separated tags
    tags = {tag.strip() for tag in raw.split(",") if tag.strip()}

    fallback_flags = {
        "is_vegan": _bool_attr(meal, "is_vegan"),
        "is_vegetarian": _bool_attr(meal, "is_vegetarian"),
        "is_mindful": _bool_attr(meal, "is_mindful"),
        "is_plant_based": _bool_attr(meal, "is_plant_based"),
    }

    flags = {
        "is_vegan": ("vegan" in tags),
        "is_vegetarian": ("vegetarian" in tags),
        "is_mindful": ("mindful" in tags),
        "is_plant_based": ("plantbased" in tags) or ("plant_based" in tags),
    }

    for key, fallback in fallback_flags.items():
        if not flags[key] and fallback:
            flags[key] = True

    #derived relationships
    if flags["is_vegan"]:
        flags["is_vegetarian"] = True
        flags["is_plant_based"] = True
    elif flags["is_vegetarian"] and not flags["is_plant_based"]:
        flags["is_plant_based"] = True

    is_standard = ("standard" in tags) or (len(tags) == 0 and not flags["is_vegan"])

    flags["is_standard"] = is_standard

    return flags
    
    
def compute_allergen_count(meal: Meal) -> int:
    if not meal.allergens:
        return 0
    
    return len([a.strip() for a in meal.allergens.split(",") if a.strip()])

def build_nutrition_tags(numeric_features: dict) -> list[str]:
    """
    Coarse discretization of nutrition for SBERT text enrichment only.
    Does not modify numeric processing/values used by models.
    """
    tags: list[str] = []

    calories = float(numeric_features.get("calories", 0.0) or 0.0)
    protein = float(numeric_features.get("protein", 0.0) or 0.0)
    sugar = float(numeric_features.get("sugar", 0.0) or 0.0)
    sodium = float(numeric_features.get("sodium", 0.0) or 0.0)
    fiber = float(numeric_features.get("fiber", 0.0) or 0.0)
    fat = float(numeric_features.get("fat", 0.0) or 0.0)
    carbs = float(numeric_features.get("carbohydrates", 0.0) or 0.0)

    cal_norm = calories if calories > 1e-6 else 100.0
    per_100_cal = lambda val: 100.0 * _safe_divide(val, cal_norm)

    protein_per_100 = per_100_cal(protein)
    sugar_per_100 = per_100_cal(sugar)
    sodium_per_100 = per_100_cal(sodium)
    fiber_per_100 = per_100_cal(fiber)
    fat_per_100 = per_100_cal(fat)
    carbs_per_100 = per_100_cal(carbs)

    if protein_per_100 >= 8.0:
        tags.append("high protein")
    elif protein_per_100 <= 4.0:
        tags.append("low protein")

    if fiber_per_100 >= 3.0:
        tags.append("high fiber")

    if sugar_per_100 >= 10.0:
        tags.append("high sugar")
    elif sugar_per_100 <= 5.0:
        tags.append("low sugar")

    if sodium_per_100 >= 600.0:
        tags.append("high sodium")
    elif sodium_per_100 <= 300.0:
        tags.append("low sodium")

    if fat_per_100 >= 10.0:
        tags.append("high fat")

    if carbs_per_100 <= 10.0:
        tags.append("low carb")

    if calories >= 700:
        tags.append("high calorie")
    elif calories > 0 and calories <= 250:
        tags.append("low calorie")

    #preserve insertion order, drop dupes
    seen = set()
    deduped = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)

    return deduped

    
def embed_ingredients(text: str) -> np.ndarray:
    if not text:
        text = ""

    model = get_embedder()
    emb = model.encode(text, show_progress_bar=False, normalize_embeddings=True)

    return np.asarray(emb, dtype=float)
