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
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model


def build_embedding_text(
    name: str = "",
    ingredients: str = "",
    diet_key: str = "",
    allergens: str = "",
) -> str:
    """Composing a richer text snippet for SBERT from available meal metadata."""
    parts = []

    if name:
        parts.append(name.strip())
    if ingredients:
        parts.append(ingredients.strip())

    tags = [tag.strip().replace("_", " ") for tag in (diet_key or "").split(",") if tag.strip()]
    if tags:
        parts.append("Diet tags: " + ", ".join(sorted(tags)))

    if allergens:
        parts.append("Allergens: " + allergens.strip())

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


def embed_ingredients(text: str) -> np.ndarray:
    if not text:
        text = ""

    model = get_embedder()
    emb = model.encode(text, show_progress_bar=False, normalize_embeddings=True)

    return np.asarray(emb, dtype=float)
