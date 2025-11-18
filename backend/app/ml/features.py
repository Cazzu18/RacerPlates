from sentence_transformers import SentenceTransformer
import numpy as np
from app.db.models import Meal

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2") #small, fast model good for <200 items
    return _model

def compute_numeric_features(meal: Meal) -> dict:
    return{
        "calories": meal.calories or 0.0,
        "fat": meal.fat or 0.0,
        "sugar": meal.sugar or 0.0,
        "protein": meal.protein or 0.0,
        "carbohydrates": meal.carbohydrates or 0.0,
        "sodium": meal.sodium or 0.0,
        "fiber": meal.fiber or 0.0,
        "cholesterol": meal.cholesterol or 0.0,
        "iron": meal.iron or 0.0,
        "calcium": meal.calcium or 0.0,
        "potassium": meal.potassium or 0.0,
    }
    
def compute_diet_features(meal: Meal) -> dict:
    
    raw = (meal.diet_key or "").lower()
    
    #splitting comma separated tags
    tags = {tag.strip() for tag in raw.split(",") if tag.strip()}
    
    is_standard = ("standard" in  tags) or (len(tags) == 0)
    
    return {
        "is_vegan": "vegan" in tags,
        "is_vegetarian": "vegetarian" in tags,
        "is_mindful": "mindful" in tags,
        "is_plant_based": "plantbased" in tags or "plant_based" in tags,
        "is_standard": is_standard,
    }
    
    
def compute_allergen_count(meal: Meal) -> int:
    if not meal.allergens:
        return 0
    
    return len([a.strip() for a in meal.allergens.split(",") if a.strip()])


def embed_ingredients(text: str) -> np.ndarray:
    if not text:
        text = ""
        
    model = get_embedder()
    emb = model.encode(text, show_progress_bar=False)
    
    return np.asarray(emb, dtype=float)