import pandas as pd
from pathlib import Path
import sys

# Ensure backend/app is on the Python path so `app.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.session import SessionLocal
from app.db.models import Meal, Rating

from app.ml.features import (
    compute_numeric_features,
    compute_diet_features,
    compute_allergen_count,
    embed_ingredients,
)

def build_training_dataframe():
    session = SessionLocal()
    
    meals = {m.id: m for m in session.query(Meal).all()}
    ratings = session.query(Rating).all()
    
    rows = []
    
    for r in ratings:
        meal = meals.get(r.meal_id)
        
        if meal is None:
            continue
        
        numeric = compute_numeric_features(meal)
        diet = compute_diet_features(meal)
        allergen_count = compute_allergen_count(meal)
        embed_vec = embed_ingredients(meal.ingredients)
        
        row = {
            "meal_id": meal.id,
            "menu_item_id": meal.menu_item_id,
            "label_3class": r.label_3class,
            "stars_5": r.stars_5,
            "dietary_pref": r.dietary_pref,
            "satisfaction_factor": r.satisfaction_factor,
            **numeric,
            **diet,
            "allergen_count": allergen_count,
        }
        
        for i,val in enumerate(embed_vec):
            row[f"emb_{i}"] = val
            
        rows.append(row)
        
    session.close()
    
    df = pd.DataFrame(rows)
    return df

def main():
    df = build_training_dataframe()

    out = PROJECT_ROOT / "backend" / "app" / "ml" / "data" / "processed" / "training_dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Saved training dataset", out)
    print(df.head())


if __name__ == "__main__":
    main()
    
