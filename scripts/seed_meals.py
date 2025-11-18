import json
import sys
from pathlib import Path

#Ensure backend/app is on the Python path so `app.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.session import SessionLocal
from app.db.models import Meal


def load_json(path:sys):
    p = Path(path)
    
    if not p.exists():
        return FileNotFoundError(f"JSON File not found: {path}")
    with p.open() as f:
        return json.load(f)     
    
def upsert_meals(session, meal_dict):
    existing = (
        session.query(Meal)
        .filter(Meal.menu_item_id == meal_dict["menu_item_id"])
        .first()
    )
    
    if existing:
        #update existing record
        for k, v in meal_dict.items():
            setattr(existing,k, v)
    
    #unpacking data from json file into Meal object
    meal = Meal(**meal_dict)
    
    session.add(meal)
    return meal

def main():
    if len(sys.argv) < 2:
        print("Usage: python seed_meals.py <json_file>")
        sys.exit(1)
        
    json_path = sys.argv[1]
    
    data = load_json(json_path)
    
    db = SessionLocal()
    
    print(f"Loading {len(data)} meals from {json_path}...")
    
    count = 0
    
    for item in data:
        upsert_meals(db, item)
        count += 1
        
    db.commit()
    
    print(f"Successfully inserted/update {count} meals.")

if __name__ == "__main__":
    main()
