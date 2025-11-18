import pandas as pd
import re
import sys
from pathlib import Path

#Ensuring backend/app is on the Python path so `app.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.session import SessionLocal
from app.db.models import Meal, Rating

#regex to extract "ID: 6991787210"
ID_PATTERN = re.compile(r"ID:\s*(\d+)", re.IGNORECASE)

def extract_menun_item(colname:str):
    match = ID_PATTERN.search(colname)
    
    if not match:
        return None
    
    return int(match.group(1))

def stars_to_label(stars: int | None):
    if stars is None:
        return None
    
    if stars <= 2:
        return 0 #dislike
    
    if stars == 3:
        return 1 #neutral
    
    return 2 #like

def main(csv_path: str):
    df = pd.read_csv(csv_path)
    
    session = SessionLocal()
    
    dietary_col ="Do you have any dietary preferences/restrictions"
    satisfaction_col = "What influences your meal satisfaction most?"
    comments_col = "Any comments about Winslow Dining food?"
    
    rating_columns = [
        col for col in df.columns
        if "ID:" in col
    ]
    print(f"Found {len(rating_columns)} meal rating columns")
    
    for _, row in df.iterrows():
        dietary_pref = row.get(dietary_col)
        satisfaction_factor = row.get(satisfaction_col) 
        comment = row.get(comments_col)
        
        for col in rating_columns:
            stars_raw = row[col]
            
            if pd.isna(stars_raw):
                continue
            
            stars_raw = int(stars_raw)
            
            menu_item_id = extract_menun_item(col)
            
            if menu_item_id is None:
                print(f"Could not extract ID from column: {col}")
                continue
            
            meal = session.query(Meal).filter_by(menu_item_id=menu_item_id).first()
            
            
            if not meal:
                print(f"Meal not found for menu_item_id={menu_item_id}") 
                continue
            
            label = stars_to_label(stars_raw)
            
            rating = Rating(
                meal_id=meal.id,
                label_3class=label,
                stars_5 = stars_raw,
                dietary_pref = dietary_pref,
                satisfaction_factor=satisfaction_factor,
                comment=comment,
            )
            
            session.add(rating)
    
    session.commit()
    session.close()
    
    print("Ratings successfully ingested!")
    

if __name__=="__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_ratings.py Winslow Feedback.csv")
        exit(1)
        
    main(sys.argv[1])
        
    
        
    


