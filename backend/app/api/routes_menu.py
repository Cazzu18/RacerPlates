from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.db.models import Meal

#API router is used to group related endpoints together
router = APIRouter()

def get_db():
    db = SessionLocal()
    
    try:
        #used to create generator functions, which are special types of iterators that allow values to be produced lazily, one at a time, instead of returning them all at once
        yield db
    finally:
        db.close()
        
#@ used to apply decorators in FastAPI. @router.get("/") decorates list_menu, mkaing it the handler for HTTP GET requests from root path "/"
@router.get("/")
def list_menu(db: Session = Depends(get_db)):
    meals = db.query(Meal).order_by(Meal.id.desc()).all()
    
    #returning a list of hashtables of meals and attributes
    return [dict(
        id= meal.id, 
        menu_item_id=meal.menu_item_id,
        name=meal.name, 
        allergens=meal.allergens, 
        station=meal.station,
        diet_key=meal.diet_key, 
        calories=meal.calories, 
        fat=meal.fat, 
        cholesterol=meal.cholesterol, 
        sodium=meal.sodium, 
        carbohydrates=meal.carbohydrates, 
        fiber=meal.fiber, 
        sugar=meal.sugar, 
        protein=meal.protein, 
        iron=meal.iron, 
        calcium=meal.calcium, 
        potassium=meal.potassium, 
        meal_time=meal.meal_time,
        is_vegan=meal.is_vegan,
        is_vegetarian=meal.is_vegetarian,
        is_mindful=meal.is_mindful
        
        ) for meal in meals]
