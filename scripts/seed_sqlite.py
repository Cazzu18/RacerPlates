from backend.app.db.session import SessionLocal
from backend.app.db.models import Meal


def main():
    db = SessionLocal()
    try:
        if db.query(Meal).count() == 0:
            demo = [
                Meal(name="Mixed Berry Apple Crisp", allergens="milk, wheat, eggs, gluten, soy", dietKey="vegetarian" calories=190, fat=6, cholesterol=10, sodium=150, carbohydrates=33, fiber=2, sugar=21, protein=2, iron=1, calcium=16, potassium=95, meal_time="dinner"),
                Meal(name="Cheese Pizza", allergens="milk, wheat, gluten", dietKey="vegetarian", calories=250, fat=8, cholesterol=15, sodium=730, carbohydrates=34, fiber=2, sugar=3, protein=11, iron=2, calcium=169, potassium=170, meal_time="dinner"),
                Meal(name="Vanilla Belgian Waffle", allergens="milk, wheat, gluten", dietKey="vegetarian", calories=260, fat=3, cholesterol=0, sodium=1150, carbohydrates=53, fiber=0, sugar=7, protein=5, iron=3, calcium=20, potassium=95, meal_time="breakfast"),
            ]
        