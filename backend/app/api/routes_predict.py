from fastapi import APIRouter
from pydantic import BaseModel
from app.ml.inference import predict_from_payload

router = APIRouter()


class PredictIn(BaseModel):
    menu_item_id: int | None = None
    name: str
    ingredients: str = ""
    allergens: str = ""
    station: str = ""
    diet_key: str = ""
    calories: float | None = None
    fat: float | None = None
    cholesterol: float | None = None
    sodium: float | None = None
    carbohydrates: float | None = None
    fiber: float | None = None
    sugar: float | None = None
    protein: float | None = None
    iron: float | None = None
    calcium: float | None = None
    potassium: float | None = None
    meal_time: str | None = None
    is_vegan: bool = False
    is_vegetarian: bool = False
    is_mindful: bool = False


@router.post("/")
def predict(payload: PredictIn):
    payload_dict = payload.dict()
    pred = predict_from_payload(payload_dict)
    return {"label": pred["label"], "probability": pred["proba"]}
