from fastapi import APIRouter, HTTPException, Query
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
    is_plant_based: bool = False
    model_name: str | None = None


def _predict_with_model(payload: PredictIn, model_name: str | None = None):
    payload_dict = payload.dict()
    try:
        return predict_from_payload(payload_dict, model_name=model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/")
def predict(payload: PredictIn, model: str | None = Query(default=None)):
    model_name = model or payload.model_name
    pred = _predict_with_model(payload, model_name=model_name)
    return {
        "model": pred.get("model"),
        "label": pred["label"],
        "probability": pred["proba"],
        "proba_per_class": pred.get("proba_per_class"),
        "classes": pred.get("classes"),
    }


@router.post("/{model_slug}")
def predict_by_slug(model_slug: str, payload: PredictIn):
    pred = _predict_with_model(payload, model_name=model_slug)
    return {
        "model": pred.get("model"),
        "label": pred["label"],
        "probability": pred["proba"],
        "proba_per_class": pred.get("proba_per_class"),
        "classes": pred.get("classes"),
    }
