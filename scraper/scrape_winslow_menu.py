import requests
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

BASE_URL = "https://api-prd.sodexomyway.net/v0.2/data/menu"
LOCATION_ID = 44432001
MENU_ID = 18525

HEADERS = {
    "api-key": "68717828-b754-420d-9488-4c37cb7d7ef7",
    "origin": "https://racerdining.sodexomyway.com",
    "referer": "https://racerdining.sodexomyway.com/",
    "user-agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
}

def fetch_menu(menu_date: str) -> Any:
    url = f"{BASE_URL}/{LOCATION_ID}/{MENU_ID}?date={menu_date}"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()

def normalize_number(value):
    '''Convert '', None, '10mg', '10g', '10' -> float or None'''
    if value is None or value == '':
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    value = value.replace("mg", "").replace("g", "").replace("cal", "").strip()
    
    try:
        return float(value)
    except Exception:
        return None
    
def as_str(v) -> str:
    return v.strip() if isinstance(v, str) else ""

def ingredients_to_str(v) -> str:
    """Sodexo menu JSON has `ingredients` as a long string (or null).
    Occasionally if the shape ever changes to a list, handle that too."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        parts = []
        for x in v:
            if isinstance(x, dict) and "name" in x and x["name"]:
                parts.append(str(x["name"]).strip())
            elif isinstance(x, str):
                parts.append(x.strip())
        return ", ".join(p for p in parts if p)
    return ""

def allergens_to_str(v) -> str:
    """Allergens is an array of objects with `name` keys."""
    if isinstance(v, list):
        return ",".join(
            a["name"].strip()
            for a in v
            if isinstance(a, dict) and a.get("name")
        )
    return ""

def derive_diet_key(item: Dict[str, Any]) -> str:
    """Map Sodexo dietary booleans into a single label."""
    diet_key = ["standard"]
    if item.get("isVegan"):
        if diet_key[0] == "standard":
            diet_key.pop()
        diet_key.append("vegan")
    if item.get("isVegetarian"):
        if diet_key[0] == "standard":
            diet_key.pop()
        diet_key.append("vegetarian")  
    if item.get("isMindful"):
        if diet_key[0] == "standard":
            diet_key.pop()
        diet_key.append("mindful")
    return ", ".join(diet_key) if len(diet_key) > 1 else "standard"

def parse_item(item: Dict[str, Any]) -> Dict[str, Any]:
    station = as_str(item.get("course", ""))
    return {
        "menu_item_id": item.get("menuItemId"),
        "name": as_str(item.get("formalName", "")),
        "ingredients": ingredients_to_str(item.get("ingredients")),
        "allergens": allergens_to_str(item.get("allergens", [])),
        "station": station,
        "diet_key": derive_diet_key(item),
        "calories": normalize_number(item.get("calories")),
        "fat": normalize_number(item.get("fat")),
        "cholesterol": normalize_number(item.get("cholesterol")),
        "sodium": normalize_number(item.get("sodium")),
        "carbohydrates": normalize_number(item.get("carbohydrates")),
        "fiber": normalize_number(item.get("dietaryFiber")),
        "sugar": normalize_number(item.get("sugar")),
        "protein": normalize_number(item.get("protein")),
        "iron": normalize_number(item.get("iron")),
        "calcium": normalize_number(item.get("calcium")),
        "potassium": normalize_number(item.get("potassium")),
        "meal_time": (item.get("meal") or "").upper() or None,
        "is_vegan": bool(item.get("isVegan")),
        "is_vegetarian": bool(item.get("isVegetarian")),
        "is_mindful": bool(item.get("isMindful")),
    }
  
def _iter_sections(raw: Any) -> Iterable[Dict[str, Any]]:
    """
    Yield every section that has an 'items' list.
    Handles a few observed shapes:
      1) {"courseMenus": [ { "items": [...] }, ... ]}
      2) {"data": {"courseMenus": [ ... ]}}
      3) [ {"courseMenus": [ ... ]}, ... ]  (root is list)
    """
    def extract_course_menus(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, dict):
            if "courseMenus" in obj and isinstance(obj["courseMenus"], list):
                return obj["courseMenus"]
            
            data = obj.get("data")
            
            if isinstance(data, dict) and isinstance(data.get("courseMenus"), list):
                return data["courseMenus"]
        return []

    if isinstance(raw, list):
        for part in raw:
            for section in extract_course_menus(part):
                if isinstance(section, dict) and isinstance(section.get("items"), list):
                    yield section
    elif isinstance(raw, dict):
        for section in extract_course_menus(raw):
            if isinstance(section, dict) and isinstance(section.get("items"), list):
                yield section  
    
def main():
    menu_date = date.today().isoformat()
    raw = fetch_menu(menu_date)

    items = []
    
    for meal_block in raw: #raw is a list
        meal_time = (meal_block.get("name") or "").upper() #BREAKFAST / LUNCH / DINNER
        
        for group in meal_block.get("groups", []):
            for item in group.get("items", []):
                parsed = parse_item(item)
                parsed["meal_time"] = meal_time #override, ensures consistency
                items.append(parsed)

    if not items:
        Path("data/debug").mkdir(parents=True, exist_ok=True)
        dbg = Path(f"data/debug/winslow_raw_{menu_date}.json")
        
        with dbg.open("w") as f:
            json.dump(raw, f, indent=2)
            
        print(f"No items found. Wrote raw payload to {dbg} for inspection.")
        return

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    outfile = Path(f"data/raw/winslow_menu_{menu_date}.json")

    with outfile.open("w") as f:
        json.dump(items, f, indent=2)

    print(f"Saved {len(items)} menu items to {outfile}")

    
if __name__ == "__main__":
    main()
