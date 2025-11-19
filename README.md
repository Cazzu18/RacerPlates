# RacerPlates

Data products for Racer Dining that predict how Murray State students will react to each dining-hall meal. The repository bundles a FastAPI backend, a lightweight Next.js dashboard, utilities for scraping Sodexo's public menu feed, and—most importantly—a full machine-learning pipeline that fuses nutrition, allergen, diet-tag, and text-embedding signals to classify satisfaction labels (`0 = dislike`, `1 = neutral`, `2 = like`).

---

## Overview
- **Goal:** forecast crowd sentiment for every Winslow Dining menu item so dining staff can highlight winners, retire underperformers, and balance dietary preferences.
- **Core idea:** combine structured nutrition data with a SentenceTransformer embedding of the name/ingredients/diet tags, then train several models culminating in a fusion ensemble that blends an MLP (trained on engineered features) with a semantic KNN.
- **Stack:** FastAPI + SQLite for serving/storing data, scikit-learn/imbalanced-learn + sentence-transformers for ML, Next.js 16 for the UI, and a set of Python scripts for ingestion.

---

## Repository Layout
| Path | Purpose |
| --- | --- |
| `backend/app/api/` | FastAPI routers for `/menu`, `/predict`, and `/health`. |
| `backend/app/core/config.py` | Environment-driven settings (`DB_URL`, `SBERT_MODEL`, CORS list, etc.). |
| `backend/app/db/` | SQLAlchemy models (`Meal`, `Rating`) plus the SQLite session factory. |
| `backend/app/ml/` | Complete ML stack: feature engineering, dataset builder, training script, inference helpers, stored models & reports. |
| `frontend/` | Next.js 16 dashboard that surfaces menu items and hits the prediction API. |
| `scraper/` | Pulls daily menus from the Sodexo JSON API and saves normalized payloads. |
| `scripts/` | Operational utilities: initialize SQLite, seed meals, ingest survey ratings, clear labels. |
| `surveys/` | Raw Winslow student feedback (`Winslow Feedback.csv`) consumed by `ingest_ratings.py`. |
| `ml/notebooks/` | Jupyter notebooks for ad hoc exploration and reporting. |

---

## Machine Learning Pipeline

### Data Sources
1. **Menu metadata** – `scraper/scrape_winslow_menu.py` fetches the public menu feed (`BASE_URL = https://api-prd.sodexomyway.net/v0.2/data/menu`) with authentication headers, normalizes nutrition fields, collapses allergen/diet arrays into CSV strings, and persists JSON to `scraper/data/raw/`.
2. **Database seeding** – `scripts/seed_meals.py <json>` loads that JSON into the `meals` table. `scripts/init_sqlite.py` bootstraps `backend/app/db/db.sqlite3`.
3. **Survey labels** – `scripts/ingest_ratings.py "Winslow Feedback.csv"` parses the Qualtrics-style export, extracts `menu_item_id` from `ID: ####` column headers, and maps 1–5 stars into `label_3class` (dislike/neutral/like). Comments, dietary preferences, and satisfaction factors feed into the ratings table for future NLP experiments.
4. **Dataset assembly** – `backend/app/ml/build_dataset.py` joins `Meal` and `Rating` rows, computes numeric/diet/allergen features, adds SBERT embeddings, and emits `backend/app/ml/data/processed/training_dataset.csv`.

### Feature Engineering (`backend/app/ml/features.py`)
- **Raw nutrition vectors:** calories, macro/micronutrients converted to floats with missingness flags for each field.
- **Derived ratios/log features:** e.g., protein-per-calorie, sugar-to-carb, macro density, log-transformed sodium/sugar/calories.
- **Diet flags:** parse `diet_key` and fallback booleans (`is_vegan`, `is_vegetarian`, `is_mindful`, `is_plant_based`, `is_standard`).
- **Allergen count:** simple length of the allergen list to capture restriction risk.
- **Text embeddings:** compose `name + ingredients + diet tags + allergens` and embed with `SentenceTransformer("all-mpnet-base-v2")`, yielding 768-d normalized vectors.

### Models (`backend/app/ml/train_models.py`)
Every model is evaluated with `StratifiedKFold(n_splits=5)` via `cross_val_predict` to get unbiased accuracy + macro-F1 scores before fitting on the full dataset. Saved artifacts live in `backend/app/ml/models/`.

| Model | Description | File |
| --- | --- | --- |
| `majority_baseline` | DummyClassifier predicting the most frequent label; sanity check. | `majority_baseline.joblib` |
| `nutrition_logreg` | StandardScaler + multinomial LogisticRegression on numeric nutrition features only. | `nutrition_logreg.joblib` |
| `tfidf_logreg` | TF‑IDF (1–2 grams, 5k vocab) on the raw text column feeding a logistic regression. | `tfidf_logreg.joblib` |
| `oracle_knn_embeddings` | KNN (k=5, distance-weighted, euclidean on L2-normalized SBERT embeddings). | `oracle_knn_embeddings.joblib` |
| `sbert_fusion_mlp` | **Primary model.** ColumnTransformer splits numeric/diet vs embeddings, uses SVD (≤128 comps) to compress embeddings, passes everything through StandardScaler + SMOTE-balanced MLP (hidden layers tuned via GridSearchCV). Output is ensembled with the SBERT KNN through a `WeightedProbEnsemble` for probability blending. | `sbert_fusion_mlp.joblib` |

`backend/app/ml/reports/` stores per-model JSON reports (accuracy, macro-F1, classification reports, confusion matrices) plus `model_comparison.json`. Current snapshot:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Majority baseline | 0.601 | 0.250 |
| Nutrition LogReg | 0.311 | 0.300 |
| TF-IDF LogReg | 0.294 | 0.284 |
| SBERT KNN | 0.378 | 0.343 |
| **SBERT Fusion MLP** | **0.441** | **0.364** |

### Notebooks
- `ml/notebooks/01_explore_data.ipynb` (placeholder for quick EDA).
- `ml/notebooks/02_train_report.ipynb` – summarize experiments, confusion matrices, and feature behavior; handy for stakeholder presentations.

---

## Reproducing the ML Workflow

### 1. Environment
```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```
Set any overrides in `.env` (same directory as `backend/`). Key variables:
- `DB_URL` – defaults to `sqlite:///backend/app/db/db.sqlite3`.
- `SBERT_MODEL` – override if you want a different SentenceTransformer.
- `CORS_ORIGINS` – comma-separated hosts for the FastAPI middleware.

### 2. Ingest menu + survey data
```bash
# Initialize DB schema
python scripts/init_sqlite.py

# Pull today's menu and write JSON
python scraper/scrape_winslow_menu.py

# Push menu JSON into SQLite
python scripts/seed_meals.py scraper/data/raw/winslow_menu_<date>.json

# Load survey labels (edit path if you have a new export)
python scripts/ingest_ratings.py "surveys/Winslow Feedback.csv"
```
Use `scripts/clear_ratings.py` if you need to restart the labeling process.

### 3. Build the training dataset
```bash
python backend/app/ml/build_dataset.py
# -> backend/app/ml/data/processed/training_dataset.csv
```

### 4. Train the models
```bash
python backend/app/ml/train_models.py
# Models -> backend/app/ml/models/
# Reports -> backend/app/ml/reports/
```
The script also prints a leaderboard and writes `model_comparison.json` for easy plotting. Expect SBERT downloads (~400 MB) during the first run.

---

## Serving Predictions

### Backend API
```bash
uvicorn backend.app.main:app --reload
```
- `GET /health/` – liveness.
- `GET /menu/` – serialized meals from SQLite (recent first).
- `POST /predict/` – accepts the schema from `backend/app/api/routes_predict.py`. Payload must include at least a `name`; nutrition/allergen fields improve accuracy.

Example request:
```jsonc
POST http://localhost:8000/predict/
{
  "name": "Vegan BBQ Jackfruit Sandwich",
  "ingredients": "jackfruit, vegan BBQ sauce, pickled onions, whole wheat bun",
  "diet_key": "vegan, mindful",
  "calories": 410,
  "protein": 18,
  "carbohydrates": 55,
  "fat": 12,
  "sodium": 620,
  "allergens": "soy, wheat",
  "is_vegan": true,
  "is_mindful": true
}
```
Response:
```json
{
  "label": 2,
  "probability": 0.71,
  "proba_per_class": [0.08, 0.21, 0.71],
  "classes": [0, 1, 2]
}
```
`backend/app/ml/inference.py` mirrors the feature pipeline: it recreates numeric/diet/allergen features, embeds text with the same SBERT encoder, loads `sbert_fusion_mlp.joblib`, and emits calibrated probabilities. The helper also injects the ensemble class definition into `__main__` so that `joblib` can unpickle the custom estimator.

### Frontend dashboard
```bash
cd frontend
npm install
npm run dev
# NEXT_PUBLIC_BACKEND_URL controls which FastAPI instance the UI hits
```
The UI fetches `/menu` to populate cards and allows staff to request predictions (and compare the fusion MLP with the SBERT-KNN oracle) via the helper in `frontend/app/(lib)/api.ts`.

---

## Supporting Utilities
- `scripts/seed_meals.py` – idempotently inserts/updates rows keyed by `menu_item_id`.
- `scripts/ingest_ratings.py` – maps 1–5 star responses to `label_3class`, attaches qualitative survey context.
- `scripts/clear_ratings.py` – truncate ratings during experimentation.
- `scraper/scrape_winslow_menu.py` – maintainable ingest code with strong typing + helper functions for normalization.
- `backend/app/ml/model_card.md` – reserved for a future model-card summary (currently empty).

---

## Extending the ML System
1. **Better labels:** incorporate free-form comments via sentiment analysis or topic modeling, then feed the insights into the `build_dataset.py` rows.
2. **Temporal awareness:** append seasonal/meal-time signals (weekday, event days, etc.).
3. **Active learning loop:** surface low-confidence predictions in the dashboard so dining staff can request new survey feedback targeted at uncertain meals.
4. **Auto model selection:** expose `ModelVariant` choice in the API (the frontend already anticipates `/predict/<model>` routes).

Contribution ideas, bug reports, and new survey instruments are always welcome—open an issue or PR describing how your change improves the ML signal or the serving pipeline.
