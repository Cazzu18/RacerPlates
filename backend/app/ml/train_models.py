import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import joblib

THIS_DIR = Path(__file__).resolve().parent
DATA_PATH = THIS_DIR / "data" / "processed" / "training_dataset.csv"
MODELS_DIR = THIS_DIR / "models"
REPORTS_DIR = THIS_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    
    #dropping rows without a lable just in case
    df = df.dropna(subset=["label_3class"])
    
    #ensuring integer labels
    df["label_3class"] = df["label_3class"].astype(int)
    
    return df

def train_val_split(df, test_size=0.2, random_state=42):
    y = df["label_3class"].values
    X = df.drop(columns=["label_3class"])
    
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        
    )
    
    return X_train, X_val, y_train, y_val

def evaluate_model(name, model, X_val, y_val):
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred).tolist() #json serializable
    
    summary = {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    
    #saving detailed repport
    report_path = REPORTS_DIR/f"{name}_report.json"
    with report_path.open("w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    
    return summary

def get_feature_groups(df):
    numeric_cols = [
        "calories",
        "fat",
        "sugar",
        "protein",
        "carbohydrates",
        "sodium",
        "fiber",
        "cholesterol",
        "iron",
        "calcium",
        "potassium",
        "allergen_count",
    ]

    diet_flag_cols = [
        "is_vegan",
        "is_vegetarian",
        "is_mindful",
        "is_plant_based",
        "is_standard",
    ]
    
    embed_cols = [c for c in df.columns if c.startswith("emb_")]
    
    return numeric_cols, diet_flag_cols, embed_cols

def train_majority_baseline(X_train, y_train, X_val, y_val):
    #uses only labels, but we still pass X to fit for API consitency
    
    #DummyClassifier is a simple non learning model used as a baseline for comparison with more sophisticated classifiers
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    
    joblib.dump(clf, MODELS_DIR / "majority_baseline.joblib")
    
    return evaluate_model("majority_baseline", clf, X_val, y_val)

def train_nutrition_logistic_regression(df, X_train, y_train, X_val, y_val):
    numeric_cols, _, _ = get_feature_groups(df)
    
    #using only numeric nutrition-style features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols)    
        ],
        remainder="drop"
    )
    
    clf = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="multinomial",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODELS_DIR / "nutrition_logreg.joblib")
    
    return evaluate_model("nutrition_logreg", clf, X_val, y_val)

def train_tfidf_logreg(df, X_train, y_train, X_val, y_val):
    if "text" not in df.columns:
        raise ValueError(
            "No 'text' column found. Add it in build_dataset.py (name + ingredients)"
        )

    #raw text as a series
    text_train = X_train["text"].fillna("")
    text_val = X_val["text"].fillna("")

    clf = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="multinomial",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    clf.fit(text_train, y_train)

    joblib.dump(clf, MODELS_DIR / "tfidf_logreg.joblib")

    return evaluate_model("tfidf_logreg", clf, text_val, y_val)


