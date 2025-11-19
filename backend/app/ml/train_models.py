import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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

    #Drop rows without a label just in case
    df = df.dropna(subset=["label_3class"])

    #Ensure int labels
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
    cm = confusion_matrix(y_val, y_pred).tolist()  #make JSON serializable

    summary = {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    #Save detailed report
    report_path = REPORTS_DIR / f"{name}_report.json"
    with report_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")

    return summary


def get_feature_groups(df):
    #Columns already in your CSV
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

    #Embedding columns: emb_0 ... emb_383
    embed_cols = [c for c in df.columns if c.startswith("emb_")]

    return numeric_cols, diet_flag_cols, embed_cols


def train_majority_baseline(X_train, y_train, X_val, y_val):
    #Uses only labels, but we still pass X to fit for API consistency
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)

    joblib.dump(clf, MODELS_DIR / "majority_baseline.joblib")

    return evaluate_model("majority_baseline", clf, X_val, y_val)


def train_nutrition_logreg(df, X_train, y_train, X_val, y_val):
    numeric_cols, _, _ = get_feature_groups(df)

    #Use only numeric nutrition-style features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
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

#term frequency-inverse diument frequency: statistical measure used to evaluate how important a word is to a document in a collection
def train_tfidf_logreg(df, X_train, y_train, X_val, y_val):
    if "text" not in df.columns:
        raise ValueError(
            "No 'text' column found. Add it in build_dataset.py (name + ingredients)"
        )

    #We'll use raw text as a series
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


def train_oracle_knn(df, X_train, y_train, X_val, y_val):
    """
    Oracle-ish model:
    - Uses only the SBERT ingredient embeddings.
    - KNN approximates 'perfect' semantic similarity matching on items.
    This is stronger than TF-IDF and acts as an upper-ish bound on text-only methods.
    """
    _, _, embed_cols = get_feature_groups(df)

    #Extract embedding matrices
    X_train_emb = X_train[embed_cols].values
    X_val_emb = X_val[embed_cols].values

    #L2 normalize embeddings to approximate cosine distance with Euclidean
    def l2_normalize(mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return mat / norms

    X_train_norm = l2_normalize(X_train_emb)
    X_val_norm = l2_normalize(X_val_emb)

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean",  #on normalized vectors ~ cosine
        weights="distance",
    )

    knn.fit(X_train_norm, y_train)

    joblib.dump(knn, MODELS_DIR / "oracle_knn_embeddings.joblib")

    return evaluate_model("oracle_knn_embeddings", knn, X_val_norm, y_val)


def train_sbert_fusion_mlp(df, X_train, y_train, X_val, y_val):
    """
    Full model:
    - Numeric nutrition
    - Diet flags
    - Allergen count
    - SBERT embeddings
    """
    numeric_cols, diet_flag_cols, embed_cols = get_feature_groups(df)

    feature_cols = numeric_cols + diet_flag_cols + embed_cols

    X_train_f = X_train[feature_cols]
    X_val_f = X_val[feature_cols]

    #ColumnTransformer:
    #- scale numeric + diet flags
    #- passthrough embeddings
    num_and_diet_cols = numeric_cols + diet_flag_cols
    indices_num_and_diet = [X_train_f.columns.get_loc(c) for c in num_and_diet_cols]
    indices_embeds = [X_train_f.columns.get_loc(c) for c in embed_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_diet", StandardScaler(), indices_num_and_diet),
            ("emb", "passthrough", indices_embeds),
        ]
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=100,
        learning_rate_init=1e-3,
        random_state=42,
    )

    clf = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("mlp", mlp),
        ]
    )

    clf.fit(X_train_f, y_train)

    joblib.dump(clf, MODELS_DIR / "sbert_fusion_mlp.joblib")

    return evaluate_model("sbert_fusion_mlp", clf, X_val_f, y_val)


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = load_data()

    print(f"Dataset shape: {df.shape}")
    print("Label distribution:")
    print(df["label_3class"].value_counts())

    X_train, X_val, y_train, y_val = train_val_split(df)

    summaries = []

    #1) Majority class baseline
    summaries.append(train_majority_baseline(X_train, y_train, X_val, y_val))

    #2) Nutrition-only logistic regression
    summaries.append(train_nutrition_logreg(df, X_train, y_train, X_val, y_val))

    #3) TF-IDF + Logistic regression (text-only)
    summaries.append(train_tfidf_logreg(df, X_train, y_train, X_val, y_val))

    #4) Oracle-ish KNN on SBERT embeddings
    summaries.append(train_oracle_knn(df, X_train, y_train, X_val, y_val))

    #5) SBERT + nutrition + diet fusion MLP
    summaries.append(train_sbert_fusion_mlp(df, X_train, y_train, X_val, y_val))

    #Save a compact comparison table
    comp = [
        {
            "name": s["name"],
            "accuracy": s["accuracy"],
            "macro_f1": s["macro_f1"],
        }
        for s in summaries
    ]
    comp_path = REPORTS_DIR / "model_comparison.json"
    with comp_path.open("w") as f:
        json.dump(comp, f, indent=2)

    print("\n=== MODEL COMPARISON ===")
    for row in comp:
        print(
            f"{row['name']:25s} | acc={row['accuracy']:.3f} | macro_f1={row['macro_f1']:.3f}"
        )
    print(f"\nSaved detailed reports to: {REPORTS_DIR}")
    print(f"Saved comparison table   to: {comp_path}")


if __name__ == "__main__":
    main()
