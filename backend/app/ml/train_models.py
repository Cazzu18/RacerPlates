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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import joblib


THIS_DIR = Path(__file__).resolve().parent
DATA_PATH = THIS_DIR / "data" / "processed" / "training_dataset.csv"
MODELS_DIR = THIS_DIR / "models"
REPORTS_DIR = THIS_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)

    #dropping rows without a label just in case
    df = df.dropna(subset=["label_3class"])

    #ensuring int labels
    df["label_3class"] = df["label_3class"].astype(int)

    return df


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def evaluate_model(name, model, X_data, y_data, cv=CV):
    y_pred = cross_val_predict(model, X_data, y_data, cv=cv, n_jobs=-1)
    acc = accuracy_score(y_data, y_pred)
    macro_f1 = f1_score(y_data, y_pred, average="macro")
    report = classification_report(y_data, y_pred, output_dict=True)
    cm = confusion_matrix(y_data, y_pred).tolist() #making JSON serializable

    summary = {
        "name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    #saving detailed report
    report_path = REPORTS_DIR / f"{name}_report.json"
    with report_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")

    return summary


def _make_knn_pipeline(embed_cols):
    embed_transformer = ColumnTransformer(
        transformers=[
            ("emb", "passthrough", embed_cols),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("emb", embed_transformer),
            ("norm", Normalizer(norm="l2")),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=5,
                    metric="euclidean", #on normalized vectors ~ cosine
                    weights="distance",
                ),
            ),
        ]
    )


class WeightedProbEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, mlp_estimator, knn_estimator, weight_mlp=0.6, weight_knn=0.4):
        self.mlp_estimator = mlp_estimator
        self.knn_estimator = knn_estimator
        self.weight_mlp = weight_mlp
        self.weight_knn = weight_knn

    def fit(self, X, y):
        self.mlp_ = clone(self.mlp_estimator)
        self.knn_ = clone(self.knn_estimator)
        self.mlp_.fit(X, y)
        self.knn_.fit(X, y)
        self.classes_ = np.unique(y)
        self._class_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def _align_proba(self, estimator, X):
        proba = estimator.predict_proba(X)
        est_classes = estimator.classes_
        if np.array_equal(est_classes, self.classes_):
            return proba
        aligned = np.zeros((proba.shape[0], len(self.classes_)))
        for col_idx, cls in enumerate(est_classes):
            target_idx = self._class_index[int(cls)]
            aligned[:, target_idx] = proba[:, col_idx]
        return aligned

    def predict_proba(self, X):
        proba_mlp = self._align_proba(self.mlp_, X)
        proba_knn = self._align_proba(self.knn_, X)
        total_weight = self.weight_mlp + self.weight_knn
        blended = (
            self.weight_mlp * proba_mlp + self.weight_knn * proba_knn
        ) / total_weight
        return blended

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.asarray([self.classes_[idx] for idx in indices])


def get_feature_groups(df):
    #columns already in the CSV
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
        "protein_per_calorie",
        "sugar_to_carb_ratio",
        "protein_to_carb_ratio",
        "fat_to_carb_ratio",
        "fiber_to_carb_ratio",
        "sodium_per_calorie",
        "sugar_per_calorie",
        "macro_density",
        "log_sodium",
        "log_sugar",
        "log_calories",
        "missing_nutrition",
        "calories_missing",
        "fat_missing",
        "sugar_missing",
        "protein_missing",
        "carbohydrates_missing",
        "sodium_missing",
        "fiber_missing",
        "cholesterol_missing",
        "iron_missing",
        "calcium_missing",
        "potassium_missing",
    ]

    diet_flag_cols = [
        "is_vegan",
        "is_vegetarian",
        "is_mindful",
        "is_plant_based",
        "is_standard",
    ]

    #embedding columns: emb_0 ... emb_383
    embed_cols = [c for c in df.columns if c.startswith("emb_")]

    return numeric_cols, diet_flag_cols, embed_cols


def train_majority_baseline(X, y):
    #we only use labels, but we still pass X to fit for API consistency
    clf = DummyClassifier(strategy="most_frequent")
    summary = evaluate_model("majority_baseline", clf, X, y)

    clf.fit(X, y)
    joblib.dump(clf, MODELS_DIR / "majority_baseline.joblib")

    return summary


def train_nutrition_logreg(df, X, y):
    numeric_cols, _, _ = get_feature_groups(df)

    #we only use numeric nutrition-style features
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

    summary = evaluate_model("nutrition_logreg", clf, X, y)

    clf.fit(X, y)
    joblib.dump(clf, MODELS_DIR / "nutrition_logreg.joblib")

    return summary

#term frequency-inverse diument frequency: statistical measure used to evaluate how important a word is to a document in a collection
def train_tfidf_logreg(df, X, y):
    if "text" not in df.columns:
        raise ValueError(
            "No 'text' column found. Add it in build_dataset.py (name + ingredients)"
        )

    #we'll use raw text as a series
    text = X["text"].fillna("")

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

    summary = evaluate_model("tfidf_logreg", clf, text, y)

    clf.fit(text, y)
    joblib.dump(clf, MODELS_DIR / "tfidf_logreg.joblib")

    return summary


def train_oracle_knn(df, X, y):
    """
    Oracle-ish model:
    - Uses only the SBERT ingredient embeddings.
    - KNN approximates 'perfect' semantic similarity matching on items.
    This is stronger than TF-IDF and acts as an upper-ish bound on text-only methods.
    """
    _, _, embed_cols = get_feature_groups(df)

    knn_pipeline = _make_knn_pipeline(embed_cols)

    summary = evaluate_model("oracle_knn_embeddings", knn_pipeline, X, y)

    knn_pipeline.fit(X, y)
    joblib.dump(knn_pipeline, MODELS_DIR / "oracle_knn_embeddings.joblib")

    return summary


def train_sbert_fusion_mlp(df, X, y):
    """
    PRIMARY model(ENSAMBLE):
    - Numeric nutrition
    - Diet flags
    - Allergen count
    - SBERT embeddings
    - Output fusion of MLP + KNN
    """
    numeric_cols, diet_flag_cols, embed_cols = get_feature_groups(df)

    feature_cols = numeric_cols + diet_flag_cols + embed_cols

    n_embed_dims = len(embed_cols)
    n_svd_components = max(1, min(128, n_embed_dims - 1))

    emb_pipeline = Pipeline(
        steps=[
            ("norm", Normalizer(norm="l2")),
            (
                "svd",
                TruncatedSVD(
                    n_components=n_svd_components,
                    random_state=42,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_diet", StandardScaler(), numeric_cols + diet_flag_cols),
            ("emb", emb_pipeline, embed_cols),
        ],
        remainder="drop",
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=1000,
        learning_rate_init=1e-3,
        learning_rate="adaptive",
        alpha=1e-3,
        batch_size=16,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42,
    )

    clf = ImbPipeline(
        steps=[
            ("pre", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("mlp", mlp),
        ]
    )

    param_grid = {
        "mlp__hidden_layer_sizes": [(256, 128), (128, 64), (64, 32)],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-3, 5e-4],
        "mlp__batch_size": [16, 32],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = GridSearchCV(
        clf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    X_f = X[feature_cols]
    search.fit(X_f, y)

    best_clf = search.best_estimator_

    print("Best params for sbert_fusion_mlp:", search.best_params_)
    print(f"Best CV macro F1: {search.best_score_:.3f}")

    knn_pipeline = _make_knn_pipeline(embed_cols)
    ensemble = WeightedProbEnsemble(
        mlp_estimator=best_clf,
        knn_estimator=knn_pipeline,
        weight_mlp=0.6,
        weight_knn=0.4,
    )

    summary = evaluate_model("sbert_fusion_mlp", ensemble, X_f, y)

    ensemble.fit(X_f, y)
    joblib.dump(ensemble, MODELS_DIR / "sbert_fusion_mlp.joblib")

    return summary


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = load_data()

    print(f"Dataset shape: {df.shape}")
    print("Label distribution:")
    print(df["label_3class"].value_counts())

    X = df.drop(columns=["label_3class"])
    y = df["label_3class"].values

    summaries = []

    #1) Majority class baseline
    summaries.append(train_majority_baseline(X, y))

    #2) Nutrition-only logistic regression
    summaries.append(train_nutrition_logreg(df, X, y))

    #3) TF-IDF + Logistic regression (text-only)
    summaries.append(train_tfidf_logreg(df, X, y))

    #4) Oracle-ish KNN on SBERT embeddings
    summaries.append(train_oracle_knn(df, X, y))

    #5) SBERT + nutrition + diet fusion MLP
    summaries.append(train_sbert_fusion_mlp(df, X, y))

    #compact comparison table
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
