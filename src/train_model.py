from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_recall_curve,
)

from src.db_utils import get_engine
from src.model_config import FEATURE_CONFIG
from src.config import BASE_DIR

def load_data() -> pd.DataFrame:
    """
    Load the modeling data from SQLite.
    """
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM telco_churn", con=engine)
    return df


def train_test_split_xy(df: pd.DataFrame):
    """
    Split into train/test with stratification on churn label.
    """
    cfg = FEATURE_CONFIG

    X = df[cfg.numeric_features + cfg.categorical_features].copy()
    y = (df[cfg.target_col] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,   # maintain class ratio in both sets
    )

    return X_train, X_test, y_train, y_test

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Scales numeric features
    - One-hot encodes categorical features
    """
    cfg = FEATURE_CONFIG

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cfg.numeric_features),
            ("cat", categorical_transformer, cfg.categorical_features),
        ]
    )

    return preprocessor

def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    """
    Build model pipelines with shared preprocessing.
    """
    models: Dict[str, Pipeline] = {}

    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handle imbalance
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    models["log_reg"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", log_reg),
        ]
    )

    models["random_forest"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", rf),
        ]
    )

    return models

def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_k: float = 0.10,
) -> Dict[str, Any]:
    """
    Evaluate a model with business-aligned metrics.
    """
    # Predicted probabilities for class 1 (churn)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc = roc_auc_score(y_test, y_proba)

    # Classification report (contains recall, precision, f1)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Precision at top K%
    n = len(y_test)
    k = int(n * top_k)
    # Indices sorted by churn probability descending
    sorted_idx = np.argsort(-y_proba)
    top_idx = sorted_idx[:k]
    y_top = y_test.iloc[top_idx]

    precision_at_k = y_top.mean()  # fraction of churners in top K%

    metrics = {
        "model": name,
        "roc_auc": roc,
        "recall_churn": report["1"]["recall"],
        "precision_churn": report["1"]["precision"],
        "precision_at_top_k": precision_at_k,
    }

    print(f"\n=== {name} ===")
    print(f"ROC AUC: {roc:.3f}")
    print(f"Recall (churn class): {metrics['recall_churn']:.3f}")
    print(f"Precision (churn class): {metrics['precision_churn']:.3f}")
    print(f"Precision at top {int(top_k*100)}%: {precision_at_k:.3f}")

    return metrics

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split_xy(df)

    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    all_metrics = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test, top_k=0.10)
        all_metrics[name] = metrics

        # Save model
        model_path = BASE_DIR / "models" / f"{name}_pipeline.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")

    print("\n=== Summary ===")
    for name, metrics in all_metrics.items():
        print(
            f"{name}: ROC_AUC={metrics['roc_auc']:.3f}, "
            f"Recall={metrics['recall_churn']:.3f}, "
            f"P@10%={metrics['precision_at_top_k']:.3f}"
        )


if __name__ == "__main__":
    main()
