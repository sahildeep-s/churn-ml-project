import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent if "tests" in str(Path(__file__)) else Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

def load_model(name: str):
    return joblib.load(MODELS_DIR / f"{name}_pipeline.joblib")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["charge_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)

    bins = [0, 6, 12, 24, 72]
    labels = ["new_risk", "early_risk", "mid_term", "loyal"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)

    # Same approximation used in app (keep consistent)
    df["high_value"] = ((df["tenure"] > 24) & (df["MonthlyCharges"] > 80)).astype(int)
    return df

def build_scenarios() -> pd.DataFrame:
    rows = [
        # Low risk
        dict(scenario="low_loyal", tenure=60, MonthlyCharges=40, TotalCharges=2400,
             gender="Female", SeniorCitizen=0, Partner="Yes", Dependents="Yes",
             PhoneService="Yes", MultipleLines="No", InternetService="DSL",
             OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes",
             StreamingTV="No", StreamingMovies="No",
             Contract="Two year", PaperlessBilling="No",
             PaymentMethod="Bank transfer (automatic)"),
        # High risk
        dict(scenario="high_m2m_expensive", tenure=2, MonthlyCharges=110, TotalCharges=220,
             gender="Male", SeniorCitizen=1, Partner="No", Dependents="No",
             PhoneService="Yes", MultipleLines="Yes", InternetService="Fiber optic",
             OnlineBackup="No", DeviceProtection="No", TechSupport="No",
             StreamingTV="Yes", StreamingMovies="Yes",
             Contract="Month-to-month", PaperlessBilling="Yes",
             PaymentMethod="Electronic check"),
        # Medium-ish
        dict(scenario="mid_m2m_midprice", tenure=10, MonthlyCharges=70, TotalCharges=700,
             gender="Male", SeniorCitizen=0, Partner="No", Dependents="No",
             PhoneService="Yes", MultipleLines="No", InternetService="DSL",
             OnlineBackup="No", DeviceProtection="No", TechSupport="No",
             StreamingTV="No", StreamingMovies="No",
             Contract="Month-to-month", PaperlessBilling="Yes",
             PaymentMethod="Mailed check"),
        # Another low risk variant
        dict(scenario="low_two_year_high_value", tenure=70, MonthlyCharges=115, TotalCharges=8050,
             gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="No",
             PhoneService="Yes", MultipleLines="No", InternetService="DSL",
             OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes",
             StreamingTV="No", StreamingMovies="No",
             Contract="Two year", PaperlessBilling="No",
             PaymentMethod="Credit card (automatic)"),
    ]
    df = pd.DataFrame(rows)
    df = engineer_features(df)
    return df

def run_for_model(model_name: str):
    model = load_model(model_name)
    df = build_scenarios()

    feature_cols = [
        "tenure","MonthlyCharges","TotalCharges","charge_ratio",
        "gender","SeniorCitizen","Partner","Dependents","PhoneService",
        "MultipleLines","InternetService","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies","Contract",
        "PaperlessBilling","PaymentMethod","tenure_bucket","high_value"
    ]
    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    out = df[["scenario"]].copy()
    out["churn_probability"] = proba
    out = out.sort_values("churn_probability", ascending=False)

    print(f"\n=== {model_name} scenario results ===")
    print(out.to_string(index=False))

    # Assertions: ordering constraints (ranking-based testing)
    p = dict(zip(out["scenario"], out["churn_probability"]))
    assert p["high_m2m_expensive"] > p["mid_m2m_midprice"], "High risk should rank above mid risk"
    assert p["mid_m2m_midprice"] > p["low_loyal"], "Mid risk should rank above low risk"

if __name__ == "__main__":
    run_for_model("log_reg")
    run_for_model("random_forest")
    print("\nAll scenario ranking tests passed.")
