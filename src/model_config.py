from dataclasses import dataclass
from typing import List


@dataclass
class FeatureConfig:
    numeric_features: List[str]
    categorical_features: List[str]
    target_col: str = "Churn"


# IMPORTANT: These names must exist in telco_churn table
FEATURE_CONFIG = FeatureConfig(
    numeric_features=[
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "charge_ratio"
    ],
    categorical_features=[
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure_bucket",
        "high_value",
    ],
    target_col="Churn",
)
