from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.db_utils import get_engine
from src.model_config import FEATURE_CONFIG
from src.config import BASE_DIR


def load_pipeline_and_data(model_name: str = "random_forest") -> Tuple:
    """Load trained pipeline and create test data."""
    model_path = BASE_DIR / "models" / f"{model_name}_pipeline.joblib"
    pipeline = joblib.load(model_path)

    engine = get_engine()
    df = pd.read_sql("SELECT * FROM telco_churn", con=engine)
    
    X = df[FEATURE_CONFIG.numeric_features + FEATURE_CONFIG.categorical_features].copy()
    y = (df[FEATURE_CONFIG.target_col] == "Yes").astype(int)
    
    # Test set = last 20%
    test_size = int(len(df) * 0.2)
    X_test = X[-test_size:].copy()
    y_test = y[-test_size:].copy()
    
    return pipeline, X_test, y_test


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """Extract feature names AFTER preprocessing."""
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Numeric features unchanged
    num_names = FEATURE_CONFIG.numeric_features
    
    # Categorical features one-hot expanded
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(FEATURE_CONFIG.categorical_features)
    
    return list(num_names) + list(cat_names)


def compute_shap_safe(pipeline: Pipeline, X_test: pd.DataFrame) -> Dict:
    """Compute SHAP values with error handling."""
    import shap
    
    # Extract raw classifier
    rf = pipeline.named_steps["clf"]
    
    # Preprocess test data
    X_processed = pipeline.named_steps["preprocessor"].transform(X_test)
    
    # SHAP explainer for trees
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_processed)
    
    # shap_values[1] for churn class
    return {
        'explainer': explainer,
        'shap_values': shap_values[1],
        'X_processed': X_processed,
        'feature_names': get_feature_names(pipeline),
    }


def create_action_segments(X_test: pd.DataFrame, y_proba: np.ndarray) -> pd.DataFrame:
    """Generate business action recommendations - BULLETPROOF VERSION."""
    segments = X_test[['Contract', 'tenure_bucket', 'MonthlyCharges']].copy()
    segments['churn_probability'] = y_proba
    segments['risk_rank'] = segments['churn_probability'].rank(pct=True)
    
    # Simple np.where() logic - NO CATEGORICALS, NO PROBLEMS
    segments['action_recommendation'] = '📊 LOW: Monitor'
    
    # High risk
    high_risk = segments['risk_rank'] > 0.90
    segments.loc[high_risk, 'action_recommendation'] = 'HIGH: Engagement Call'
    
    # IMMEDIATE for month-to-month high risk
    immediate = high_risk & (segments['Contract'] == 'Month-to-month')
    segments.loc[immediate, 'action_recommendation'] = 'IMMEDIATE: Retention Offer'
    
    return segments.sort_values('churn_probability', ascending=False).head(15)


def main():
    print("Loading pipeline and data...")
    pipeline, X_test, y_test = load_pipeline_and_data()
    
    print("Computing SHAP values...")
    shap_result = compute_shap_safe(pipeline, X_test)
    
    # Get probabilities
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Save for notebook
    shap_data = {
        'shap_result': shap_result,
        'X_test': X_test,
        'y_proba': y_proba,
        'feature_names': shap_result['feature_names'],
    }
    joblib.dump(shap_data, BASE_DIR / "models" / "shap_data.joblib")
    
    # Show action segments
    segments = create_action_segments(X_test, y_proba)
    print("\n=== TOP 15 ACTION SEGMENTS ===")
    print(segments[['churn_probability', 'risk_rank', 'action_recommendation']].round(3))
    
    print(f"\n SHAP data saved to models/shap_data.joblib")
    print(f"{len(shap_result['feature_names'])} features ready for visualization")


if __name__ == "__main__":
    main()
