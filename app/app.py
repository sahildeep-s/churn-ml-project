import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import math

from src.config import BASE_DIR
from src.model_config import FEATURE_CONFIG

# Custom CSS for eye-pleasing design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .high-risk {
        background-color: #fee;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #d62728;
    }
    .low-risk {
        background-color: #f0f9f0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models_dir = BASE_DIR / "models"
    log_reg_path = models_dir / "log_reg_pipeline.joblib"
    rf_path = models_dir / "random_forest_pipeline.joblib"
    
    log_reg_model = joblib.load(log_reg_path)
    rf_model = joblib.load(rf_path)
    
    return {
        "Logistic Regression (Best Precision@10%)": log_reg_model,
        "Random Forest (Robust)": rf_model,
    }

models = load_models()

# ---------- HELPER FUNCTIONS (Unchanged) ----------
def get_empty_input_df():
    cols = FEATURE_CONFIG.numeric_features + FEATURE_CONFIG.categorical_features
    return pd.DataFrame(columns=cols)

def build_single_customer_df(**kwargs):
    df = get_empty_input_df()
    
    row = kwargs.copy()
    # Auto-compute TotalCharges if not provided
    if 'total_charges' in row:
        row["TotalCharges"] = row['total_charges']
    
    # Recompute engineered features
    row["charge_ratio"] = row["TotalCharges"] / (row["tenure"] + 1)
    
    # Tenure bucket
    if row["tenure"] < 6:
        row["tenure_bucket"] = "new_risk"
    elif row["tenure"] < 12:
        row["tenure_bucket"] = "early_risk"
    elif row["tenure"] < 24:
        row["tenure_bucket"] = "mid_term"
    else:
        row["tenure_bucket"] = "loyal"
    
    # High value
    row["high_value"] = int((row["tenure"] > 24) and (row["MonthlyCharges"] > 80))
    
    df.loc[0] = row
    return df

def score_single_customer(model, customer_df):
    proba = model.predict_proba(customer_df)[:, 1][0]
    label = proba >= 0.5
    return float(proba), bool(label)

def score_batch(model, df_batch):
    # ... (same as before, unchanged)
    required_raw_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", "gender", "SeniorCitizen", 
        "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    
    missing = [c for c in required_raw_cols if c not in df_batch.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df = df_batch.copy()
    df["charge_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0,6,12,24,72], labels=["new_risk","early_risk","mid_term","loyal"], right=False)
    df["high_value"] = ((df["tenure"] > 24) & (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75))).astype(int)
    
    X = df[FEATURE_CONFIG.numeric_features + FEATURE_CONFIG.categorical_features]
    proba = model.predict_proba(X)[:, 1]
    
    df_results = df.copy()
    df_results["churn_probability"] = proba
    df_results["churn_label"] = df_results["churn_probability"] >= 0.5
    df_results["risk_rank"] = df_results["churn_probability"].rank(pct=True)
    df_results["high_risk"] = df_results["risk_rank"] > 0.90
    
    return df_results.sort_values("churn_probability", ascending=False)

# ---------- STREAMLIT APP ----------

st.markdown('<h1 class="main-header">🚀 Telco Churn Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
**Production-ready churn risk scoring for telecom retention teams.**
- **Online**: Score individual customers instantly.
- **Batch**: Upload CSV for bulk risk assessment + action recommendations.
""")

mode = st.sidebar.selectbox("🎯 Scoring Mode", ["Single Customer", "Batch CSV"])
model_name = st.sidebar.selectbox("🤖 Model", list(models.keys()))
model = models[model_name]

st.sidebar.info(f"**Selected:** {model_name}")

if mode == "Single Customer":
    st.header("👤 Single Customer Prediction")
    
    with st.form("customer_form", clear_on_submit=False):
        # Auto TotalCharges calculation
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            st.subheader("💰 **Billing**")
            tenure = st.number_input("📅 Tenure (months)", min_value=0, max_value=72, value=12, key="tenure")
            monthly_charges = st.number_input("💳 Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, key="monthly")
        
        # Auto-calculate TotalCharges
        total_charges = tenure * monthly_charges
        st.info(f"**Total Charges:** ${total_charges:.2f} (auto-calculated)")
        
        with col2:
            st.subheader("👥 **Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], key="senior")
            partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
            dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")
        
        with col3:
            st.subheader("📞 **Phone Services**")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone")
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], key="multiline")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌐 **Internet Services**")
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet")
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup")
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device")
        
        with col2:
            st.subheader("📺 **Streaming**")
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="tv")
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="movies")
        
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.subheader("🛠️ **Support**")
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], key="tech")
        
        with col2:
            st.subheader("📄 **Billing**")
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless")
        
        with col3:
            st.subheader("📋 **Contract**")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ], key="payment")
        
        predict_btn = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)
    
    if predict_btn:
        input_df = build_single_customer_df(
            tenure=tenure,
            MonthlyCharges=monthly_charges,
            TotalCharges=total_charges,
            gender=gender,
            SeniorCitizen=senior_citizen,
            Partner=partner,
            Dependents=dependents,
            PhoneService=phone_service,
            MultipleLines=multiple_lines,
            InternetService=internet_service,
            OnlineBackup=online_backup,
            DeviceProtection=device_protection,
            TechSupport=tech_support,
            StreamingTV=streaming_tv,
            StreamingMovies=streaming_movies,
            Contract=contract,
            PaperlessBilling=paperless_billing,
            PaymentMethod=payment_method,
        )
        
        proba, label = score_single_customer(model, input_df)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Churn Probability", f"{proba:.1%}", delta=None)
        with col2:
            st.markdown(f"**Prediction:** {'🛑 WILL CHURN' if label else '✅ WILL STAY'}")
        
        # Business recommendation
        if proba > 0.8 and contract == "Month-to-month":
            st.markdown("""
            <div class="high-risk">
                <h3>🚨 IMMEDIATE ACTION REQUIRED</h3>
                <p><strong>Offer 1-year contract discount immediately</strong></p>
                <p>Month-to-month + high risk = highest ROI retention target</p>
            </div>
            """, unsafe_allow_html=True)
        elif proba > 0.8:
            st.markdown("""
            <div class="high-risk">
                <h3>⚠️ HIGH RISK</h3>
                <p>Engagement call: investigate billing / service issues</p>
            </div>
            """, unsafe_allow_html=True)
        elif proba > 0.5:
            st.info("📞 Medium risk → Add to monitoring queue")
        else:
            st.markdown("""
            <div class="low-risk">
                <h3>✅ LOW RISK</h3>
                <p>Continue normal service. No immediate action needed.</p>
            </div>
            """, unsafe_allow_html=True)

else:  # Batch mode
    st.header("📊 Batch Scoring - CSV Upload")
    
    st.info("""
    **Upload CSV** with these columns:
    `tenure, MonthlyCharges, TotalCharges, gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod`
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.dataframe(df_upload.head())
        
        if st.button("🚀 Score All Customers", type="primary", use_container_width=True):
            with st.spinner("Scoring customers..."):
                results = score_batch(model, df_upload)
            
            st.success("✅ Scoring complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Customers", len(results))
                st.metric("High Risk (Top 10%)", results["high_risk"].sum())
            with col2:
                top_k_precision = results.nlargest(int(0.1 * len(results)), "churn_probability")["churn_label"].mean()
                st.metric("Precision @ Top 10%", f"{top_k_precision:.1%}")
            
            st.subheader("🎯 Top 20 Highest Risk Customers")
            st.dataframe(results[['churn_probability', 'churn_label', 'risk_rank', 'high_risk', 'Contract', 'tenure', 'MonthlyCharges']].head(20))
            
            csv_out = results.to_csv(index=False)
            st.download_button(
                "💾 Download Full Results CSV",
                csv_out,
                "churn_risk_scoring_results.csv",
                "text/csv",
                use_container_width=True
            )
