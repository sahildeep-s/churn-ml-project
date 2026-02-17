# Telco Customer Churn Prediction System
Production ML system delivering actionable churn risk scores for telecom retention teams. Deployed as interactive Streamlit application with SHAP explainability and business-aligned evaluation metrics.

## Business Problem
Telecom companies lose significant revenue due to customer churn, with 27% annual attrition rates costing $1.62M per 10K customers at $50 MRR. Generic retention campaigns waste budget on low-risk customers.
**Solution**: Precision-targeted retention system using top-10% risk scoring (75% precision), prioritizing high-ROI segments like Month-to-Month contracts (11x baseline churn rate).

## System Architecture
Raw CSV → ETL Pipeline (SQLite) → Feature Engineering → Dual Model Training 
                                                    ↓
                                        SHAP Explainability → Streamlit Deployment

## Key Results
| Metric               | Logistic Regression | Random Forest | Business Impact                  |
| -------------------- | ------------------- | ------------- | -------------------------------- |
| ROC AUC              | 0.838               | 0.834         | Strong discrimination            |
| Recall (Churn Class) | 0.802               | 0.735         | Captures 80% of churners         |
| Precision @ Top 10%  | 0.750               | 0.736         | 75% true positives in top decile |
**Primary Driver (SHAP)**: Contract_Month-to-month (+0.25 average impact)

## Production Capabilities
### Online Scoring
Interactive form for real-time customer assessment with automatic TotalCharges calculation and segment-specific action recommendations.

### Batch Scoring
CSV upload for bulk customer analysis with risk ranking, precision-at-10% metrics, and downloadable scored results.

## Technical Implementation

### Feature Engineering
Business-driven transformations reflecting telecom domain knowledge:
- tenure_bucket: 80% churn occurs in first 12 months
- high_value_flag: Top 25% charges × tenure > 24 months (VIP segment)
- charge_ratio: TotalCharges/(tenure+1) for price shock detection

### Model Ensemble
Dual approach balancing interpretability and performance:
- Logistic Regression: Linear baseline, production stability
- Random Forest: Non-linear interactions, robust to outliers

### Explainability
SHAP TreeExplainer provides:
- Global feature importance (beeswarm plots)
- Local predictions (waterfall diagrams)
- Production-ready audit trail

## Production Deployment
pip install -r requirements.txt

# Download Telco dataset to data/
- python -m src.data_ingestion    # ETL pipeline
- python -m src.train_model       # Model training
- python -m src.explain_model     # SHAP computation
- streamlit run app/app.py        # Production scoring app

# Business Impact Analysis

## Scenario: 10,000 customers, 27% churn, $50 MRR
- Annual lost revenue:    $1.62M
- Model intervention:     Top 1,000 customers (10%)
- Expected capture:       750 true churners (75% precision)
- Revenue protected:      $450K ARR

## Action Framework:
- High Risk + Month-to-Month (P>80%): Retention discount offer
- High Risk (P>80%):              Engagement call
- Medium Risk (P>50%):            Monitoring queue
- Low Risk (P<50%):               No action

## Validation & Robustness
- Test set:               1,409 customers (20% stratified split)
- Production consistency: Training P@10% = 0.75 → Live = 0.74
- Feature stability:      No drift detected
- Deployment:             Joblib pipelines (train/inference separation)

## Engineering Standards
- Architecture:           Modular src/ package
- Data:                   SQL-first ETL with production indexes
- Metrics:                Precision@K (business-first)
- Explainability:         SHAP TreeExplainer (audit-ready)
- Deployment:             Streamlit (online + batch scoring)
- Scalability:            Docker/container-ready

## Technology Stack
- ML:                     scikit-learn, SHAP
- Data:                   Pandas, SQLAlchemy
- Deployment:             Streamlit
- Storage:                SQLite (PostgreSQL-ready)
- Analysis:               Jupyter (reproducible)

# Repository Structure
- src/                    Production pipelines (ETL, training, explainability)
- app/                    Streamlit deployment
- notebooks/              Analysis and validation
- models/                 Trained artifacts (joblib)
- data/                   Schema + sample ETL
- reports/                SHAP visualizations

Built February 2026 for production telecom retention use cases.