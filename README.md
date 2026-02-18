# 📊 Telco Customer Churn Prediction System

Production ML system delivering **actionable churn risk scores** for telecom retention teams.

Designed around **business ROI rather than model accuracy**, with SHAP explainability, automated testing, and containerized deployment.

---

# 🚨 Business Problem

Telecom companies lose significant recurring revenue when high-value customers churn.

- 27% annual attrition
- $1.62M revenue loss per 10K customers at $50 MRR
- Generic retention campaigns waste budget targeting low-risk accounts

## 🎯 Solution

This system identifies the **riskiest 10% of customers with 75% precision**, allowing retention teams to:

- Focus on high-risk accounts only
- Protect approximately $450K ARR
- Minimize campaign spend

---

# 📈 Key Results

| Metric | Logistic Regression | Random Forest | Business Relevance |
|--------|--------------------|---------------|--------------------|
| **ROC AUC** | **0.838** | 0.834 | Strong discrimination across thresholds |
| **Recall (Churn)** | **0.802** | 0.735 | Captures 80% of true churners |
| **Precision @ Top 10%** | **0.750** | 0.736 | 75% true positives in top decile |

## 🔎 Primary Churn Drivers (SHAP)

- Month-to-month contract
- Short tenure
- Electronic check payment

---

# 💰 Business Impact Analysis

**Total customers:** 10,000  
**Annual churn rate:** 27% (2,700 customers)  
**Average MRR:** $50/month ($600/year)

## 🎯 Model Intervention (Top 10%)

- Targeted customers: 1,000
- Expected true churners captured: 750
- Revenue protected: **$450,000 ARR**

## 💸 Cost & ROI

- Cost per contact: ~$10
- Total campaign cost: $10,000
- **Net ROI: ~$440,000 ARR**

---

# 🖥️ Application Features

## 🔹 Online Scoring

Real-time single-customer risk assessment with:

- Auto-calculated `TotalCharges`
- Structured and categorized input form
- SHAP waterfall explanation per customer
- Segment-specific action recommendation

## 🔹 Batch Scoring

Bulk CSV assessment with:

- Risk ranking across all uploaded customers
- Live Precision@Top 10% metric
- Downloadable scored output with action flags

---

# 🎯 Action Framework

| Risk Level | Condition | Recommended Action |
|------------|-----------|-------------------|
| **IMMEDIATE** | High Risk + Month-to-Month (P > 80%) | Retention discount offer |
| **HIGH** | High Risk (P > 80%) | Proactive engagement call |
| **MONITOR** | Medium Risk (P > 50%) | Light outreach |
| **SAFE** | Low Risk (P < 50%) | No action required |

---

# 📊 Demo

## Single Customer Scoring
![Single Customer Scoring](reports/single_customer.png)

## Batch Risk Scoring
![Batch Scoring Results](reports/batch_results.png)

## SHAP Feature Importance
![SHAP Beeswarm](reports/beeswarm_churn.png)

---

# 🚀 Quick Start

## 🖥️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/churn-ml-project.git
cd churn-ml-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place Telco-Customer-Churn.csv inside data/

# 4. Run the pipeline
python src/data_ingestion.py
python src/train_model.py
python src/explain_model.py

# 5. Launch the app
streamlit run app.py
```

---

## 🐳 Docker Setup

```bash
# Build the image
docker build -t churn-predictor .

# Run the container
docker run -p 8501:8501 churn-predictor

# Open in browser
http://localhost:8501
```

---

# 🏗️ Architecture

Raw CSV  
↓  
ETL Pipeline (SQLite)  
↓  
Feature Engineering  
↓  
Dual Model Training (Logistic Regression + Random Forest)  
↓  
SHAP Explainability  
↓  
Streamlit Application (Online + Batch Scoring)  
↓  
Docker Container Deployment  

---

# ⚙️ Technical Implementation

## 🧠 Feature Engineering

Business-driven transformations grounded in telecom domain knowledge:

| Feature | Logic | Business Rationale |
|----------|--------|-------------------|
| `tenure_bucket` | 0–6, 6–12, 12–24, 24+ months | 80% churn occurs in first 12 months |
| `high_value_flag` | Top 25% charges AND tenure > 24 months | VIP segment requiring priority retention |
| `charge_ratio` | TotalCharges / (tenure + 1) | Detects price sensitivity and billing shock |

## 📊 Model Selection Rationale

Logistic Regression selected as production model because:

- Higher Precision@K and Recall
- Strong interpretability for stakeholders
- Greater stability under distribution shift

Random Forest retained as challenger model for validation.

---

## 🔍 Explainability

SHAP provides:

- Global feature importance (beeswarm plots)
- Local per-customer waterfall explanations
- Audit-ready exportable rationale

---

## 🧪 Automated Testing

Scenario-based ranking tests validate model consistency:

```bash
python tests/test_scenarios.py
```

Tests assert:

- High-risk profiles rank above medium-risk profiles
- Medium-risk profiles rank above low-risk profiles
- Results consistent across both models

---

# 📂 Repository Structure

```
churn-ml-project/
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── src/
├── tests/
├── notebooks/
├── models/
├── data/
└── reports/
```

---

# 🏆 Engineering Standards

| Area | Approach |
|------|----------|
| Architecture | Modular `src/` package |
| Data Layer | SQL-first ETL with indexed SQLite |
| Evaluation | Precision@K (business-first metric) |
| Explainability | SHAP global + local explanations |
| Deployment | Docker containerized |
| Testing | Automated scenario ranking validation |

---

# 🛠️ Technology Stack

| Layer | Tools |
|--------|--------|
| ML | scikit-learn, SHAP |
| Data | Pandas, SQLAlchemy, SQLite |
| Deployment | Streamlit, Docker |
| Analysis | Jupyter |
| Language | Python 3.11 |

---

# 🔮 Production Extensions Planned

- FastAPI scoring endpoint for CRM integration
- Scheduled batch retraining pipeline (Airflow / cron)
- Model drift monitoring dashboard
- A/B testing framework for retention interventions
- PostgreSQL migration for multi-tenant scale

---

# 📊 Dataset

IBM Telco Customer Churn dataset (Kaggle):

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the downloaded CSV at:

data/Telco-Customer-Churn.csv

---

# 📅 Built

February 2026  
Designed for production telecom retention workflows.
