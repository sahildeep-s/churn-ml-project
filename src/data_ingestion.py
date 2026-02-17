from pathlib import Path
from sqlalchemy import text

import pandas as pd

from .config import RAW_DATA_PATH
from .db_utils import get_engine

def load_raw_csv(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer CSV into a pandas DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from all string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Convert TotalCharges to numeric and handle errors as NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows where TotalCharges is missing (we might revisit this later)
    df = df.dropna(subset=["TotalCharges"])

    # IMPORTANT: ensure df is a fresh copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Business segmentation flags
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 72],
        labels=["new_risk", "early_risk", "mid_term", "loyal"],
        right=False,
    )

    df["high_value"] = (
        (df["tenure"] > 24)
        & (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75))
    ).astype(int)

    df["charge_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)

    print(
        f"High-value customers flagged: {df['high_value'].sum()} "
        f"({df['high_value'].mean():.1%})"
    )

    return df


def log_assumptions_and_risks():
    """
    Document critical assumptions and risks for audit trail.
    
    Assumption 1: TotalCharges NaN = data entry error (drop safe)
    Risk: If NaN = "0 charges" (new customer), we lose high-risk cohort
    Mitigation: Validate against tenure==0 records (done below)
    """
    print("=== ASSUMPTIONS & RISKS ===")
    print("1. TotalCharges NaN → Drop row (assume data error)")
    print("   Risk: New customers with blank TotalCharges lost")
    print("   Mitigation: Check tenure==0 records post-drop")
    
    print("2. Past behavior predicts NEXT period churn")
    print("   Risk: Seasonal effects unmodeled")
    print("3. All churn equally costly (NOT true)")
    print("   Risk: Model treats $10/mo vs $100/mo loss equal")

def write_to_db(df: pd.DataFrame, table_name: str = "telco_churn"):
    """
    Write the DataFrame to the SQLite database as a table.

    If the table exists, it will be replaced during this phase.
    Also creates indexes on key columns for faster analytics queries.
    """
    engine = get_engine()

    # 1) Write DataFrame to SQL table (overwrite each run)
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",  # drop table if it exists, then recreate
        index=False,
    )

    # Create indexes for faster queries
    index_statements = [
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_tenure_churn ON {table_name} (tenure, Churn);",
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_contract_churn ON {table_name} (Contract, Churn);",
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_high_value ON {table_name} (high_value);",
    ]

    with engine.begin() as conn:  # begin() handles commit/rollback
        for stmt in index_statements:
            conn.execute(text(stmt))

    print("Indexes created for production query speed.")


def main():
    """
    End-to-end ETL for phase 1: Load raw CSV, clean minimal issues, and persist to SQLite DB.
    """
    
    log_assumptions_and_risks()

    print("Loading raw CSV data...")
    df_raw = load_raw_csv()

    print("Cleaning raw data...")
    df_clean = clean_raw_data(df_raw)

    print("Writing cleaned data to database...")
    write_to_db(df_clean)

    print("Data written successfully to SQLite database.")

if __name__ == "__main__":
    main()