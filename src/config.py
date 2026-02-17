from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "Telco-Customer-Churn.csv"

# Database path
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "churn.db"
DB_URL = f"sqlite:///{DB_PATH}"