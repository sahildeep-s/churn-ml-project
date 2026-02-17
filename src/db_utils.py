from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from .config import DB_URL

def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine for the churn database."""
    engine = create_engine(DB_URL, echo = False, future = True)
    return engine