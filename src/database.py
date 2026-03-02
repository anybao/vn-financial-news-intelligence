from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Define the local SQLite database path
DATABASE_URL = "sqlite:///./database.db"

# Create Database Engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Setup SessionLocal class for local sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for the SQLAlchemy Models
from sqlalchemy.orm import declarative_base
Base = declarative_base()

# Initialize tables
def init_db():
    from src.models import Article # Import models here to avoid circular imports
    Base.metadata.create_all(bind=engine)

init_db()

def get_db():
    """Dependency pattern to get DB session and automatically close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
