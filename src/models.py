from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime
from src.database import Base
import datetime

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    link = Column(String, unique=True, index=True) # Unique to prevent identical links
    published = Column(String)
    source = Column(String)
    
    # Raw parsed text
    raw_summary = Column(Text)
    
    # NLP Pipeline Outputs
    nlp_summary = Column(Text)
    sentiment = Column(String)
    stocks = Column(String) # Stored as comma-separated tags
    is_duplicate = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
