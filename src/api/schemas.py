from pydantic import BaseModel, Field
from typing import List

class ArticleRequest(BaseModel):
    text: str = Field(..., title="The text of the financial news article to process.")

class SummaryResponse(BaseModel):
    summary: str
    
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    
class NERResponse(BaseModel):
    stocks: List[str]

class EventPredictionResponse(BaseModel):
    summary: str
    sentiment: str
    confidence: float
    stocks: List[str]
    is_duplicate: bool = False
