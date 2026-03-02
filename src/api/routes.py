from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from src.api.schemas import ArticleRequest, SummaryResponse, SentimentResponse, NERResponse, EventPredictionResponse
from src.summarization.infer import SummarizerInference
from src.sentiment.predict_sentiment import SentimentPredictor
from src.ner.predict_ner import HybridNERPredictor
from src.database import get_db
from src.models import Article
import os

# Initialize Router
router = APIRouter()

# Global instances (in production, use dependency injection or app state)
summarizer = SummarizerInference()
sentiment_predictor = SentimentPredictor()
ner_predictor = HybridNERPredictor()

@router.post("/summarize", response_model=SummaryResponse)
def summarize_text(request: ArticleRequest):
    try:
        summary = summarizer.summarize(request.text)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: ArticleRequest):
    try:
        result = sentiment_predictor.predict(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ner", response_model=NERResponse)
def map_stocks(request: ArticleRequest):
    try:
        result = ner_predictor.extract_stocks(request.text)
        return NERResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_event", response_model=EventPredictionResponse)
def predict_financial_event(request: ArticleRequest):
    """Integrates Summarization, Sentiment, and NER simultaneously."""
    try:
        summary = summarizer.summarize(request.text)
        sentiment_res = sentiment_predictor.predict(request.text)
        stocks_res = ner_predictor.extract_stocks(request.text)
        
        # Deduplication check can be added here
        
        return EventPredictionResponse(
            summary=summary,
            sentiment=sentiment_res["sentiment"],
            confidence=sentiment_res["confidence"],
            stocks=stocks_res["stocks"],
            is_duplicate=False # Placeholder
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/articles")
async def get_articles(db: Session = Depends(get_db)):
    """Fetch the latest 50 NLP-processed articles."""
    articles = db.query(Article).order_by(Article.created_at.desc()).limit(50).all()
    return [{
        "id": a.id,
        "title": a.title,
        "link": a.link,
        "published": a.published,
        "source": a.source,
        "nlp_summary": a.nlp_summary,
        "sentiment": a.sentiment,
        "stocks": [s for s in a.stocks.split(",") if s] if a.stocks else []
    } for a in articles]

@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the real-time Tailwind dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Dashboard UI not found.</h1>"
