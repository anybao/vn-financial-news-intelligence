from fastapi import APIRouter, HTTPException
from src.api.schemas import ArticleRequest, SummaryResponse, SentimentResponse, NERResponse, EventPredictionResponse
from src.summarization.infer import SummarizerInference
from src.sentiment.predict_sentiment import SentimentPredictor
from src.ner.predict_ner import HybridNERPredictor

# Initialize Router
router = APIRouter()

# Global instances (in production, use dependency injection or app state)
summarizer = SummarizerInference()
sentiment_predictor = SentimentPredictor()
ner_predictor = HybridNERPredictor()

@router.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: ArticleRequest):
    try:
        summary = summarizer.summarize(request.text)
        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: ArticleRequest):
    try:
        result = sentiment_predictor.predict(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ner", response_model=NERResponse)
async def map_stocks(request: ArticleRequest):
    try:
        result = ner_predictor.extract_stocks(request.text)
        return NERResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict_event", response_model=EventPredictionResponse)
async def predict_financial_event(request: ArticleRequest):
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
