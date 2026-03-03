from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from src.api.schemas import ArticleRequest, SummaryResponse, SentimentResponse, NERResponse, EventPredictionResponse
from src.database import get_db
from src.models import Article
import os
import logging

logger = logging.getLogger(__name__)

# Initialize Router
router = APIRouter()

# --- Model loading with MLflow Registry fallback to local ---

_model_sources = {}

def _load_sentiment_predictor():
    """Try MLflow Registry first, then fall back to local."""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
        model_uri = "models:/SentimentModel/Production"
        pipe = mlflow.transformers.load_model(model_uri)
        logger.info("Loaded SentimentModel from MLflow Registry (Production)")
        _model_sources["sentiment"] = "MLflow Registry (Production)"
        
        class MLflowSentimentPredictor:
            def __init__(self, pipeline):
                self._pipe = pipeline
                self.labels = ["Negative", "Neutral", "Positive"]
            def predict(self, text: str) -> dict:
                result = self._pipe(text)
                if isinstance(result, list):
                    result = result[0]
                label = result.get("label", "LABEL_0")
                score = result.get("score", 0.0)
                # Map LABEL_X format to human-readable
                if label.startswith("LABEL_"):
                    idx = int(label.split("_")[1])
                    label = self.labels[idx] if idx < len(self.labels) else label
                return {"sentiment": label, "confidence": score}
        
        return MLflowSentimentPredictor(pipe)
    except Exception as e:
        logger.warning(f"Could not load SentimentModel from MLflow: {e}. Falling back to local.")
        from src.sentiment.predict_sentiment import SentimentPredictor
        _model_sources["sentiment"] = "Local (models/sentiment)"
        return SentimentPredictor()

def _load_ner_predictor():
    """Try MLflow Registry first, then fall back to local."""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
        model_uri = "models:/NERModel/Production"
        pipe = mlflow.transformers.load_model(model_uri)
        logger.info("Loaded NERModel from MLflow Registry (Production)")
        _model_sources["ner"] = "MLflow Registry (Production)"
        
        class MLflowNERPredictor:
            def __init__(self, pipeline):
                self._pipe = pipeline
                # Keep rule-based dictionary for hybrid approach
                from src.ner.predict_ner import HybridNERPredictor
                self._rule_based = HybridNERPredictor.__new__(HybridNERPredictor)
                self._rule_based.stock_dictionary = {
                    "FPT": ["FPT", "Công ty Cổ phần FPT", "Tập đoàn FPT"],
                    "VCB": ["VCB", "Vietcombank", "Ngân hàng Ngoại thương Việt Nam"],
                    "VNM": ["VNM", "Vinamilk", "Công ty Cổ phần Sữa Việt Nam"],
                    "VIC": ["VIC", "Vingroup", "Tập đoàn Vingroup"],
                    "HPG": ["HPG", "Hòa Phát", "Tập đoàn Hòa Phát"],
                    "TCB": ["TCB", "Techcombank", "Ngân hàng Kỹ thương Việt Nam"]
                }
            def extract_stocks(self, text: str) -> dict:
                rule_stocks = self._rule_based.predict_rule_based(text)
                return {"stocks": list(set(rule_stocks))}
        
        return MLflowNERPredictor(pipe)
    except Exception as e:
        logger.warning(f"Could not load NERModel from MLflow: {e}. Falling back to local.")
        from src.ner.predict_ner import HybridNERPredictor
        _model_sources["ner"] = "Local (models/ner)"
        return HybridNERPredictor()

def _load_summarizer():
    """Always use local summarizer (custom Seq2Seq architecture)."""
    from src.summarization.infer import SummarizerInference
    _model_sources["summarization"] = "Local (models/summarizer)"
    return SummarizerInference()


# Initialize global model instances
summarizer = _load_summarizer()
sentiment_predictor = _load_sentiment_predictor()
ner_predictor = _load_ner_predictor()

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
        
        return EventPredictionResponse(
            summary=summary,
            sentiment=sentiment_res["sentiment"],
            confidence=sentiment_res["confidence"],
            stocks=stocks_res["stocks"],
            is_duplicate=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/model_info")
async def model_info():
    """Returns which model source (MLflow Registry or Local) is being used for each model."""
    return {
        "models": _model_sources,
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    }

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

