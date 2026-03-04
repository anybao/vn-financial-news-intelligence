from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from src.api.schemas import ArticleRequest, SummaryResponse, SentimentResponse, NERResponse, EventPredictionResponse
from src.database import get_db
from src.models import Article
import html
import os
import re
import logging

logger = logging.getLogger(__name__)


def _clean_entities(text: str) -> str:
    """Decode broken HTML entities like #225; -> á."""
    if not text:
        return ""
    text = re.sub(r'(?<!&)#(\d+);', r'&#\1;', text)
    return html.unescape(text)


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
                    "ACB": ["ACB", "Ngân hàng Á Châu", "Asia Commercial Bank"],
                    "BCM": ["BCM", "Becamex IDC", "Tổng Công ty Becamex IDC"],
                    "BID": ["BID", "BIDV", "Ngân hàng Đầu tư và Phát triển Việt Nam"],
                    "BVH": ["BVH", "Bảo Việt", "Tập đoàn Bảo Việt"],
                    "CTG": ["CTG", "VietinBank", "Ngân hàng Công Thương Việt Nam"],
                    "FPT": ["FPT", "Công ty Cổ phần FPT", "Tập đoàn FPT"],
                    "GAS": ["GAS", "PV Gas", "Tổng Công ty Khí Việt Nam"],
                    "GVR": ["GVR", "Cao su Việt Nam", "Tập đoàn Công nghiệp Cao su Việt Nam"],
                    "HDB": ["HDB", "HDBank", "Ngân hàng Phát triển TP.HCM"],
                    "HPG": ["HPG", "Hòa Phát", "Tập đoàn Hòa Phát"],
                    "MBB": ["MBB", "MB Bank", "Ngân hàng Quân đội", "MB"],
                    "MSN": ["MSN", "Masan", "Tập đoàn Masan"],
                    "MWG": ["MWG", "Thế Giới Di Động", "Mobile World"],
                    "PLX": ["PLX", "Petrolimex", "Tập đoàn Xăng dầu Việt Nam"],
                    "POW": ["POW", "PV Power", "Tổng Công ty Điện lực Dầu khí Việt Nam"],
                    "SAB": ["SAB", "Sabeco", "Tổng Công ty Bia Rượu Nước giải khát Sài Gòn"],
                    "SHB": ["SHB", "Ngân hàng Sài Gòn - Hà Nội"],
                    "SSB": ["SSB", "SeABank", "Ngân hàng Đông Nam Á"],
                    "SSI": ["SSI", "Chứng khoán SSI", "Công ty Chứng khoán SSI"],
                    "STB": ["STB", "Sacombank", "Ngân hàng Sài Gòn Thương Tín"],
                    "TCB": ["TCB", "Techcombank", "Ngân hàng Kỹ thương Việt Nam"],
                    "TPB": ["TPB", "TPBank", "Ngân hàng Tiên Phong"],
                    "VCB": ["VCB", "Vietcombank", "Ngân hàng Ngoại thương Việt Nam"],
                    "VHM": ["VHM", "Vinhomes", "Công ty Cổ phần Vinhomes"],
                    "VIB": ["VIB", "Ngân hàng Quốc tế Việt Nam"],
                    "VIC": ["VIC", "Vingroup", "Tập đoàn Vingroup"],
                    "VJC": ["VJC", "Vietjet Air", "Vietjet", "Hãng hàng không Vietjet"],
                    "VNM": ["VNM", "Vinamilk", "Công ty Cổ phần Sữa Việt Nam"],
                    "VPB": ["VPB", "VPBank", "Ngân hàng Việt Nam Thịnh Vượng"],
                    "VRE": ["VRE", "Vincom Retail", "Công ty Cổ phần Vincom Retail"]
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

@router.post(
    "/summarize",
    response_model=SummaryResponse,
    summary="Summarize text",
    description=(
        "Generate a summary of the input Vietnamese financial news article.\n\n"
        "**Models available:**\n\n"
        "| Model | Type | Architecture | Quality | Speed |\n"
        "|-------|------|-------------|---------|-------|\n"
        "| `vit5` | Abstractive | VietAI/vit5-base-vietnews-summarization (T5) | ★★★★★ | ~2s |\n"
        "| `seq2seq` | Abstractive | BiLSTM + Bahdanau Attention + Beam Search | ★★☆☆☆ | ~0.5s |\n"
        "| `extractive` | Extractive | TF + Position + Length + Data scoring | ★★★☆☆ | <0.1s |\n\n"
        "All abstractive models automatically fall back to extractive if they fail or produce degenerate output."
    ),
    response_description="The generated summary and the model that produced it.",
    tags=["Summarization"],
)
def summarize_text(request: ArticleRequest):
    try:
        model_name = (request.model.value if request.model else "vit5")
        logger.info(f"POST /summarize | model={model_name} | input_len={len(request.text)} chars")
        summary = summarizer.summarize(request.text, model=model_name)
        logger.info(f"POST /summarize | model={model_name} | output_len={len(summary)} chars | summary_preview='{summary[:80]}...'")
        return SummaryResponse(summary=summary, model=model_name)
    except Exception as e:
        logger.error(f"POST /summarize | ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/sentiment",
    response_model=SentimentResponse,
    summary="Analyze sentiment",
    description="Classify the sentiment of a Vietnamese financial news article as Positive, Neutral, or Negative using PhoBERT.",
    tags=["NLP Analysis"],
)
def analyze_sentiment(request: ArticleRequest):
    try:
        result = sentiment_predictor.predict(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/ner",
    response_model=NERResponse,
    summary="Extract stock entities",
    description="Extract VN30 stock ticker symbols from the text using hybrid NER (rule-based dictionary + PhoBERT token classification).",
    tags=["NLP Analysis"],
)
def map_stocks(request: ArticleRequest):
    try:
        result = ner_predictor.extract_stocks(request.text)
        return NERResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/predict_event",
    response_model=EventPredictionResponse,
    summary="Full NLP pipeline",
    description=(
        "Run all three NLP tasks simultaneously on the input article:\n\n"
        "1. **Summarization** — generate a concise summary\n"
        "2. **Sentiment analysis** — classify as Positive / Neutral / Negative\n"
        "3. **NER** — extract VN30 stock tickers mentioned\n\n"
        "Returns a unified response with summary, sentiment, stocks, and duplicate flag."
    ),
    tags=["NLP Analysis"],
)
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
    """Fetch the latest 50 NLP-processed articles with on-the-fly NLP fallback."""
    articles = db.query(Article).order_by(Article.created_at.desc()).limit(50).all()
    results = []
    for a in articles:
        stocks = [s for s in a.stocks.split(",") if s] if a.stocks else []
        nlp_summary = a.nlp_summary if a.nlp_summary else None

        # On-the-fly fallback: use raw_summary for NLP (not title, to avoid title leaking into summary)
        text_for_nlp = a.raw_summary or a.title or ''
        text_for_ner = f"{a.title}. {a.raw_summary or ''}"
        if not stocks:
            try:
                stocks_res = ner_predictor.extract_stocks(text_for_ner)
                stocks = stocks_res.get("stocks", [])
                if stocks:
                    a.stocks = ",".join(stocks)
                    db.commit()
            except Exception:
                pass

        # On-the-fly fallback: generate summary if missing
        if not nlp_summary:
            try:
                nlp_summary = summarizer.summarize(text_for_nlp)
                if nlp_summary:
                    a.nlp_summary = nlp_summary
                    db.commit()
            except Exception:
                pass

        # Final fallback: use raw_summary if NLP summary still empty
        display_summary = _clean_entities(nlp_summary or a.raw_summary or "")

        # On-the-fly fallback: run sentiment if still default
        sentiment = a.sentiment
        if not sentiment or sentiment == "Neutral":
            try:
                sent_res = sentiment_predictor.predict(text_for_nlp)
                sentiment = sent_res.get("sentiment", "Neutral")
                if sentiment != a.sentiment:
                    a.sentiment = sentiment
                    db.commit()
            except Exception:
                pass

        results.append({
            "id": a.id,
            "title": a.title,
            "link": a.link,
            "published": a.published,
            "source": a.source,
            "raw_summary": _clean_entities(a.raw_summary or ""),
            "nlp_summary": display_summary,
            "sentiment": sentiment,
            "stocks": stocks
        })
    return results

@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the real-time Tailwind dashboard."""
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Dashboard UI not found.</h1>"

