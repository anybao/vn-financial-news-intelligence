# Financial News Intelligence (NLP501 Final Project)

This project is an End-to-End NLP System for Financial News, developed for the NLP501 Final Project.
It features Abstractive Summarization, Sentiment Analysis, Stock Mapping (NER), and Semantic Deduplication.
The backend API is served by FastAPI and heavily monitored utilizing a full MLOps stack.

## Architecture Highlights
- **Summarization:** BiLSTM Encoder + LSTM Decoder with Bahdanau Attention. Decoding via Beam Search.
- **Sentiment:** PhoBERT sequence classification fine-tuning.
- **NER:** Hybrid approach (Rule-based dictionary + PhoBERT token classification).
- **MLOps Stack:** MLflow (Experiment tracking), Prometheus (Metrics), Grafana (Dashboard), Evidently (Drift monitoring).

## Getting Started

### Prerequisites
- Python 3.10 to 3.13 (Python 3.14 unsupported for some Rust bindings)
- Docker and Docker Compose (Optional, for MLOps stack)

### 🚀 Running the System

The easiest way to spin up the entire system (MLOps containers, Python Virtual Environment, Data Ingestion Scheduler, and FastAPI Server) is using the provided bash script:

```bash
chmod +x start_all.sh
./start_all.sh
```

### 💡 Using the API

Once the system is running (available on `localhost:8000`), you can interact with it in several ways:

#### 1. Live NLP Dashboard
Navigate to your auto-updating AI Dashboard: **[http://localhost:8000/api/v1/dashboard](http://localhost:8000/api/v1/dashboard)** (The home directory `/` automatically redirects here!)

#### 2. Interactive API Docs (Swagger UI)
Open your browser and navigate to: **[http://localhost:8000/docs](http://localhost:8000/docs)**.
You can click on the `POST /api/v1/predict_event` endpoint, click **"Try it out"**, enter a test sentence, and hit Execute.

#### 2. Test via Terminal (cURL)
You can send a test financial article directly from your terminal to see the NLP pipelines in action (Summarization + Sentiment + NER all at once):

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict_event' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Cổ phiếu FPT tăng mạnh sau báo cáo tài chính tích cực trong quý 3."
}'
```

#### 3. View MLOps Metrics
- **API Performance:** You can view real-time request counts and latencies at **[http://localhost:8000/metrics](http://localhost:8000/metrics)**.
- **MLflow Tracking:** Access the MLflow dashboard for training runs at **[http://localhost:5001](http://localhost:5001)**.

### 🌐 Accessible Endpoints

| Endpoint | URL | Description |
|---|---|---|
| **Dashboard** | [http://localhost:8000/api/v1/dashboard](http://localhost:8000/api/v1/dashboard) | Live auto-updating NLP dashboard |
| **Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive API documentation |
| **Summarize** | `POST http://localhost:8000/api/v1/summarize` | Abstractive summarization |
| **Sentiment** | `POST http://localhost:8000/api/v1/sentiment` | Sentiment classification |
| **NER** | `POST http://localhost:8000/api/v1/ner` | Stock ticker extraction |
| **Predict Event** | `POST http://localhost:8000/api/v1/predict_event` | Combined pipeline (Summary + Sentiment + NER) |
| **Articles** | `GET http://localhost:8000/api/v1/articles` | Retrieve processed articles from database |
| **Health** | `GET http://localhost:8000/api/v1/health` | Health check |
| **Model Info** | `GET http://localhost:8000/api/v1/model_info` | Model source info (MLflow Registry vs Local) |
| **Prometheus Metrics** | [http://localhost:8000/metrics](http://localhost:8000/metrics) | API request counts & latencies |
| **MLflow** | [http://localhost:5001](http://localhost:5001) | Experiment tracking & model registry |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Metrics collection |
| **Grafana** | [http://localhost:3000](http://localhost:3000) | Monitoring dashboards (admin/admin) |

### 🧠 Training the Models

To train the models on your own datasets, replace `data/processed/sentiment.csv` and `data/processed/ner.json` with your annotated datasets, then run the training scripts:

```bash
source venv/bin/activate
python src/sentiment/train_sentiment.py
python src/ner/train_ner.py
```

## Directory Structure
- `src/`: Source code for the application split into domain areas (NLP, ingestion, API).
- `mlops/`: Utility scripts for MLflow tracking, Prometheus metrics, etc.
- `notebooks/`: Jupyter notebooks for EDA and baseline experimentation.
- `docs/`: Project documentation and reports.
- `tests/`: Pytest unit tests.
- `data/`: Raw and processed dataset storage.
- `models/`: Weights and tokenizers for the trained models.
- `configs/`: YAML Configuration files.
