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
- Python 3.10+
- Docker and Docker Compose

### Application Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start MLOps Services (MLFlow, Prometheus, Grafana)**
   ```bash
   docker-compose up -d
   ```

3. **Start the FastAPI Server**
   ```bash
   uvicorn src.api.main:app --reload
   ```

### Running Tests
```bash
pytest tests/
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
