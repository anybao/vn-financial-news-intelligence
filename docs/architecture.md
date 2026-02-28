# Architecture Documentation

## 1. High-Level Architecture Overview
The system is built as a modular pipeline to process financial news articles, extract structured sentiments and entities, generate abstractive summaries, and serve these models through an MLOps-ready RESTful API.

### Core Pipeline Flow:
1. **Ingestion:** RSS feeds are fetched periodically via `apscheduler`.
2. **Preprocessing:** Text is cleaned (URL removal, character normalization) and tokenized.
3. **Deduplication:** A fast embedding hash (using PhoBERT) is used to calculate cosine similarity to existing articles in the database/event store. If similarity > 0.85, the article is deduplicated.
4. **NLP Processing (Parallelized Tasks):**
    - **Summarization:** A seq2seq PyTorch model encodes the text and decodes it using a customized Beam Search tree and Bahdanau Attention.
    - **Sentiment:** A text-classification pipeline determines Positive, Negative, or Neutral sentiment.
    - **NER:** A hybrid dictionary/predictive NER model extracts company tickers.
5. **Serving:** FastAPI acts as the primary service interface container.
6. **MLOps Tracking:** MLflow tracks training hyperparams; Prometheus collects API latencies.

## 2. Infrastructure Diagram (Text Outline)

`News Sources -> Scheduled Ingestion -> Deduplication Module -> NLP Core -> API Database -> FastAPI Serves Dashboard`

## 3. Technology Stack List
- **Language:** Python 3.10
- **Deep Learning Framework:** PyTorch & HuggingFace Transformers
- **Web Framework:** FastAPI, Uvicorn
- **MLOps:** MLflow, Prometheus, Grafana, GitHub Actions
- **Evaluation:** Rouge (via HF Evaluate)
