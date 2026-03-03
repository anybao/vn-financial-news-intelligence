# Architecture Documentation

## 1. High-Level Architecture Overview

The system is built as a modular pipeline to process Vietnamese financial news articles end-to-end — from RSS ingestion through NLP processing (summarization, sentiment analysis, NER), deduplication, and serving via a real-time dashboard with full MLOps monitoring.

### Core Pipeline Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   RSS Sources   │───▶│   Ingestion &    │───▶│  Text Preprocessing │
│ VnExpress       │    │   Scheduling     │    │  HTML cleaning      │
│ VnEconomy       │    │   (APScheduler)  │    │  Entity decoding    │
└─────────────────┘    └──────────────────┘    └─────────┬───────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  SQLite + ORM   │◀───│  API Serving     │◀───│   NLP Processing    │
│  (database.db)  │    │  (FastAPI)       │    │  ┌───────────────┐  │
└────────┬────────┘    └────────┬─────────┘    │  │ Summarization │  │
         │                      │               │  │ Sentiment     │  │
         ▼                      ▼               │  │ NER           │  │
┌─────────────────┐    ┌──────────────────┐    │  │ Deduplication │  │
│  Dashboard UI   │    │  MLOps Stack     │    │  └───────────────┘  │
│  (Tailwind)     │    │  MLflow          │    └─────────────────────┘
│  Auto-polling   │    │  Prometheus      │
│  Modal viewer   │    │  Grafana         │
└─────────────────┘    └──────────────────┘
```

### Processing Stages

1. **Ingestion:** RSS feeds are fetched via `feedparser` on a 3-minute schedule using `APScheduler`. Articles are parsed for title, link, published date, and raw summary.
2. **Preprocessing:** HTML tags are stripped (BeautifulSoup), broken HTML entities are decoded (regex + `html.unescape`), and text is cleaned.
3. **Deduplication:** PhoBERT CLS embeddings + cosine similarity (threshold 0.85) detect near-duplicate articles.
4. **NLP Processing** (via `/api/v1/predict_event`):
   - **Summarization:** Seq2Seq BiLSTM with Bahdanau Attention + Beam Search. Falls back to extractive summarization on degenerate outputs.
   - **Sentiment:** PhoBERT fine-tuned for 3-class classification (Positive / Neutral / Negative).
   - **NER:** Hybrid approach — rule-based VN30 dictionary (6 tickers with aliases) + PhoBERT token classification.
5. **Serving:** FastAPI serves the dashboard, API endpoints, and on-the-fly NLP fallback for articles missing NLP data.
6. **Monitoring:** Prometheus middleware collects request counts and latency histograms; Grafana visualizes them.

---

## 2. Component Architecture

### 2.1 Data Ingestion (`src/ingestion/`)

| Component | File | Description |
|---|---|---|
| RSS Scraper | `rss_scraper.py` | Fetches and parses RSS feeds with `feedparser`. Includes `decode_html_entities()` to fix broken entities (e.g., `#225;` → `á`). |
| Scheduler | `scheduler.py` | `APScheduler` BackgroundScheduler runs ingestion every 3 minutes. `save_articles()` sends each article to `/api/v1/predict_event` for NLP processing and persists to SQLite. |

**RSS Feeds:**
- `https://vnexpress.net/rss/kinh-doanh.rss`
- `https://vneconomy.vn/chung-khoan.rss`

**Deduplication by link:** Articles with duplicate `link` values are skipped at the DB level (unique constraint).

### 2.2 NLP Models (`src/summarization/`, `src/sentiment/`, `src/ner/`)

#### Summarization — Seq2Seq BiLSTM

```
Input Text → PhoBERT Tokenizer → Encoder (BiLSTM) → Attention (Bahdanau) → Decoder (LSTM) → Beam Search → Summary
                                                                                                  ▼
                                                                              Degenerate? → Extractive Fallback
```

| Parameter | Value |
|---|---|
| Vocabulary size | 64,000 (PhoBERT) |
| Embedding dim | 256 |
| Hidden size | 512 |
| Layers | 1 |
| Beam width | 3 |
| Length penalty α | 0.7 |
| Min/Max output | 10 / 64 tokens |

**Degenerate detection:** Outputs are checked for known collapse patterns (repeated tokens, "generated summary based on", length < 5 chars). On detection, falls back to `_extractive_summarize()` which scores sentences by position (60%) + length (40%) and returns top-3.

#### Sentiment — PhoBERT Fine-tuning

| Parameter | Value |
|---|---|
| Base model | `vinai/phobert-base` (RoBERTa, 768-dim) |
| Classification head | Linear (768 → 3) on CLS token |
| Labels | Negative, Neutral, Positive |
| Max sequence length | 256 tokens |
| Learning rates (sweep) | {2e-5, 5e-5} |

#### NER — Hybrid Dictionary + Neural

| Component | Coverage |
|---|---|
| Rule-based dictionary | 6 VN30 tickers: FPT, VCB, VNM, VIC, HPG, TCB with Vietnamese aliases |
| PhoBERT token classification | 2 labels: O, TICKER |
| Matching | Word-boundary regex (`\bALIAS\b`), case-insensitive |
| Merge strategy | Union of rule-based + neural results, deduplicated |

### 2.3 Semantic Deduplication (`src/deduplication/`)

| Component | File | Description |
|---|---|---|
| Embedder | `embedder.py` | Generates 768-dim vectors via PhoBERT CLS token (or mean pooling fallback) |
| Deduplicator | `similarity.py` | Cosine similarity with threshold 0.85 |

### 2.4 API Layer (`src/api/`)

| File | Purpose |
|---|---|
| `main.py` | FastAPI app factory — mounts router, Prometheus middleware, `/metrics` endpoint, root redirect |
| `routes.py` | 8 endpoints including on-the-fly NLP fallback logic for `/articles` |
| `schemas.py` | Pydantic request/response models |
| `dashboard.html` | Tailwind CSS single-column timeline with modal article viewer |

**Model Loading Strategy:**
1. Try **MLflow Model Registry** (Production stage) for sentiment and NER models
2. Fall back to **local model weights** in `models/` directory
3. Summarizer always loads from local (custom Seq2Seq architecture)
4. Model sources are exposed via `GET /api/v1/model_info`

**On-the-fly NLP Fallback** (in `GET /articles`):
- If `stocks` is empty → runs NER on `title + raw_summary`, persists result
- If `nlp_summary` is empty → runs summarizer on `raw_summary` only (avoids title leaking), persists result
- If `sentiment` is default → re-runs sentiment predictor, persists result
- All outputs pass through `_clean_entities()` to fix broken HTML entities

### 2.5 Database (`src/database.py`, `src/models.py`)

**Engine:** SQLite at `sqlite:///./database.db` with `check_same_thread=False` for async compatibility.

**Article Model:**

| Field | Type | Notes |
|---|---|---|
| `id` | Integer | Primary key |
| `title` | String | Indexed |
| `link` | String | Unique, indexed (dedup key) |
| `published` | String | Publication date from RSS |
| `source` | String | RSS source URL |
| `raw_summary` | Text | Original RSS content |
| `nlp_summary` | Text | AI-generated summary |
| `sentiment` | String | Positive / Neutral / Negative |
| `stocks` | String | Comma-separated tickers |
| `is_duplicate` | Boolean | Semantic deduplication flag |
| `created_at` | DateTime | Auto-set to UTC now |

### 2.6 Dashboard (`src/api/dashboard.html`)

- **Framework:** Tailwind CSS (CDN), vanilla JavaScript
- **Layout:** Single-column timeline with date group headers
- **Features:** Sentiment color badges, stock ticker badges, AI summary blocks, modal popup for full content, 10s auto-polling
- **Data flow:** `fetchArticles()` → `GET /api/v1/articles` → DOM diffing by content hash

---

## 3. MLOps Infrastructure

Deployed via Docker Compose (`docker-compose.yml`) with three services:

### 3.1 MLflow (port 5001)

- **Image:** `ghcr.io/mlflow/mlflow:latest`
- **Backend:** SQLite (`/tmp/mlflow.db`)
- **Tracks:** Learning rate, epochs, batch size, loss curves, ROUGE/F1/accuracy metrics, model artifacts
- **Model Registry:** Staging → Production stage transitions. API auto-loads Production models on startup.

### 3.2 Prometheus (port 9090)

- **Image:** `prom/prometheus:latest`
- **Scrape config:** `configs/prometheus.yml` — targets `host.docker.internal:8000` every 15s
- **Metrics collected:**
  - `api_request_count_total` (Counter) — by method, endpoint, HTTP status
  - `api_request_latency_seconds` (Histogram) — request processing time distribution

### 3.3 Grafana (port 3000)

- **Image:** `grafana/grafana:latest`
- **Auto-provisioned:**
  - Datasource: Prometheus at `http://prometheus:9090` (via `configs/grafana/provisioning/datasources/prometheus.yml`)
  - Dashboard: "Financial News API - Monitoring" (via `configs/grafana/dashboards/api_monitoring.json`)
- **Dashboard panels (5):**
  1. Request Rate per second (timeseries) — `rate(api_request_count_total[1m])`
  2. Request Latency p50/p95/p99 (timeseries) — `histogram_quantile` on latency buckets
  3. Total Requests by Endpoint (stat)
  4. Average Latency by Endpoint (stat)
  5. Error Count — non-200 responses (stat)
- **Auto-refresh:** 10s, default time range: last 1h

---

## 4. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10–3.13 |
| Deep Learning | PyTorch, HuggingFace Transformers |
| Pre-trained Model | PhoBERT (`vinai/phobert-base`) |
| Web Framework | FastAPI + Uvicorn |
| Database | SQLite + SQLAlchemy |
| Task Scheduling | APScheduler |
| RSS Parsing | feedparser, BeautifulSoup |
| Frontend | Tailwind CSS (CDN), vanilla JS |
| Containerization | Docker Compose |
| Experiment Tracking | MLflow |
| Metrics Collection | Prometheus + custom FastAPI middleware |
| Monitoring | Grafana (auto-provisioned dashboards) |
| Evaluation | ROUGE (HuggingFace evaluate), seqeval, scikit-learn |

---

## 5. Data Flow Diagram

```
                        ┌──────────────┐
                        │  RSS Feeds   │
                        │  (VnExpress, │
                        │  VnEconomy)  │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │  APScheduler        │
                    │  (every 3 min)      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  rss_scraper.py     │
                    │  feedparser         │
                    │  HTML entity decode │
                    └──────────┬──────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │          POST /api/v1/predict_event      │
          │  ┌─────────┐ ┌──────────┐ ┌───────────┐ │
          │  │  Summ.  │ │Sentiment │ │    NER    │ │
          │  │ BiLSTM  │ │ PhoBERT  │ │  Hybrid   │ │
          │  │ +extract│ │  3-class │ │ Dict+BERT │ │
          │  └────┬────┘ └────┬─────┘ └─────┬─────┘ │
          └───────┼───────────┼─────────────┼───────┘
                  └───────────┼─────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  SQLite Database    │
                   │  (Article table)    │
                   └──────────┬──────────┘
                              │
               ┌──────────────▼──────────────┐
               │  GET /api/v1/articles       │
               │  (on-the-fly NLP fallback)  │
               └──────────────┬──────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Dashboard (HTML)   │
                   │  Auto-poll 10s      │
                   │  Timeline + Modal   │
                   └─────────────────────┘
```
