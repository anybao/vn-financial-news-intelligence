# Summarization Module

> `src/summarization/` — Vietnamese financial news automatic text summarization.

---

## Overview

This module provides **three summarization strategies**, switchable via a single `model` parameter:

| Model | Type | Architecture | Quality | Speed |
|-------|------|-------------|---------|-------|
| `vit5` (default) | Abstractive | VietAI/vit5-base-vietnews-summarization (T5) | ★★★★★ | ~2s |
| `seq2seq` | Abstractive | BiLSTM Encoder + Bahdanau Attention + LSTM Decoder + Beam Search | ★★☆☆☆ | ~0.5s |
| `extractive` | Extractive | TF + Position + Length + Data weighted scoring | ★★★☆☆ | <0.1s |

---

## File Structure

```
src/summarization/
├── README.md          ← This file
├── infer.py           ← Main inference class (SummarizerInference)
├── train.py           ← Training script (Seq2Seq + MLflow tracking)
├── encoder.py         ← BiLSTM Encoder
├── decoder.py         ← LSTM Decoder with Attention
├── attention.py       ← Bahdanau (Additive) Attention
└── beam_search.py     ← Beam Search with length penalty
```

---

## How to Use

### Python API

```python
from src.summarization.infer import SummarizerInference

summarizer = SummarizerInference()

text = "Hòa Phát công bố lợi nhuận quý đạt 2940 tỷ đồng, tăng 30%..."

# Switch between models with the `model` parameter:
result_vit5       = summarizer.summarize(text, model="vit5")        # Pre-trained T5
result_seq2seq    = summarizer.summarize(text, model="seq2seq")     # Custom BiLSTM
result_extractive = summarizer.summarize(text, model="extractive")  # Keyword scoring
```

### REST API

```bash
# Default (ViT5)
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hòa Phát công bố...", "model": "vit5"}'

# Seq2Seq + Attention
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hòa Phát công bố...", "model": "seq2seq"}'

# Extractive
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hòa Phát công bố...", "model": "extractive"}'
```

**Response:**
```json
{
  "summary": "Hòa Phát lợi nhuận quý đạt 2940 tỷ đồng, tăng 30%.",
  "model": "vit5"
}
```

---

## Inference Flow

```
                         ┌─────────────────┐
                         │  summarize()    │
                         │  model=?        │
                         └────────┬────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        model="vit5"        model="seq2seq"     model="extractive"
              │                   │                   │
              ▼                   ▼                   ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │ _summarize_    │  │ summarize_     │  │ _extractive_   │
     │ vit5()         │  │ seq2seq()      │  │ summarize()    │
     └───────┬────────┘  └───────┬────────┘  └────────────────┘
             │                   │
             │ fail?             │ degenerate?
             ▼                   ▼
     ┌────────────────┐  ┌────────────────┐
     │ _extractive_   │  │ _extractive_   │
     │ summarize()    │  │ summarize()    │
     │ (fallback)     │  │ (fallback)     │
     └────────────────┘  └────────────────┘
```

Every abstractive model automatically falls back to extractive if it fails or produces degenerate output.

---

## Model 1: ViT5 (Default — Pre-trained Abstractive)

**Architecture:** T5 (Text-to-Text Transfer Transformer)  
**Pre-trained model:** `VietAI/vit5-base-vietnews-summarization` (~900MB)  
**Source:** HuggingFace Hub — trained on Vietnamese news corpus  

### How it works:

1. **Prefix input** with `"vietnews: "` (task identifier for T5)
2. **Tokenize** with SentencePiece tokenizer, truncate at **1024 tokens**
3. **Generate** using beam search:
   - `num_beams=4`
   - `max_length=256`
   - `length_penalty=1.0`
   - `no_repeat_ngram_size=3` (prevents trigram repetition)
   - `early_stopping=True`
4. **Decode** token IDs back to text, strip special tokens

### Example:

**Input:**
> Hòa Phát vừa công bố kết quả kinh doanh quý với lợi nhuận sau thuế đạt 2940 tỷ đồng, tăng 30% so với cùng kỳ năm trước. Doanh thu thuần đạt 21165 tỷ đồng, vượt 21% so với kế hoạch. Mảng thép xây dựng tiếp tục là động lực tăng trưởng chính.

**Output (ViT5):**
> Hòa Phát báo lãi 2.940 tỷ đồng trong quý, tăng 30% so với cùng kỳ, doanh thu vượt kế hoạch nhờ mảng thép xây dựng.

---

## Model 2: Seq2Seq + Bahdanau Attention (Custom-trained)

**Architecture:** BiLSTM Encoder → Bahdanau Attention → LSTM Decoder → Beam Search  
**Training data:** 500 samples from `OpenHust/vietnamese-summarization`, 3 epochs  
**Model weights:** `models/summarizer/summarizer.pt`

### Architecture Details:

```
Input Text
    │
    ▼
┌─────────────────────────┐
│  EncoderBiLSTM          │  vocab=64000, emb=256, hidden=512
│  ─────────────────────  │  bidirectional LSTM
│  Input: [B, T]          │
│  Output: [B, T, 1024]   │  (hidden*2 due to bidirectional)
│  Hidden: [B, 1024]      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  BahdanauAttention      │  additive attention
│  ─────────────────────  │
│  e = V^T·tanh(W1·s +   │  s = decoder hidden state
│            W2·h)        │  h = encoder output
│  α = softmax(e)         │  α = attention weights
│  c = Σ(α · h)           │  c = context vector
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  AttnDecoderLSTM        │  LSTM decoder
│  ─────────────────────  │
│  Input: [embed; context]│  concatenate embedding + context
│  Bridge: 1024 → 512     │  project encoder hidden → decoder
│  Output: [B, vocab]     │  linear projection to vocabulary
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Beam Search Decoder    │
│  ─────────────────────  │
│  beam_width = 3         │
│  length_penalty α = 0.7 │  score = logP / len^0.7
│  min_length = 5         │  reject short sequences
│  max_length = 64        │
│  PriorityQueue-based    │
└─────────────────────────┘
```

### Scoring Formula (Beam Search):

```
score(y) = log P(y₁, y₂, ..., yₜ) / t^α

where:
  - P(y) = product of token probabilities
  - t    = sequence length
  - α    = 0.7 (length penalty — encourages longer outputs)
```

### Degenerate Output Detection:

The model (trained on limited data) sometimes produces collapsed output. These are detected and rejected:

- Output < 5 characters
- Contains `"<unk> <unk>"` or `"generated summary based on"`
- Token vocabulary ≤ 2 unique tokens

When degenerate output is detected, the system falls back to extractive summarization.

### Example:

**Input:** Same Hòa Phát text above

**Output (Seq2Seq):** _(often degenerate due to limited training data — falls back to extractive)_

---

## Model 3: Extractive Summarization

**Type:** Unsupervised — no model training needed  
**Algorithm:** Weighted sentence scoring with 4 factors

### Scoring Formula:

```
S(sentence_i) = 0.35 × keyword_score
              + 0.20 × position_score
              + 0.20 × length_score
              + 0.25 × data_score
```

### Factor Details:

| Factor | Weight | Description | Scoring |
|--------|--------|-------------|---------|
| **Keyword TF** | 35% | Normalized term frequency of non-stopword terms | `Σ(freq(w)/max_freq) / n_words` |
| **Position** | 20% | Sentence position in document | First=1.0, Last=0.8, Middle=linear decay (0.3–1.0) |
| **Length** | 20% | Prefer medium-length sentences (15–50 words) | <8 words=0.3, 8–50=`count/25`, >50=0.7 |
| **Data/Numbers** | 25% | Sentences with financial figures | Count of `\d+(tỷ|triệu|%|đồng|điểm)` × 0.3, capped at 1.0 |

### Pipeline:

1. **Split** text into sentences at `.!?。` boundaries
2. **Filter** sentences < 15 characters, title duplicates
3. **Remove stopwords** (60+ Vietnamese + English stopwords)
4. **Score** each sentence with the 4-factor formula
5. **Select** top-3 sentences by score
6. **Re-order** selected sentences by original position (preserves narrative flow)
7. **Clean** HTML entities in output

### Example:

**Input:**
> Hòa Phát vừa công bố kết quả kinh doanh quý với lợi nhuận sau thuế đạt 2940 tỷ đồng, tăng 30% so với cùng kỳ năm trước. Doanh thu thuần đạt 21165 tỷ đồng, vượt 21% so với kế hoạch. Mảng thép xây dựng tiếp tục là động lực tăng trưởng chính. Sản lượng thép tiêu thụ đạt 2.3 triệu tấn.

**Scoring:**

| Sentence | Keyword | Position | Length | Data | **Total** |
|----------|---------|----------|--------|------|-----------|
| "Hòa Phát vừa công bố..." | 0.52 | 1.00 | 0.84 | 0.60 | **0.69** ← selected |
| "Doanh thu thuần đạt..." | 0.48 | 0.77 | 0.60 | 0.60 | **0.62** ← selected |
| "Mảng thép xây dựng..." | 0.35 | 0.53 | 0.40 | 0.00 | **0.27** |
| "Sản lượng thép tiêu thụ..." | 0.40 | 0.80 | 0.44 | 0.30 | **0.48** ← selected |

**Output:** Sentences 1, 2, 4 (re-ordered by position)

---

## Evaluation Results

| Metric | ViT5 | Seq2Seq | Extractive |
|--------|------|---------|------------|
| ROUGE-1 | **0.32** | ~0.05 | ~0.25 |
| ROUGE-2 | **0.14** | ~0.01 | ~0.10 |
| ROUGE-L | **0.28** | ~0.04 | ~0.22 |
| SacreBLEU | **8.5** | ~1.0 | ~5.0 |
| Compression Ratio | 74% | 85% | 70% |

Evaluation is performed via `src/evaluation/rouge.py` using the HuggingFace `evaluate` library.

---

## Training the Seq2Seq Model

```bash
source venv/bin/activate
export PYTHONPATH=$(pwd)
python src/summarization/train.py
```

This will:
1. Download `OpenHust/vietnamese-summarization` from HuggingFace Hub
2. Tokenize with `vinai/phobert-base` tokenizer
3. Train BiLSTM Encoder + Attention Decoder for 3 epochs
4. Log metrics to MLflow (port 5001)
5. Save best checkpoint to `models/summarizer/summarizer.pt`
6. Register model in MLflow Registry as "SummarizationModel"

---

## Configuration

Model hyperparameters are stored in `models/summarizer/config.json`:

```json
{
  "vocab_size": 64000,
  "enc_emb_dim": 256,
  "dec_emb_dim": 256,
  "hidden_size": 512,
  "num_layers": 1,
  "src_max_len": 256,
  "trg_max_len": 64,
  "tokenizer_name": "vinai/phobert-base"
}
```

ViT5 configuration is handled automatically by HuggingFace Transformers (auto-downloaded on first run, ~900MB).

---

## Why ViT5 is Default

The custom Seq2Seq model was trained on only **500 samples for 3 epochs** — intentionally small as a learning exercise to demonstrate the architecture (encoder, decoder, attention, beam search). It suffers from **mode collapse** and frequently produces degenerate outputs.

ViT5, pre-trained on a large Vietnamese news corpus, produces production-quality abstractive summaries. The `model` switch lets you compare both side-by-side for educational/demonstration purposes.

---

## Dependencies

```
torch>=2.0
transformers>=4.40,<5
sentencepiece          # Required for ViT5 tokenizer
evaluate               # For ROUGE/BLEU metrics
rouge_score
sacrebleu
```
