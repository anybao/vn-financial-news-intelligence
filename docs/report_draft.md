# Final Project Report (Draft)

## 1. Abstract
This project presents an End-to-End Financial News Intelligence platform designed to extract, analyze, and summarize Vietnamese financial news. By combining advanced Natural Language Processing approaches, specifically Abstractive Summarization with Attention mechanisms, sequence classification for sentiment analysis, and NER token classification for stock ticker extraction, the system delivers structured insights directly from unstructured text. An industry-standard MLOps stack accompanies the deployment.

## 2. Introduction
Financial markets move fast, and traders/investors are constantly overwhelmed by continuous news cycles. The motivation of this system is to mitigate information overload by providing real-time, deduplicated, and sentiment-scored summaries mapped to specific market entities.

## 3. Methodology
### 3.1 Summarization
We utilized an Encoder-Decoder structure with a Bidirectional LSTM on the input and a standard LSTM on the output. We integrated *Bahdanau Additive Attention* to allow the decoder to focus on specific parts of the source text. Beam search decoding was implemented to generate fluent and high-quality generation outputs.

### 3.2 Sentiment Analysis & NER
Using PhoBERT as a foundation base, classification heads were added for both Sequence and Token classification tasks.

### 3.3 Deduplication
Embeddings are hashed upon text ingress. Cosine similarity operations check against existing records, ensuring no overlapping duplicate "events" are re-published.

## 4. Experiments & Output Metrics
*(Placeholders for actual running data upon training)*
- ROUGE-1: ~XX.XX
- ROUGE-2: ~XX.XX
- ROUGE-L: ~XX.XX
- Sentiment Accuracy: XX%
- NER F1 Score: XX%

## 5. Conclusion
The initial pipeline successfully connects raw scattered RSS data points into concrete, actionable APIs, powered by sequence-to-sequence academic foundations combined with robust MLOps practices. Future enhancements include implementing LLMs (like Llama-3) instead of LSTM seq2seq for better zero-shot context translation.
