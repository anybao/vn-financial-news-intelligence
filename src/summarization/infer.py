import torch
import json
import html
import os
import re
import logging
from src.summarization.train import Seq2SeqSummarizer
from src.summarization.beam_search import decode_beam_search
from src.summarization.encoder import EncoderBiLSTM
from src.summarization.decoder import AttnDecoderLSTM
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _clean_html_entities(text: str) -> str:
    """Decode HTML entities including broken ones (e.g. #225; without &)."""
    if not text:
        return ""
    # Fix broken numeric entities: #225; -> &#225;
    text = re.sub(r'(?<!&)#(\d+);', r'&#\1;', text)
    text = html.unescape(text)
    return text


def _extractive_summarize(text: str, max_sentences: int = 3, exclude_title: str = None) -> str:
    """
    Extractive summarization fallback using keyword frequency + position + length scoring.
    Picks the top-N most informative sentences from the input text.
    Used when the seq2seq model produces degenerate output.
    """
    # Split into sentences using Vietnamese/general punctuation
    sentences = re.split(r'(?<=[.!?。])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    # Filter out sentences that are just the title repeated
    if exclude_title:
        title_norm = exclude_title.strip().rstrip('.!?').lower()
        sentences = [s for s in sentences if s.strip().rstrip('.!?').lower() != title_norm]

    if not sentences:
        return _clean_html_entities(text[:200].strip()) if len(text) > 200 else _clean_html_entities(text.strip())

    if len(sentences) <= max_sentences:
        return _clean_html_entities(" ".join(sentences))

    # ── Build word frequency map (simple TF-IDF-like keyword importance) ──
    # Vietnamese stopwords (common words that don't carry meaning)
    _stopwords = {
        "và", "của", "các", "là", "có", "được", "cho", "trong", "với", "này",
        "đã", "để", "không", "từ", "một", "những", "theo", "về", "khi", "cũng",
        "như", "đó", "nhưng", "còn", "hay", "vào", "bởi", "tại", "hơn", "ra",
        "do", "nên", "thì", "sẽ", "đến", "lại", "mà", "rằng", "nếu", "đều",
        "qua", "trên", "sau", "trước", "giữa", "dù", "tuy", "vì", "ông", "bà",
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for",
    }

    # Tokenize all words across the document
    all_words = []
    sentence_words = []
    for sent in sentences:
        words = [w.lower() for w in re.findall(r'\w+', sent) if len(w) > 1 and w.lower() not in _stopwords]
        sentence_words.append(words)
        all_words.extend(words)

    # Word frequency across the whole document
    from collections import Counter
    word_freq = Counter(all_words)
    max_freq = max(word_freq.values()) if word_freq else 1

    # ── Score each sentence ──
    scored = []
    for i, (sent, words) in enumerate(zip(sentences, sentence_words)):
        word_count = len(sent.split())

        # 1. Keyword score: sum of normalized word frequencies in this sentence
        if words:
            keyword_score = sum(word_freq[w] / max_freq for w in words) / len(words)
        else:
            keyword_score = 0.0

        # 2. Position score: first & last sentences get a small bonus, but NOT dominant
        if i == 0:
            position_score = 1.0
        elif i == len(sentences) - 1:
            position_score = 0.8
        else:
            position_score = max(0.3, 1.0 - (i / len(sentences)) * 0.7)

        # 3. Length score: prefer medium-length sentences (15-50 words)
        if word_count < 8:
            length_score = 0.3
        elif word_count <= 50:
            length_score = min(word_count / 25.0, 1.0)
        else:
            length_score = 0.7

        # 4. Number/data score: sentences with numbers (percentages, money) are often key
        num_count = len(re.findall(r'\d+[.,]?\d*\s*(%|tỷ|triệu|nghìn|phần trăm|đồng|điểm)', sent, re.IGNORECASE))
        data_score = min(num_count * 0.3, 1.0)

        # Weighted combination: keyword-heavy to capture important content
        score = (keyword_score * 0.35
                 + position_score * 0.20
                 + length_score * 0.20
                 + data_score * 0.25)
        scored.append((score, i, sent))

    # Sort by score descending, pick top-N, then re-order by original position
    top = sorted(scored, key=lambda x: -x[0])[:max_sentences]
    top = sorted(top, key=lambda x: x[1])  # restore original order

    return _clean_html_entities(" ".join(s for _, _, s in top))


# Known degenerate outputs produced by collapsed models
_DEGENERATE_PATTERNS = [
    "generated summary based on",
    "decoded tokens",
    "<unk> <unk>",
]


class SummarizerInference:
    """Summarizer with ViT5 abstractive (primary) → extractive (fallback).

    The custom Seq2Seq BiLSTM model is still loaded for demonstration purposes
    (e.g. /api/v1/model_info) but ViT5 handles actual inference.
    """

    _VIT5_MODEL_NAME = "VietAI/vit5-base-vietnews-summarization"

    def __init__(self, model_dir: str = "models/summarizer", device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model_dir = model_dir

        # ── 1. Load pre-trained ViT5 abstractive model (primary) ──────────
        self._vit5_model = None
        self._vit5_tokenizer = None
        try:
            from transformers import AutoTokenizer as AT, AutoModelForSeq2SeqLM
            self._vit5_tokenizer = AT.from_pretrained(self._VIT5_MODEL_NAME)
            self._vit5_model = AutoModelForSeq2SeqLM.from_pretrained(self._VIT5_MODEL_NAME).to(self.device)
            self._vit5_model.eval()
            logger.info(f"Loaded ViT5 abstractive model: {self._VIT5_MODEL_NAME}")
        except Exception as e:
            logger.warning(f"Could not load ViT5 model ({self._VIT5_MODEL_NAME}): {e}. "
                           "Will use extractive summarization only.")

        # ── 2. Load custom Seq2Seq BiLSTM (kept for project demonstration) ─
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded Seq2Seq config from {config_path}")
            VOCAB_SIZE = config["vocab_size"]
            ENC_EMB_DIM = config["enc_emb_dim"]
            DEC_EMB_DIM = config["dec_emb_dim"]
            HID_DIM = config["hidden_size"]
            N_LAYERS = config["num_layers"]
            self.src_max_len = config.get("src_max_len", 256)
            self.trg_max_len = config.get("trg_max_len", 64)
            tokenizer_name = config.get("tokenizer_name", "vinai/phobert-base")
        else:
            logger.warning(f"No config.json found at {config_path}. Using default parameters.")
            VOCAB_SIZE = 64000
            ENC_EMB_DIM = 256
            DEC_EMB_DIM = 256
            HID_DIM = 512
            N_LAYERS = 1
            self.src_max_len = 256
            self.trg_max_len = 64
            tokenizer_name = "vinai/phobert-base"

        try:
            self._seq2seq_tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.info(f"Loaded Seq2Seq tokenizer from {model_dir}")
        except Exception:
            self._seq2seq_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Loaded Seq2Seq tokenizer from {tokenizer_name}")

        self.encoder = EncoderBiLSTM(VOCAB_SIZE, ENC_EMB_DIM, HID_DIM, N_LAYERS).to(self.device)
        self.decoder = AttnDecoderLSTM(VOCAB_SIZE, DEC_EMB_DIM, HID_DIM, N_LAYERS).to(self.device)
        self.model = Seq2SeqSummarizer(self.encoder, self.decoder, self.device).to(self.device)

        weights_path = os.path.join(model_dir, "summarizer.pt")
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded Seq2Seq weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Could not load Seq2Seq weights from {weights_path}: {e}")

        self.SOS_IDX = self._seq2seq_tokenizer.cls_token_id or self._seq2seq_tokenizer.bos_token_id or 0
        self.EOS_IDX = self._seq2seq_tokenizer.sep_token_id or self._seq2seq_tokenizer.eos_token_id or 2

    # ── Supported model names ─────────────────────────────────────────────
    MODELS = ("vit5", "seq2seq", "extractive")

    # ── Public API ────────────────────────────────────────────────────────

    def summarize(self, text: str, max_len: int = 256, model: str = "vit5") -> str:
        """Generate a summary using the selected model.

        Args:
            text:    Input article text.
            max_len: Maximum output length (tokens for abstractive, sentences for extractive).
            model:   One of ``'vit5'`` (default), ``'seq2seq'``, or ``'extractive'``.
                     Falls back to extractive if the chosen model fails.

        Returns:
            Generated summary string.
        """
        model = model.lower().strip()
        text = _clean_html_entities(text)

        # ── Route to selected model ───────────────────────────────────────
        if model == "seq2seq":
            logger.info("Model switch → Seq2Seq BiLSTM + Bahdanau Attention")
            summary = self.summarize_seq2seq(text, max_len=max_len)
            if summary:
                logger.info(f"Seq2Seq summary generated ({len(summary)} chars)")
                return summary
            logger.warning("Seq2Seq produced degenerate output; falling back to extractive.")
            return _extractive_summarize(text)

        if model == "extractive":
            logger.info("Model switch → Extractive summarization")
            return _extractive_summarize(text)

        # Default: model == "vit5"
        # 1. Try ViT5 abstractive summarization
        if self._vit5_model is not None:
            try:
                summary = self._summarize_vit5(text, max_len)
                if summary and len(summary.strip()) > 10:
                    logger.info(f"ViT5 abstractive summary generated ({len(summary)} chars)")
                    return summary
                logger.warning("ViT5 returned empty/short output; falling back to extractive.")
            except Exception as e:
                logger.warning(f"ViT5 inference failed: {e}; falling back to extractive.")

        # 2. Fallback: extractive summarization
        logger.info("Using extractive summarization fallback.")
        return _extractive_summarize(text)

    def summarize_seq2seq(self, text: str, max_len: int = None, beam_width: int = 3) -> str:
        """Run the custom Seq2Seq BiLSTM model (kept for demonstration / evaluation)."""
        if max_len is None:
            max_len = self.trg_max_len

        encoded = self._seq2seq_tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.src_max_len, padding="max_length",
        )
        src_tensor = encoded["input_ids"].to(self.device)

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src_tensor)
            output_tokens = decode_beam_search(
                self.decoder, encoder_outputs, hidden, cell,
                self.SOS_IDX, self.EOS_IDX, max_len, beam_width, self.device,
            )

        summary = self._seq2seq_tokenizer.decode(output_tokens, skip_special_tokens=True)

        summary_lower = summary.lower().strip()
        is_degenerate = (
            len(summary.strip()) < 5
            or any(pat in summary_lower for pat in _DEGENERATE_PATTERNS)
            or len(set(output_tokens)) <= 2
        )
        if is_degenerate:
            logger.info("Seq2Seq produced degenerate output.")
            return ""
        return summary

    # ── Private helpers ───────────────────────────────────────────────────

    def _summarize_vit5(self, text: str, max_len: int = 256) -> str:
        """Run VietAI/vit5-base-vietnews-summarization."""
        input_text = "vietnews: " + text
        inputs = self._vit5_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding="longest",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._vit5_model.generate(
                **inputs,
                max_length=max_len,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        summary = self._vit5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary.strip()


if __name__ == "__main__":
    summarizer = SummarizerInference()
    test = ("Hòa Phát vừa công bố kết quả kinh doanh quý với lợi nhuận sau thuế "
            "đạt 2940 tỷ đồng, tăng 30% so với cùng kỳ năm trước. "
            "Doanh thu thuần đạt 21165 tỷ đồng, vượt 21% so với kế hoạch.")
    print("=== ViT5 (default) ===")
    print(summarizer.summarize(test, model="vit5"))
    print("\n=== Seq2Seq + Attention ===")
    print(summarizer.summarize(test, model="seq2seq"))
    print("\n=== Extractive ===")
    print(summarizer.summarize(test, model="extractive"))
