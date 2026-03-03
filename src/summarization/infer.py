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


def _extractive_summarize(text: str, max_sentences: int = 3) -> str:
    """
    Simple extractive summarization fallback.
    Picks the top-N sentences by a basic importance score (length + position bias).
    Used when the seq2seq model produces degenerate output.
    """
    # Split into sentences using Vietnamese/general punctuation
    sentences = re.split(r'(?<=[.!?。])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return _clean_html_entities(text[:200].strip()) if len(text) > 200 else _clean_html_entities(text.strip())

    if len(sentences) <= max_sentences:
        return _clean_html_entities(" ".join(sentences))

    # Score sentences: prefer longer sentences earlier in the text
    scored = []
    for i, sent in enumerate(sentences):
        word_count = len(sent.split())
        # Position bias: first sentences are usually more important
        position_score = max(0, 1.0 - (i / len(sentences)) * 0.5)
        # Length bonus: prefer medium-length sentences (10-40 words)
        length_score = min(word_count / 20.0, 1.0) if word_count < 40 else 0.8
        score = position_score * 0.6 + length_score * 0.4
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
    """Class wrapper for inferring summaries using the Seq2Seq model and beam search."""
    
    def __init__(self, model_dir: str = "models/summarizer", device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_dir = model_dir
        
        # Load config.json if available, otherwise fall back to defaults
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Loaded model config from {config_path}")
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

        # Load tokenizer from saved model dir, fall back to HuggingFace Hub
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.info(f"Loaded tokenizer from {model_dir}")
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Loaded tokenizer from {tokenizer_name}")
        
        INPUT_DIM = VOCAB_SIZE
        OUTPUT_DIM = VOCAB_SIZE
        
        self.encoder = EncoderBiLSTM(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS).to(self.device)
        self.decoder = AttnDecoderLSTM(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS).to(self.device)
        self.model = Seq2SeqSummarizer(self.encoder, self.decoder, self.device).to(self.device)
        
        # Load state dict if available
        weights_path = os.path.join(model_dir, "summarizer.pt")
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Successfully loaded fine-tuned summarizer model from {weights_path}")
        except Exception as e:
            logger.warning(f"Could not load trained model from {weights_path}: {e}. Running with initialized weights.")

        # Special Tokens (from tokenizer)
        self.SOS_IDX = self._tokenizer.cls_token_id or self._tokenizer.bos_token_id or 0
        self.EOS_IDX = self._tokenizer.sep_token_id or self._tokenizer.eos_token_id or 2
             
    def summarize(self, text: str, max_len: int = None, beam_width: int = 3) -> str:
        """Takes raw text and outputs a summary string.
        Falls back to extractive summarization if the seq2seq model produces degenerate output.
        """
        if max_len is None:
            max_len = self.trg_max_len
        
        # 1. Tokenize the input text using the saved tokenizer
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.src_max_len,
            padding="max_length",
        )
        src_tensor = encoded["input_ids"].to(self.device)
        
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src_tensor)
            
            output_tokens = decode_beam_search(
                self.decoder, 
                encoder_outputs, 
                hidden, 
                cell,
                self.SOS_IDX, 
                self.EOS_IDX, 
                max_len, 
                beam_width, 
                self.device
            )
            
        # 2. Decode token IDs back to a readable string
        summary = self._tokenizer.decode(output_tokens, skip_special_tokens=True)

        # 3. Check for degenerate / collapsed model output
        summary_lower = summary.lower().strip()
        is_degenerate = (
            len(summary.strip()) < 5
            or any(pat in summary_lower for pat in _DEGENERATE_PATTERNS)
            or len(set(output_tokens)) <= 2  # repetitive single-token output
        )

        if is_degenerate:
            logger.info("Seq2Seq model produced degenerate output; falling back to extractive summarization.")
            summary = _extractive_summarize(text)

        return summary

if __name__ == "__main__":
    summarizer = SummarizerInference()
    result = summarizer.summarize("This is a long financial news article about VCB stocks rising heavily.")
    print(result)
