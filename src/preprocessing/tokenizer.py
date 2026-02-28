from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class TokenizerWrapper:
    """Wrapper around HuggingFace tokenizers (e.g., PhoBERT)."""
    
    def __init__(self, model_name: str = "vinai/phobert-base"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load tokenizer {model_name}: {e}")
            self.tokenizer = None

    def tokenize(self, text: str, max_length: int = 256, padding: bool = True, truncation: bool = True):
        """Tokenizes input text and returns input_ids and attention_mask."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized.")
            
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors="pt"
        )
        
    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decodes token IDs back to a string."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized.")
            
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
