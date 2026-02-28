from transformers import AutoModel, AutoTokenizer
import torch
import logging

logger = logging.getLogger(__name__)

class SemanticEmbedder:
    """Generates embeddings for semantic deduplication using PhoBERT or distilUSE."""
    
    def __init__(self, model_name: str = "vinai/phobert-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Error loading embedder model {model_name}: {e}")
            raise

    def get_embedding(self, text: str) -> torch.Tensor:
        """Generates the sentence embedding based on the CLS token or mean pooling."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256, 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the pooled output or mean of last hidden state
            # PhoBERT usually has a pooler_output
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                # Mean pooling
                attention_mask = inputs['attention_mask']
                last_hidden_state = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
                
        return embedding.cpu().squeeze()
