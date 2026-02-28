import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class SentimentPredictor:
    """Loads a fine-tuned sentiment model and makes predictions."""
    
    def __init__(self, model_path: str = "models/sentiment"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels = ["Negative", "Neutral", "Positive"] # Depending on training labels
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load sentiment model from {model_path}: {e}. Initializing generic base model.")
            # Fallback to base model for initialization testing purposes
            model_path = "vinai/phobert-base"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3).to(self.device)
            self.model.eval()

    def predict(self, text: str) -> dict:
        """Predicts the sentiment of a given text."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probs, dim=-1).item()
            
        return {
            "sentiment": self.labels[predicted_class_id],
            "confidence": probs[0][predicted_class_id].item()
        }

if __name__ == "__main__":
    predictor = SentimentPredictor()
    result = predictor.predict("Cổ phiếu VCB hôm nay tăng trần, thanh khoản vượt trội.")
    print(f"Prediction: {result}")
