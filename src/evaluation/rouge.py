import evaluate
from typing import List

class RougeEvaluator:
    """Evaluates generated summaries against reference summaries using ROUGE."""
    
    def __init__(self):
        try:
            self.rouge = evaluate.load('rouge')
        except Exception as e:
            print(f"Failed to load ROUGE metric: {e}. Please ensure 'evaluate' and 'rouge_score' are installed.")
            self.rouge = None

    def compute_scores(self, predictions: List[str], references: List[str]) -> dict:
        """
        Computes ROUGE-1, ROUGE-2, ROUGE-L metrics.
        
        Args:
            predictions (List[str]): List of generated summaries.
            references (List[str]): List of ground-truth summaries.
            
        Returns:
            dict: Dictionary containing ROUGE scores.
        """
        if not self.rouge:
            return {}
            
        results = self.rouge.compute(predictions=predictions, references=references)
        return results

if __name__ == "__main__":
    evaluator = RougeEvaluator()
    
    # Dummy data
    preds = ["Chỉ số VN-Index tăng mạnh do dòng tiền khối ngoại."]
    refs = ["Thị trường chứng khoán Việt Nam khởi sắc nhờ khối ngoại mua ròng."]
    
    scores = evaluator.compute_scores(preds, refs)
    print(f"ROUGE Scores: {scores}")
