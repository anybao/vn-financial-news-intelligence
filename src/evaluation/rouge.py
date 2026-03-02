import evaluate
from typing import List

class SummarizationEvaluator:
    """Evaluates generated summaries against reference summaries using ROUGE and BLEU."""
    
    def __init__(self):
        try:
            self.rouge = evaluate.load('rouge')
            self.bleu = evaluate.load('sacrebleu')
        except Exception as e:
            print(f"Failed to load metrics: {e}. Please ensure 'evaluate', 'rouge_score', and 'sacrebleu' are installed.")
            self.rouge = None
            self.bleu = None

    def compute_scores(self, predictions: List[str], references: List[str]) -> dict:
        """
        Computes ROUGE-1, ROUGE-2, ROUGE-L and BLEU metrics.
        
        Args:
            predictions (List[str]): List of generated summaries.
            references (List[str]): List of ground-truth summaries (or lists of ground truths).
            
        Returns:
            dict: Dictionary containing ROUGE and BLEU scores.
        """
        results = {}
        
        if self.rouge:
            rouge_res = self.rouge.compute(predictions=predictions, references=references)
            results.update(rouge_res)
            
        if self.bleu:
            # sacrebleu requires references to be a list of lists of strings
            bleu_refs = [[ref] for ref in references]
            bleu_res = self.bleu.compute(predictions=predictions, references=bleu_refs)
            results["sacrebleu"] = bleu_res["score"]
            
        return results

if __name__ == "__main__":
    evaluator = SummarizationEvaluator()
    
    # Dummy data
    preds = ["Chỉ số VN-Index tăng mạnh do dòng tiền khối ngoại."]
    refs = ["Thị trường chứng khoán Việt Nam khởi sắc nhờ khối ngoại mua ròng."]
    
    scores = evaluator.compute_scores(preds, refs)
    print(f"Metrics: {scores}")
