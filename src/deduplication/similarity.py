import torch
import torch.nn.functional as F
from typing import List, Dict

class Deduplicator:
    """Handles semantic deduplication utilizing cosine similarity."""
    
    def __init__(self, threshold: float = 0.85):
        """
        Args:
            threshold (float): Cosine similarity threshold for considering two articles as duplicates.
        """
        self.threshold = threshold
        
    def calculate_similarity(self, embed1: torch.Tensor, embed2: torch.Tensor) -> float:
        """Calculates cosine similarity between two embeddings."""
        # Ensure 1D tensors
        if embed1.dim() == 1:
            embed1 = embed1.unsqueeze(0)
        if embed2.dim() == 1:
            embed2 = embed2.unsqueeze(0)
            
        similarity = F.cosine_similarity(embed1, embed2).item()
        return similarity

    def is_duplicate(self, new_embed: torch.Tensor, existing_embeds: List[torch.Tensor]) -> bool:
        """
        Checks if the new embedding is a duplicate against a list of existing embeddings.
        Returns True if the maximum similarity exceeds the threshold.
        """
        if not existing_embeds:
            return False
            
        max_sim = 0.0
        for existing in existing_embeds:
            sim = self.calculate_similarity(new_embed, existing)
            if sim > max_sim:
                max_sim = sim
                
        return max_sim > self.threshold
