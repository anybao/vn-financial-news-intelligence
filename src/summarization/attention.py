import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention mechanism."""
    
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)     # transforms decoder hidden state
        self.W2 = nn.Linear(hidden_size * 2, hidden_size) # transforms encoder output (BiLSTM = hidden_size * 2)
        self.V = nn.Linear(hidden_size, 1)                # transforms to scalar score

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        hidden: [batch_size, hidden_size]
        encoder_outputs: [batch_size, seq_len, hidden_size * 2]
        """
        # Add seq_len dimension to hidden: [batch_size, 1, hidden_size]
        hidden_with_time_axis = hidden.unsqueeze(1)
        
        # Calculate alignment scores
        # score shape == [batch_size, seq_len, 1]
        score = self.V(torch.tanh(self.W1(hidden_with_time_axis) + self.W2(encoder_outputs)))
        
        # Optional: apply mask to ignore padding tokens
        if mask is not None:
             score = score.masked_fill(mask == 0, -1e10)
             
        # Convert scores to probabilities: [batch_size, seq_len, 1]
        attention_weights = F.softmax(score, dim=1)
        
        # Context vector
        # context shape == [batch_size, 1, hidden_size * 2]
        context_vector = attention_weights * encoder_outputs
        context_vector = torch.sum(context_vector, dim=1)
        
        return context_vector, attention_weights
