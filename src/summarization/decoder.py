import torch
import torch.nn as nn
from src.summarization.attention import BahdanauAttention

class AttnDecoderLSTM(nn.Module):
    """LSTM Decoder with Bahdanau Attention."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.1):
        super(AttnDecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_size)
        
        # Input to LSTM: embedding_dim + context vector (hidden_size * 2 from BiLSTM)
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size * 3, vocab_size) 
        # (decoder_hidden + context + embed = alternative architecture)
        # Standard: out = linear(decoder_hidden)
        # We'll use a linear layer taking decoder hidden state to vocab
        self.fc_out_simple = nn.Linear(hidden_size, vocab_size)

        # Bridge from encoder hidden state (2*hidden_size) to decoder state (hidden_size)
        self.bridge_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.bridge_cell = nn.Linear(hidden_size * 2, hidden_size)

    def init_hidden(self, encoder_final_hidden, encoder_final_cell):
        """Initializes decoder hidden states from encoder."""
        # [batch_size, hidden_size * 2] -> [batch_size, hidden_size] -> [1, batch_size, hidden_size]
        h = self.bridge_hidden(encoder_final_hidden).unsqueeze(0)
        c = self.bridge_cell(encoder_final_cell).unsqueeze(0)
        return h, c

    def forward(self, input_step, hidden_state, cell_state, encoder_outputs, mask=None):
        """
        input_step: [batch_size, 1] (one token at a time)
        hidden_state: [1, batch_size, hidden_size] 
        cell_state: [1, batch_size, hidden_size]
        encoder_outputs: [batch_size, src_seq_len, hidden_size * 2]
        """
        # [batch_size, 1, embedding_dim]
        embedded = self.dropout(self.embedding(input_step))
        
        # Calculate attention context
        # hidden_state.squeeze(0) -> [batch_size, hidden_size]
        context_vector, attention_weights = self.attention(hidden_state.squeeze(0), encoder_outputs, mask)
        
        # Add time dimension to context_vector -> [batch_size, 1, hidden_size * 2]
        context_vector = context_vector.unsqueeze(1)
        
        # Concatenate embedded input and context vector
        # [batch_size, 1, embedding_dim + hidden_size * 2]
        rnn_input = torch.cat((embedded, context_vector), dim=2)
        
        # Pass through LSTM
        output, (hidden_state, cell_state) = self.lstm(rnn_input, (hidden_state, cell_state))
        
        # Optional: concatenate context with decoder output before vocab mapping
        # Here we just map the decoder output to vocabulary
        # output shape: [batch_size, 1, hidden_size] -> FC maps to [batch_size, vocab_size]
        prediction = self.fc_out_simple(output.squeeze(1))
        
        return prediction, hidden_state, cell_state, attention_weights
