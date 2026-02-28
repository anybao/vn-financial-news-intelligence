import torch
import torch.nn as nn

class EncoderBiLSTM(nn.Module):
    """Bidirectional LSTM Encoder for Seq2Seq summarization."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.1):
        super(EncoderBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, src_seq, src_lengths=None):
        """
        src_seq: [batch_size, seq_len]
        src_lengths: Lengths for pack_padded_sequence optionally.
        """
        # embedded: [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(src_seq))
        
        # Optional: Pack padded sequence if lengths provided
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
        # output: [batch_size, seq_len, hidden_size * 2]
        # hidden, cell: [num_layers * 2, batch_size, hidden_size]
        outputs, (hidden, cell) = self.lstm(embedded)
        
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            
        # Concatenate bidirectional hidden states for the decoder initialization
        # Forward hidden: hidden[0:num_layers*2:2]
        # Backward hidden: hidden[1:num_layers*2:2]
        # Combine them -> [num_layers, batch_size, hidden_size*2] => but decoder usually expects hidden_size
        # One simple way is to sum them or use a linear layer
        
        # We'll map hidden_size * 2 to hidden_size for decoder init
        # shape [batch_size, hidden_size * 2] (taking last layer)
        final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        final_cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        
        return outputs, final_hidden, final_cell
