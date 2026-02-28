import torch
import torch.nn as nn
from torch import optim
from src.summarization.encoder import EncoderBiLSTM
from src.summarization.decoder import AttnDecoderLSTM
import logging

logger = logging.getLogger(__name__)

class Seq2SeqSummarizer(nn.Module):
    """Wrapper model combining Encoder and Decoder."""
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqSummarizer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        """Standard forward pass usually used during training."""
        batch_size = source.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.vocab_size
        
        # Store outputs
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        
        # Encoder forward
        encoder_outputs, hidden, cell = self.encoder(source)
        
        # Initialize decoder hidden with mapped encoder hidden
        hidden, cell = self.decoder.init_hidden(hidden, cell)
        
        # First input to the decoder is the <sos> token
        input_token = target[:, 0].unsqueeze(1)
        
        for t in range(1, target_len):
            # Pass through decoder
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs)
            
            # Store prediction
            outputs[:, t, :] = output
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            input_token = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs

def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    """Training loop for a single epoch."""
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(dataloader):
        # Dummy batch extraction (assuming source and target are yielded)
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        # output: [batch_size, trg_len, vocab_size] -> reshape to [batch_size * trg_len, vocab_size]
        # trg: [batch_size, trg_len] -> reshape to [batch_size * trg_len]
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

if __name__ == "__main__":
    logger.info("Initializing Summarizer Models...")
    # These parameters would normally come from config
    INPUT_DIM = 50000
    OUTPUT_DIM = 50000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 1
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    enc = EncoderBiLSTM(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = AttnDecoderLSTM(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    
    model = Seq2SeqSummarizer(enc, dec, device).to(device)
    logger.info("Summarization model ready for training.")
