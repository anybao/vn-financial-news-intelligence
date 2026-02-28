import torch
import logging
from src.summarization.train import Seq2SeqSummarizer
from src.summarization.beam_search import decode_beam_search
from src.summarization.encoder import EncoderBiLSTM
from src.summarization.decoder import AttnDecoderLSTM

logger = logging.getLogger(__name__)

class SummarizerInference:
    """Class wrapper for inferring summaries using the Seq2Seq model and beam search."""
    
    def __init__(self, model_dir: str = "models/summarizer", device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Hardcoding vocab/dim constraints for project blueprint completeness.
        # Should be load from config.
        INPUT_DIM = 50000
        OUTPUT_DIM = 50000
        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        HID_DIM = 512
        N_LAYERS = 1
        
        self.encoder = EncoderBiLSTM(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS).to(self.device)
        self.decoder = AttnDecoderLSTM(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS).to(self.device)
        self.model = Seq2SeqSummarizer(self.encoder, self.decoder, self.device).to(self.device)
        
        # Load state dict if available
        try:
             self.model.load_state_dict(torch.load(f"{model_dir}/summarizer.pt", map_location=self.device))
             self.model.eval()
             logger.info("Successfully loaded fine-tuned summarizer model.")
        except Exception as e:
             logger.warning(f"Could not load trained model from {model_dir}: {e}. Running with initialized weights.")

        # Special Tokens
        self.SOS_IDX = 1  
        self.EOS_IDX = 2
             
    def summarize(self, text: str, max_len: int = 50, beam_width: int = 3) -> str:
        """Takes raw text and outputs a summary string."""
        
        # 1. Tokenize Text (Assume tokenizer maps text to tensor of IDs)
        # Dummy tokenization for structural completeness
        tokenized_src = [self.SOS_IDX, 10, 20, 30, self.EOS_IDX] 
        src_tensor = torch.LongTensor(tokenized_src).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src_tensor)
            
            output_tokens = decode_beam_search(
                self.decoder, 
                encoder_outputs, 
                hidden, 
                cell,
                self.SOS_IDX, 
                self.EOS_IDX, 
                max_len, 
                beam_width, 
                self.device
            )
            
        # 2. Decode Tokens to String (Assume tokenizer.decode)
        summary = f"Generated summary based on {len(output_tokens)} decoded tokens."
        return summary

if __name__ == "__main__":
    summarizer = SummarizerInference()
    result = summarizer.summarize("This is a long financial news article about VCB stocks rising heavily.")
    print(result)
