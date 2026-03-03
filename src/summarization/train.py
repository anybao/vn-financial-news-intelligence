import torch
import torch.nn as nn
from torch import optim
from src.summarization.encoder import EncoderBiLSTM
from src.summarization.decoder import AttnDecoderLSTM
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Initializing Summarization Module (Pre-Main)...")

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
    print("ENTRY: Starting Summarizer Models execution block...")
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
    logger.info("Summarization model initialized.")
    
    import mlflow
    import random
    import os
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("Summarization_Model")
    
    # 1. Real Dataset Ingestion
    logger.info("Loading `OpenHust/vietnamese-summarization` Dataset from HuggingFace...")
    try:
        # Pull 500 samples for fast local training showcase
        dataset = load_dataset("OpenHust/vietnamese-summarization", split="train[:500]")
        eval_dataset = load_dataset("OpenHust/vietnamese-summarization", split="train[500:550]")
        
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        # 2. Truncation and Handling Long Documents
        def process_data(examples):
            # Truncate source articles to max 256 tokens to fit into BiLSTM safely without OOM
            src_encoded = tokenizer(examples["Document"], padding="max_length", max_length=256, truncation=True)
            # Target summary truncated to max 64 tokens 
            trg_encoded = tokenizer(examples["Summary"], padding="max_length", max_length=64, truncation=True)
            return {"src_ids": src_encoded["input_ids"], "trg_ids": trg_encoded["input_ids"]}
            
        logger.info("Mapping tokenization over datasets...")
        dataset = dataset.map(process_data, batched=True, remove_columns=dataset.column_names)
        eval_dataset = eval_dataset.map(process_data, batched=True, remove_columns=eval_dataset.column_names)
        
        dataset.set_format(type="torch", columns=["src_ids", "trg_ids"])
        eval_dataset.set_format(type="torch", columns=["src_ids", "trg_ids"])
        
        logger.info("Initializing DataLoaders...")
        from torch.utils.data import TensorDataset, DataLoader
        
        # Avoid MacOS PyTorch DataLoader deadlocks
        train_loader = DataLoader(
            TensorDataset(dataset["src_ids"], dataset["trg_ids"]), 
            batch_size=16, 
            shuffle=True, 
            num_workers=0
        )
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        with mlflow.start_run():
            mlflow.log_param("dataset", "OpenHust/vietnamese-summarization")
            mlflow.log_param("epochs", 3)
            mlflow.log_param("learning_rate", 0.001)
            
            from src.evaluation.rouge import SummarizationEvaluator
            from src.summarization.beam_search import decode_beam_search
            evaluator = SummarizationEvaluator()
            
            logger.info("Starting Actual Training Loop over Real Dataset...")
            for epoch in range(3):
                model.train()
                epoch_loss = 0
                for batch_idx, (src, trg) in enumerate(train_loader):
                    src = src.to(device)
                    trg = trg.to(device)
                    
                    optimizer.zero_grad()
                    output = model(src, trg)
                    
                    output_dim = output.shape[-1]
                    output = output[:, 1:].reshape(-1, output_dim)
                    trg = trg[:, 1:].reshape(-1)
                    
                    loss = criterion(output, trg)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                avg_loss = epoch_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                logger.info(f"Epoch {epoch + 1}/3: Avg Loss: {avg_loss:.4f}")
                
                # --- Validation Loop with BLEU & ROUGE ---
                model.eval()
                eval_loader = DataLoader(TensorDataset(src_eval, trg_eval), batch_size=1, shuffle=False)
                predictions = []
                references = []
                
                with torch.no_grad():
                    for src, trg in eval_loader:
                        src = src.to(device)
                        encoder_outputs, hidden, cell = model.encoder(src)
                        
                        sos_id = tokenizer.cls_token_id or tokenizer.bos_token_id or 0
                        eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id or 2
                        
                        # Use our updated length-controlled beam search
                        decoded_ids = decode_beam_search(model.decoder, encoder_outputs, hidden, cell, sos_id, eos_id, max_len=64, beam_width=3, device=device, min_len=10)
                        
                        pred_str = tokenizer.decode(decoded_ids, skip_special_tokens=True)
                        ref_str = tokenizer.decode(trg[0], skip_special_tokens=True)
                        
                        predictions.append(pred_str)
                        references.append(ref_str)
                
                scores = evaluator.compute_scores(predictions, references)
                if scores:
                    mlflow.log_metric("val_rougeL", scores.get("rougeL", 0), step=epoch)
                    mlflow.log_metric("val_bleu", scores.get("sacrebleu", 0), step=epoch)
                    logger.info(f"Validation Scores - ROUGE-L: {scores.get('rougeL', 0):.4f} | BLEU: {scores.get('sacrebleu', 0):.4f}")
                
            # Save the trained model to disk
            os.makedirs("models/summarizer", exist_ok=True)
            torch.save(model.state_dict(), "models/summarizer/summarizer.pt")
            logger.info("Saved summarization model to models/summarizer/summarizer.pt")

            # Register to MLflow Model Registry
            logger.info("Registering SummarizationModel to MLflow Model Registry...")
            import mlflow.pytorch
            info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="SummarizationModel",
            )
            logger.info(f"Registered SummarizationModel: {info.model_uri}")

        # Transition to Production
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        try:
            versions = client.get_latest_versions("SummarizationModel")
            if versions:
                client.transition_model_version_stage(
                    name="SummarizationModel",
                    version=versions[-1].version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(f"SummarizationModel v{versions[-1].version} → Production")
        except Exception as e:
            logger.warning(f"Could not transition model stage: {e}")

        logger.info("Summarization model training complete on real dataset.")
        
    except Exception as e:
        logger.error(f"Failed to load real dataset: {e}")
        import traceback; traceback.print_exc()

