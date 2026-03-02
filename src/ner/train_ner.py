import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import logging
import os

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "NER_Model"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_ner_model(model_name: str = "vinai/phobert-base", dataset_path: str = "data/processed/ner.json", output_dir: str = "models/ner"):
    """Fine-tunes a PhoBERT model for Token Classification (NER)."""
    logger.info("Starting NER model training...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Define labels arbitrarily high to handle varying NER tag formats across Vietnam datasets
    num_labels = 15 
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    
    try:
        logger.info("Loading `truongpdd/NER-covid-vietnamese-word` Dataset from HuggingFace...")
        dataset = load_dataset("truongpdd/NER-covid-vietnamese-word")
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                label_ids = [-100] * 128
                for j, tag in enumerate(label):
                    if j + 1 < 128:  # +1 for start token
                        label_ids[j + 1] = tag
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
            
        tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
        # remove original columns so PyTorch can batch
        tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "ner_tags"])
        
        train_dataset = tokenized_datasets['train'].select(range(min(500, len(tokenized_datasets['train']))))
        eval_dataset = tokenized_datasets['validation'].select(range(min(50, len(tokenized_datasets['validation']))))
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback; traceback.print_exc()
        return

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="mlflow"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete and NER model saved.")

if __name__ == "__main__":
    train_ner_model()
