import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def evaluate_metrics(eval_pred):
    """Compute accuracy and f1 for sentiment classification."""
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1).numpy()
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}

def train_sentiment_model(model_name: str = "vinai/phobert-base", dataset_path: str = "data/processed/sentiment.csv", output_dir: str = "models/sentiment"):
    """Fine-tunes a PhoBERT model for sentiment analysis."""
    logger.info("Starting sentiment model training...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3) # Positive, Negative, Neutral
    
    # Example logic using datasets library (assuming dataset has 'text' and 'label' columns)
    try:
        dataset = load_dataset('csv', data_files=dataset_path)
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", max_length=128, truncation=True)
            
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(min(1000, len(tokenized_datasets['train'])))) # Select subset for quick testing
        eval_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(min(200, len(tokenized_datasets['train']))))
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}. Using dummy data instead. ERROR: {e}")
        # dummy dataset definition skipped
        return

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        report_to="mlflow" # Directly logs to MLflow
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=evaluate_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete and model saved.")

if __name__ == "__main__":
    train_sentiment_model()
