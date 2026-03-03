import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import logging
import os
import mlflow
import mlflow.transformers

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "Sentiment_Analysis"

logging.basicConfig(level=logging.INFO)
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
    """Fine-tunes a PhoBERT model for sentiment analysis with MLflow tracking multiple hyperparameter runs."""
    logger.info("Starting sentiment model training sweep...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        logger.info(f"Loading local dataset from {dataset_path}...")
        # Load custom CSV
        dataset = load_dataset('csv', data_files=dataset_path)
        
        # We need an eval split to evaluate metrics properly
        split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
        
        def tokenize_function(examples):
            # PhoBERT tokenizer takes 'text' string
            return tokenizer(examples['text'], padding="max_length", max_length=128, truncation=True)
            
        tokenized_datasets = split_dataset.map(tokenize_function, batched=True)
        # Remove text so Trainer can form PyTorch batches seamlessly
        tokenized_datasets = tokenized_datasets.remove_columns(["text"])
        
        train_dataset = tokenized_datasets['train']
        eval_dataset = tokenized_datasets['test']
        logger.info(f"Dataset successfuly formatted! Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_path}. ERROR: {e}")
        return

    # Hyperparameter sweep settings
    learning_rates = [2e-5, 5e-5]
    best_f1 = 0
    best_model_dir = output_dir
    best_trainer = None

    for lr in learning_rates:
        run_name = f"PhoBERT_lr_{lr}"
        logger.info(f"=== Starting Run: {run_name} ===")
        
        # Instantiate a fresh model for each run
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/lr_{lr}",
            eval_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            push_to_hub=False,
            report_to="mlflow", # Hooks directly into 'Sentiment_Analysis' experiment
            run_name=run_name,  # Separates it on MLflow UI
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=evaluate_metrics,
        )

        # Triggers training and MLflow callback logging
        try:
            train_output = trainer.train()
            metrics = trainer.evaluate()
            
            # Check if this is the best model so far
            current_f1 = metrics.get('eval_f1', 0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_trainer = trainer
                logger.info(f"New best model found! (F1: {best_f1:.4f}) -> Saving globally to {output_dir}")
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)
        except Exception as e:
            logger.error(f"Failed during training run {run_name}: {e}")
            import traceback; traceback.print_exc()

    # --- Register best model to MLflow Model Registry ---
    if best_trainer is not None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Registering best Sentiment model to MLflow Model Registry...")
            best_model = best_trainer.model
            pipe = hf_pipeline("text-classification", model=best_model, tokenizer=tokenizer)
            
            with mlflow.start_run(run_name="register_SentimentModel"):
                mlflow.log_param("best_f1", best_f1)
                info = mlflow.transformers.log_model(
                    transformers_model=pipe,
                    artifact_path="model",
                    registered_model_name="SentimentModel",
                )
                logger.info(f"Registered SentimentModel: {info.model_uri}")
            
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.get_latest_versions("SentimentModel")
            if versions:
                client.transition_model_version_stage(
                    name="SentimentModel",
                    version=versions[-1].version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(f"SentimentModel v{versions[-1].version} → Production")
        except Exception as e:
            logger.error(f"Failed to register model in MLflow Registry: {e}")
            import traceback; traceback.print_exc()
            
    logger.info("Sweep completed.")

if __name__ == "__main__":
    train_sentiment_model()
