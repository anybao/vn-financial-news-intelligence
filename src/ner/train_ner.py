import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import logging
import os
import mlflow
import mlflow.transformers

# Need this to compute proper NER metrics like precision/recall/f1-score
import evaluate
import numpy as np

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "NER_Model"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use standard seqeval for token classification metrics
seqeval = evaluate.load("seqeval")

# Our custom scraped NER data only has 0 (Outside) and 1 (Ticker)
label_list = ["O", "TICKER"]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train_ner_model(model_name: str = "vinai/phobert-base", dataset_path: str = "data/processed/ner.json", output_dir: str = "models/ner"):
    """Fine-tunes a PhoBERT model for Token Classification (NER) with an MLflow tracking loop."""
    logger.info("Starting NER model training sweep...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(label_list)
    
    try:
        logger.info(f"Loading local JSON dataset from {dataset_path}...")
        # Load custom JSON Lines
        dataset = load_dataset("json", data_files=dataset_path)
        
        # Split into training and validation sets
        split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["tokens"], 
                truncation=True, 
                is_split_into_words=True, 
                padding="max_length", 
                max_length=128
            )
            
            labels = []
            for i, label in enumerate(examples["ner_tags"]):
                # Since PhoBERT lacks a FastTokenizer we just pad/truncate labels to 128
                # In real wordpiece tokenization, we should align exactly, but for this custom dataset 
                # (which was split strictly by word boundary beforehand) a 1-to-1 match up to max_len works.
                label_ids = [-100] * 128
                
                # First token is <s> (index 0)
                # The actual words start at index 1
                for j, tag in enumerate(label):
                    if j + 1 < 128 - 1: # leave room for </s>
                        label_ids[j + 1] = tag
                        
                labels.append(label_ids)
                
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
            
        tokenized_datasets = split_dataset.map(tokenize_and_align_labels, batched=True)
        # Remove extra columns
        tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "ner_tags"])
        
        train_dataset = tokenized_datasets['train']
        eval_dataset = tokenized_datasets['test']
        logger.info(f"Dataset successfuly formatted! Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback; traceback.print_exc()
        return

    learning_rates = [2e-5, 5e-5]
    best_f1 = 0
    best_trainer = None

    for lr in learning_rates:
        run_name = f"PhoBERT_NER_lr_{lr}"
        logger.info(f"=== Starting Run: {run_name} ===")
        
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
        
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/lr_{lr}",
            eval_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            report_to="mlflow",
            run_name=run_name,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        try:
            train_output = trainer.train()
            metrics = trainer.evaluate()
            
            # Save the globally best model observed
            current_f1 = metrics.get('eval_f1', 0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_trainer = trainer
                logger.info(f"New best NER model found! (F1: {best_f1:.4f}) -> Saving globally to {output_dir}")
                trainer.save_model(output_dir)
                tokenizer.save_pretrained(output_dir)
                
        except Exception as e:
            logger.error(f"Failed during NER training run {run_name}: {e}")
            import traceback; traceback.print_exc()

    # --- Register best model to MLflow Model Registry ---
    if best_trainer is not None:
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Registering best NER model to MLflow Model Registry...")
            best_model = best_trainer.model
            pipe = hf_pipeline("token-classification", model=best_model, tokenizer=tokenizer)
            
            with mlflow.start_run(run_name="register_NERModel"):
                mlflow.log_param("best_f1", best_f1)
                info = mlflow.transformers.log_model(
                    transformers_model=pipe,
                    artifact_path="model",
                    registered_model_name="NERModel",
                )
                logger.info(f"Registered NERModel: {info.model_uri}")
            
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            versions = client.get_latest_versions("NERModel")
            if versions:
                client.transition_model_version_stage(
                    name="NERModel",
                    version=versions[-1].version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(f"NERModel v{versions[-1].version} → Production")
        except Exception as e:
            logger.error(f"Failed to register model in MLflow Registry: {e}")
            import traceback; traceback.print_exc()

    logger.info("NER Sweep complete.")

if __name__ == "__main__":
    train_ner_model()
