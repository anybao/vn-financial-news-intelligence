"""
One-time script to register existing trained models into the MLflow Model Registry.

Usage:
    PYTHONPATH=. venv/bin/python scripts/register_models.py
"""

import os
import sys
import logging
import mlflow
import mlflow.pytorch
import mlflow.transformers
import torch
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

# --- Model definitions ---
MODELS_TO_REGISTER = [
    {
        "name": "SentimentModel",
        "local_path": "models/sentiment",
        "experiment": "Sentiment_Analysis",
        "flavor": "transformers",
        "description": "PhoBERT fine-tuned for Vietnamese financial sentiment analysis (3-class: Negative, Neutral, Positive)",
    },
    {
        "name": "NERModel",
        "local_path": "models/ner",
        "experiment": "NER_Model",
        "flavor": "transformers",
        "description": "PhoBERT fine-tuned for Vietnamese stock ticker NER (Token Classification)",
    },
    {
        "name": "SummarizationModel",
        "local_path": "models/summarizer",
        "experiment": "Summarization_Model",
        "flavor": "pytorch",
        "description": "BiLSTM Seq2Seq with Attention for Vietnamese financial news summarization",
    },
]


def register_transformers_model(model_info: dict):
    """Register a HuggingFace Transformers model (sentiment or NER)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

    local_path = model_info["local_path"]
    registry_name = model_info["name"]

    if not os.path.isdir(local_path):
        logger.warning(f"Model directory '{local_path}' not found. Skipping {registry_name}.")
        return None

    # Check if config.json exists (indicator of a valid saved model)
    if not os.path.isfile(os.path.join(local_path, "config.json")):
        logger.warning(f"No config.json found in '{local_path}'. Skipping {registry_name}.")
        return None

    logger.info(f"Loading tokenizer + model from '{local_path}' for {registry_name}...")

    tokenizer = AutoTokenizer.from_pretrained(local_path)

    # Determine model class based on registry name
    if "Sentiment" in registry_name:
        model = AutoModelForSequenceClassification.from_pretrained(local_path)
        task = "text-classification"
    else:
        model = AutoModelForTokenClassification.from_pretrained(local_path)
        task = "token-classification"

    # Create a transformers pipeline for MLflow
    from transformers import pipeline
    pipe = pipeline(task, model=model, tokenizer=tokenizer)

    mlflow.set_experiment(model_info["experiment"])

    with mlflow.start_run(run_name=f"register_{registry_name}") as run:
        mlflow.log_param("source", "local_registration")
        mlflow.log_param("model_path", local_path)

        # Log the transformers pipeline as an MLflow model
        # Provide explicit pip_requirements to avoid auto-detecting tensorflow
        model_info_result = mlflow.transformers.log_model(
            transformers_model=pipe,
            artifact_path="model",
            registered_model_name=registry_name,
            pip_requirements=["transformers", "torch", "tokenizers"],
        )
        logger.info(f"Logged {registry_name} as MLflow artifact: {model_info_result.model_uri}")

    # Transition to Production
    client = MlflowClient()
    latest_versions = client.get_latest_versions(registry_name)
    if latest_versions:
        latest = latest_versions[-1]
        client.transition_model_version_stage(
            name=registry_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"✅ {registry_name} v{latest.version} → Production")

    return registry_name


def register_pytorch_model(model_info: dict):
    """Register a PyTorch model (summarization)."""
    local_path = model_info["local_path"]
    registry_name = model_info["name"]
    model_file = os.path.join(local_path, "summarizer.pt")

    if not os.path.isfile(model_file):
        logger.warning(f"Model file '{model_file}' not found. Skipping {registry_name}.")
        return None

    logger.info(f"Loading PyTorch state dict from '{model_file}' for {registry_name}...")

    # Reconstruct the model architecture
    from src.summarization.encoder import EncoderBiLSTM
    from src.summarization.decoder import AttnDecoderLSTM
    from src.summarization.train import Seq2SeqSummarizer

    INPUT_DIM = 50000
    OUTPUT_DIM = 50000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 1

    device = torch.device("cpu")
    enc = EncoderBiLSTM(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS)
    dec = AttnDecoderLSTM(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)
    model = Seq2SeqSummarizer(enc, dec, device)

    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        logger.info("Successfully loaded model weights.")
    except Exception as e:
        logger.warning(f"Could not load weights: {e}. Registering with initialized weights.")

    mlflow.set_experiment(model_info["experiment"])

    with mlflow.start_run(run_name=f"register_{registry_name}") as run:
        mlflow.log_param("source", "local_registration")
        mlflow.log_param("model_path", local_path)

        model_info_result = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=registry_name,
        )
        logger.info(f"Logged {registry_name} as MLflow artifact: {model_info_result.model_uri}")

    # Transition to Production
    client = MlflowClient()
    latest_versions = client.get_latest_versions(registry_name)
    if latest_versions:
        latest = latest_versions[-1]
        client.transition_model_version_stage(
            name=registry_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"✅ {registry_name} v{latest.version} → Production")

    return registry_name


def main():
    logger.info("=" * 60)
    logger.info("  MLflow Model Registration Script")
    logger.info("=" * 60)

    mlflow.set_tracking_uri(TRACKING_URI)
    logger.info(f"MLflow Tracking URI: {TRACKING_URI}")

    registered = []

    for model_info in MODELS_TO_REGISTER:
        logger.info(f"\n--- Registering: {model_info['name']} ---")
        try:
            if model_info["flavor"] == "transformers":
                result = register_transformers_model(model_info)
            elif model_info["flavor"] == "pytorch":
                result = register_pytorch_model(model_info)
            else:
                logger.warning(f"Unknown flavor '{model_info['flavor']}' — skipping.")
                result = None

            if result:
                registered.append(result)
        except Exception as e:
            logger.error(f"Failed to register {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 60)
    logger.info(f"  Registration Complete: {len(registered)}/{len(MODELS_TO_REGISTER)} models registered")
    for name in registered:
        logger.info(f"    ✅ {name}")
    logger.info("=" * 60)
    logger.info("Open MLflow UI → Models tab to verify: http://localhost:5001/#/models")


if __name__ == "__main__":
    main()
