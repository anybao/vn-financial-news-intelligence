import mlflow
from mlflow.tracking import MlflowClient
import os
import logging

logger = logging.getLogger(__name__)

def setup_mlflow():
    """Initialize MLflow tracking URI."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri

def log_experiment_parameters(params: dict):
    """Log parameters to a current active MLflow run."""
    mlflow.log_params(params)

def log_experiment_metrics(metrics: dict, step: int = None):
    """Log metrics to a current active MLflow run."""
    mlflow.log_metrics(metrics, step=step)

def log_model_artifact(model_path: str, artifact_path: str):
    """Log an artifact (such as a model file) to MLflow."""
    mlflow.log_artifact(model_path, artifact_path)

def register_model_to_registry(model_uri: str, registry_name: str):
    """
    Register a logged model artifact to the MLflow Model Registry.
    
    Args:
        model_uri: The MLflow artifact URI (e.g. 'runs:/<run_id>/model')
        registry_name: Name to register under (e.g. 'SentimentModel')
    
    Returns:
        The ModelVersion object.
    """
    result = mlflow.register_model(model_uri, registry_name)
    logger.info(f"Registered model '{registry_name}' version {result.version}")
    return result

def transition_model_stage(registry_name: str, version: str, stage: str = "Production"):
    """
    Transition a registered model version to a given stage.
    
    Args:
        registry_name: Name of the registered model
        version: Version number to transition
        stage: Target stage ('Staging', 'Production', 'Archived')
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registry_name,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )
    logger.info(f"Transitioned '{registry_name}' v{version} to '{stage}'")
