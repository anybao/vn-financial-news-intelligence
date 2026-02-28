import mlflow
import os

def setup_mlflow():
    """Initialize MLflow tracking URI."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

def log_experiment_parameters(params: dict):
    """Log parameters to a current active MLflow run."""
    mlflow.log_params(params)

def log_experiment_metrics(metrics: dict, step: int = None):
    """Log metrics to a current active MLflow run."""
    mlflow.log_metrics(metrics, step=step)

def log_model_artifact(model_path: str, artifact_path: str):
    """Log an artifact (such as a model file) to MLflow."""
    mlflow.log_artifact(model_path, artifact_path)
