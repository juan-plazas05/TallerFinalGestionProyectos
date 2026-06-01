import logging
import os
from pathlib import Path

import mlflow

logger = logging.getLogger("mlflow_setup")

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(Path(__file__).resolve().parent / "mlflow.db").replace("\\", "/"),
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sign-language")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sign-language-detector")


def setup() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", TRACKING_URI)


def log_params(params: dict) -> None:
    mlflow.log_params(params)


def log_metrics(results_dict: dict) -> None:
    flat = {}
    for k, v in results_dict.items():
        if isinstance(v, (int, float)):
            clean = k.replace("/", "_").replace("(", "_").replace(")", "")
            flat[clean] = float(v)
    mlflow.log_metrics(flat)
    logger.info(
        "mAP50=%.4f, mAP50-95=%.4f",
        results_dict.get("metrics/mAP50(B)", 0),
        results_dict.get("metrics/mAP50-95(B)", 0),
    )


def register_model(run_id: str, best_pt: Path, best_onnx: Path) -> None:
    mlflow.log_artifact(str(best_pt))
    logger.info("Artifact logged: %s", best_pt)

    if best_onnx.exists():
        mlflow.log_artifact(str(best_onnx))
        logger.info("Artifact logged: %s", best_onnx)
    else:
        logger.warning("ONNX model not found at %s", best_onnx)

    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
        logger.info("Registered model '%s' created", MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        logger.info("Registered model '%s' already exists", MODEL_NAME)

    source = f"runs:/{run_id}/artifacts"
    version = client.create_model_version(MODEL_NAME, source, run_id)
    client.transition_model_version_stage(MODEL_NAME, version.version, "Production")
    logger.info(
        "Model '%s' version %s registered and promoted to Production",
        MODEL_NAME,
        version.version,
    )
