import logging
import os
from pathlib import Path

import mlflow

logger = logging.getLogger("mlflow_setup")

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file:///" + str(Path(__file__).resolve().parent.parent / "mlruns").replace("\\", "/"),
)
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sign-language-yolo11n")
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


def register_best_model(run_id: str) -> None:
    best_pt = Path("runs") / run_id / "weights" / "best.pt"
    if not best_pt.exists():
        logger.warning("Best model not found at %s", best_pt)
        return
    mlflow.log_artifact(str(best_pt))
    mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    logger.info("Model '%s' registered in MLflow Registry", MODEL_NAME)
