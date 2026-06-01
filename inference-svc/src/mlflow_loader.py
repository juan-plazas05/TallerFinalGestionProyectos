import logging
import os
import tempfile
import warnings
from pathlib import Path

import mlflow
from ultralytics import YOLO

logger = logging.getLogger("inference-svc.mlflow_loader")

MODEL_DIR = Path(tempfile.gettempdir()) / "sign-language-models"


def _get_client() -> mlflow.tracking.MlflowClient:
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(
        Path(__file__).resolve().parent.parent.parent / "training" / "mlflow.db"
    ).replace("\\", "/"))
    mlflow.set_tracking_uri(uri)
    return mlflow.tracking.MlflowClient()


def load_model(stage: str | None = None) -> YOLO:
    model_name = os.getenv("MLFLOW_MODEL_NAME", "sign-language-detector")
    stage = stage or os.getenv("MODEL_STAGE", "Production")

    logger.info("Loading model '%s' stage '%s' from MLflow", model_name, stage)
    client = _get_client()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get_latest_versions.*")
        versions = client.get_latest_versions(model_name, stages=[stage])

    if not versions:
        raise RuntimeError(
            f"No version of '{model_name}' found in stage '{stage}'"
        )

    version = versions[0]
    run_id = version.run_id
    logger.info("Found version %s (run %s)", version.version, run_id)

    local_dir = MODEL_DIR / run_id
    if not local_dir.exists():
        local_dir.mkdir(parents=True, exist_ok=True)
        client.download_artifacts(run_id, "", str(local_dir))
        logger.info("Artifacts downloaded to %s", local_dir)

    onnx_path = local_dir / "best.onnx"
    if not onnx_path.exists():
        pt_path = local_dir / "best.pt"
        if pt_path.exists():
            logger.info("ONNX not found, loading .pt instead: %s", pt_path)
            return YOLO(str(pt_path), task="detect")
        raise FileNotFoundError(f"Neither best.onnx nor best.pt found in {local_dir}")

    logger.info("Loading ONNX model: %s", onnx_path)
    return YOLO(str(onnx_path), task="detect")
