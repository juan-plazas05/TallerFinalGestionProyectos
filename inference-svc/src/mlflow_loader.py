import logging
import os
import tempfile
import warnings
from pathlib import Path

import mlflow
from ultralytics import YOLO

logger = logging.getLogger("inference-svc.mlflow_loader")

MODEL_DIR = Path(tempfile.gettempdir()) / "sign-language-models"


def _load_local_model(path: str | os.PathLike[str]) -> YOLO:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    logger.info("Loading local model: %s", model_path)
    return YOLO(str(model_path), task="detect")


def _get_client() -> mlflow.tracking.MlflowClient:
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(
        Path(__file__).resolve().parent.parent.parent / "training" / "mlflow.db"
    ).replace("\\", "/"))
    mlflow.set_tracking_uri(uri)
    return mlflow.tracking.MlflowClient()


def load_model(stage: str | None = None) -> YOLO:
    local_path = os.getenv("MODEL_LOCAL_PATH")
    if local_path:
        model_path = Path(local_path)
        if model_path.exists():
            return _load_local_model(model_path)
        logger.warning(
            "MODEL_LOCAL_PATH is set but file does not exist: %s. Falling back to MLflow.",
            model_path,
        )

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
