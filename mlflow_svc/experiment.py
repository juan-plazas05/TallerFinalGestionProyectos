import logging

import mlflow

from mlflow.config import (
    EXPERIMENT_NAME,
    MODEL_NAME,
    MODEL_STAGE,
    TRACKING_URI,
)

logger = logging.getLogger("mlflow-setup")


def setup_experiment() -> mlflow.entities.Experiment:
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(
        "Experiment '%s' ready (ID: %s)", EXPERIMENT_NAME, experiment.experiment_id
    )
    return experiment


def load_production_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)
