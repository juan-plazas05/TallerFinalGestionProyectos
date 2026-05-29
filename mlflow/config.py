import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sign-language-yolov7")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "sign-language-detector")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
