#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import mlflow
from ultralytics import YOLO

import mlflow_setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

TRAINING_DIR = Path(__file__).resolve().parent
DATASET_YAML = TRAINING_DIR.parent / "data" / "dataset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on sign language dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--lr", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Training config: %s", vars(args))
    logger.info("Dataset: %s", DATASET_YAML)

    mlflow_setup.setup()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow_setup.log_params(vars(args))

        model = YOLO(args.model)
        model.train(
            data=str(DATASET_YAML),
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=args.device,
            lr0=args.lr,
            project="runs",
            name=run_id,
            exist_ok=True,
            verbose=True,
        )

        val_results = model.val()
        mlflow_setup.log_metrics(val_results.results_dict)
        mlflow_setup.register_best_model(run_id)

    logger.info("Training complete. Run ID: %s", run_id)


if __name__ == "__main__":
    main()
