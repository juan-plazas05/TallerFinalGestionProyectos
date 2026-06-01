import logging

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("inference-svc.process")


def detect(
    model: YOLO,
    image_bytes: bytes,
    width: int,
    height: int,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
) -> list[dict]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")

    results = model.predict(
        img,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
            detections.append({
                "label": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            })

    return detections


def draw_detections(
    image_bytes: bytes, width: int, height: int, detections: list[dict]
) -> bytes:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    for det in detections:
        x_min = int(det["x_min"])
        y_min = int(det["y_min"])
        x_max = int(det["x_max"])
        y_max = int(det["y_max"])
        label = det["label"]
        conf = det["confidence"]

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x_min, y_min - th - 6), (x_min + tw + 4, y_min), (0, 255, 0), -1)
        cv2.putText(img, text, (x_min + 2, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes()
