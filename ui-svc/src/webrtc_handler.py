import logging

import cv2
import numpy as np

from grpc_client import GrpcClient

logger = logging.getLogger("ui-svc.webrtc")


class SignLanguageProcessor:
    def __init__(self, client: GrpcClient):
        self.client = client
        self.last_label = ""
        self.last_conf = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = encoded.tobytes()
        h, w = img.shape[:2]

        detections = self.client.detect(image_bytes, w, h)
        if detections is None:
            return img

        for d in detections:
            x_min, y_min, x_max, y_max = d["x_min"], d["y_min"], d["x_max"], d["y_max"]
            label, conf = d["label"], d["confidence"]

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x_min, y_min - th - 6), (x_min + tw + 6, y_min), (0, 255, 0), -1)
            cv2.putText(img, text, (x_min + 3, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            self.last_label = label
            self.last_conf = conf

        return img
