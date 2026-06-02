import logging

import cv2
import numpy as np
from av import VideoFrame

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

        result = self.client.detect(image_bytes, w, h)
        if result is None:
            return VideoFrame.from_ndarray(img, format="bgr24")

        annotated = cv2.imdecode(
            np.frombuffer(result.annotated_image, np.uint8),
            cv2.IMREAD_COLOR,
        )
        if annotated is not None:
            img = annotated

        for d in result.detections:
            label, conf = d["label"], d["confidence"]

            self.last_label = label
            self.last_conf = conf

        return VideoFrame.from_ndarray(img, format="bgr24")
