import logging
from dataclasses import dataclass
from typing import Any

import grpc

import vision_pb2
import vision_pb2_grpc

logger = logging.getLogger("ui-svc.grpc")


@dataclass(frozen=True)
class DetectionResult:
    detections: list[dict[str, Any]]
    annotated_image: bytes
    frame_id: int
    inference_time_ms: float


class GrpcClient:
    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self.target = f"{host}:{port}"
        self.timeout = timeout
        self._channel: grpc.Channel | None = None
        self._stub: vision_pb2_grpc.SignLanguageRecognizerStub | None = None
        self._frame_id = 0

    def _get_stub(self):
        if self._stub is None:
            logger.info("Creating channel to %s", self.target)
            self._channel = grpc.insecure_channel(
                self.target,
                options=[
                    ("grpc.connect_timeout_ms", int(self.timeout * 1000)),
                    ("grpc.keepalive_time_ms", 30000),
                ],
            )
            self._stub = vision_pb2_grpc.SignLanguageRecognizerStub(self._channel)
        return self._stub

    def _reset(self):
        self._channel = None
        self._stub = None

    def detect(self, image_bytes: bytes, width: int, height: int) -> DetectionResult | None:
        try:
            stub = self._get_stub()
        except Exception as e:
            logger.warning("Failed to create stub: %s", e)
            self._reset()
            return None

        self._frame_id += 1
        req = vision_pb2.FrameRequest(
            image_data=image_bytes,
            width=width,
            height=height,
            frame_id=self._frame_id,
        )

        try:
            responses = stub.RecognizeStream(iter([req]), timeout=self.timeout)
            for resp in responses:
                dets = [
                    {
                        "label": d.label,
                        "confidence": d.confidence,
                        "x_min": d.x_min,
                        "y_min": d.y_min,
                        "x_max": d.x_max,
                        "y_max": d.y_max,
                    }
                    for d in resp.detections
                ]
                logger.info(
                    "Frame %d: %d detections in %.1f ms",
                    resp.frame_id, len(dets), resp.inference_time_ms,
                )
                return DetectionResult(
                    detections=dets,
                    annotated_image=bytes(resp.annotated_image),
                    frame_id=int(resp.frame_id),
                    inference_time_ms=float(resp.inference_time_ms),
                )
        except grpc.RpcError as e:
            logger.warning("gRPC error: %s - %s", e.code(), e.details())
            self._reset()
            return None

        return None

    def health_check(self) -> bool:
        try:
            ch = grpc.insecure_channel(self.target)
            grpc.channel_ready_future(ch).result(timeout=2.0)
            ch.close()
            return True
        except Exception:
            return False
