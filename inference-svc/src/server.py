import logging
import os
import time
from concurrent import futures

import grpc

import vision_pb2
import vision_pb2_grpc
import mlflow_loader
import process
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference-svc")

CONFIDENCE = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IOU = float(os.getenv("IOU_THRESHOLD", "0.45"))

class SignLanguageRecognizerServicer(vision_pb2_grpc.SignLanguageRecognizerServicer):
    def __init__(self, model):
        self.model = model

    def RecognizeStream(self, request_iterator, context):
        logger.info(f"Client connected: {request_iterator}")
        for request in request_iterator:
            logger.info(f"Processing frame %d", request.frame_id)
            t0 = time.perf_counter()

            try:
                detections = process.detect(
                    self.model,
                    request.image_data,
                    request.width,
                    request.height,
                    conf_threshold=CONFIDENCE,
                    iou_threshold=IOU,
                )

                annotated = process.draw_detections(
                    request.image_data, request.width, request.height, detections
                )

            except Exception as exc:
                logger.exception("Error processing frame %d", request.frame_id)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Error processing frame {request.frame_id}: {exc}")
                return

            inference_ms = (time.perf_counter() - t0) * 1000

            proto_dets = [
                vision_pb2.Detection(
                    label=d["label"],
                    confidence=d["confidence"],
                    x_min=d["x_min"],
                    y_min=d["y_min"],
                    x_max=d["x_max"],
                    y_max=d["y_max"],
                )
                for d in detections
            ]

            yield vision_pb2.FrameResponse(
                detections=proto_dets,
                annotated_image=annotated,
                frame_id=request.frame_id,
                inference_time_ms=round(inference_ms, 1),
            )

            logger.debug(
                "Frame %d: %d detections in %.1f ms",
                request.frame_id,
                len(detections),
                inference_ms,
            )


        logger.info("Client disconnected")


def serve() -> None:
    port = int(os.getenv("GRPC_PORT", "50051"))
    model = mlflow_loader.load_model()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    vision_pb2_grpc.add_SignLanguageRecognizerServicer_to_server(
        SignLanguageRecognizerServicer(model), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server ready on port %d. Press Ctrl+C to stop.", port)

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        server.stop(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
