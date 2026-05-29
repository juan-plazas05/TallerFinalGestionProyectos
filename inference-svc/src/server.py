import logging
import os
from concurrent import futures

import grpc

import vision_pb2
import vision_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference-svc")


class SignLanguageRecognizerServicer(vision_pb2_grpc.SignLanguageRecognizerServicer):
    def RecognizeStream(self, request_iterator, context):
        logger.info("Client connected, waiting for frames...")
        for request in request_iterator:
            logger.info(
                "Received frame %d (%dx%d, %d bytes)",
                request.frame_id,
                request.width,
                request.height,
                len(request.image_data),
            )
            yield vision_pb2.FrameResponse(
                detections=[],
                annotated_image=request.image_data,
                frame_id=request.frame_id,
                inference_time_ms=0.0,
            )


def serve() -> None:
    port = int(os.getenv("GRPC_PORT", "50051"))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    vision_pb2_grpc.add_SignLanguageRecognizerServicer_to_server(
        SignLanguageRecognizerServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server ready on port %d", port)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
