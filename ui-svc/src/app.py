import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from grpc_client import GrpcClient
from webrtc_handler import SignLanguageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ui-svc")

GRPC_HOST = os.getenv("INFERENCE_GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("INFERENCE_GRPC_PORT", "50051"))
EDA_DIR = Path(__file__).resolve().parents[2] / "data" / "eda_output"

st.set_page_config(page_title="Sign Language Recognition", layout="wide")

if "grpc_client" not in st.session_state:
    st.session_state.grpc_client = GrpcClient(GRPC_HOST, GRPC_PORT)
client: GrpcClient = st.session_state.grpc_client

if "connected" not in st.session_state:
    st.session_state.connected = False

with st.sidebar:
    st.markdown("## Sign Language Recognition")
    st.markdown("### Conexión")

    ok = st.session_state.connected
    if ok:
        st.info(f"🟢 Conectado a inference-svc en {GRPC_HOST}:{GRPC_PORT}")
    else:
        st.warning("🔴 Inference service no disponible")

    if st.button("Verificar conexión", type="primary"):
        with st.spinner("Verificando..."):
            st.session_state.connected = client.health_check()
        st.rerun()

    st.markdown("### Modelo")
    st.markdown("**Modelo:** sign-language-detector (Production)")
    st.markdown("**Clases:** A-Z (26 letras)")
    st.markdown("**Backend:** YOLO11n (ONNX)")

    st.markdown("---")
    st.caption("vc-lenguaje-senas v1.0")

st.title("🤟 Sign Language Recognition")

tab_det, tab_eda = st.tabs(["Detección en Vivo", "Dataset EDA"])

with tab_det:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Cámara en Vivo")
        from streamlit_webrtc import webrtc_streamer, WebRtcMode

        ctx = webrtc_streamer(
            key="sign-language",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: SignLanguageProcessor(client),
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("---")
        st.subheader("📷 Subir Imagen")
        uploaded_file = st.file_uploader(
            "Selecciona una imagen para analizar",
            type=["jpg", "jpeg", "png"],
            key="image_upload",
        )

        if uploaded_file:
            arr = cv2.imdecode(
                np.frombuffer(uploaded_file.read(), np.uint8),
                cv2.IMREAD_COLOR,
            )
            h, w = arr.shape[:2]
            _, encoded = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_bytes = encoded.tobytes()

            col_orig, col_result = st.columns(2)
            with col_orig:
                st.image(arr, channels="BGR", caption="Original", width="stretch")

            with col_result:
                with st.spinner("Analizando..."):
                    detections = client.detect(image_bytes, w, h)

                if detections:
                    for d in detections:
                        cv2.rectangle(
                            arr,
                            (d["x_min"], d["y_min"]),
                            (d["x_max"], d["y_max"]),
                            (0, 255, 0), 2,
                        )
                        label, conf = d["label"], d["confidence"]
                        text = f"{label} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            arr,
                            (d["x_min"], d["y_min"] - th - 6),
                            (d["x_min"] + tw + 6, d["y_min"]),
                            (0, 255, 0), -1,
                        )
                        cv2.putText(
                            arr, text,
                            (d["x_min"] + 3, d["y_min"] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                        )

                    st.image(
                        arr, channels="BGR",
                        caption=f"{len(detections)} detecciones",
                        width="stretch",
                    )
                    for d in detections:
                        st.markdown(f"- **{d['label']}** — confianza {d['confidence']:.2%}")
                else:
                    st.error(
                        "No se obtuvieron detecciones. "
                        "Verifica que inference-svc esté corriendo "
                        f"en {GRPC_HOST}:{GRPC_PORT}"
                    )

    with col_right:
        st.subheader("📊 Estadísticas")
        st.metric("Última letra", st.session_state.get("last_label", "—"))
        st.metric("Confianza", st.session_state.get("last_conf", "—"))
        st.metric("Estado", "🟢 Stream activo" if (ctx and ctx.state.playing) else "⏸️ Cámara detenida")
        st.caption("Activa la cámara arriba o sube una imagen para ver detecciones.")

with tab_eda:
    st.header("📊 Análisis Exploratorio del Dataset (EDA)")
    st.markdown(
        "Reporte generado durante el análisis del dataset de lenguaje de señas "
        "(A-Z). El dataset contiene **1728 imágenes** etiquetadas en formato YOLO."
    )

    report_path = EDA_DIR / "report.md"
    if not report_path.exists():
        st.warning(f"Reporte no encontrado en {report_path}")
        st.stop()

    images_in_report = {
        "class_distribution.png": "Distribución de Clases",
        "per_class_bbox.png": "Bounding Boxes por Clase",
        "bbox_analysis.png": "Análisis General de Bounding Boxes",
    }

    for png, title in images_in_report.items():
        img_path = EDA_DIR / png
        if img_path.exists():
            st.subheader(title)
            st.image(str(img_path), width="stretch")

    st.subheader("Resumen del Reporte")
    with open(report_path, encoding="utf-8") as f:
        content = f.read()
    skip_patterns = [
        "# Dataset Report", "## ", "![",
        "class_distribution.png", "per_class_bbox.png", "bbox_analysis.png",
    ]
    summary_lines = [
        l for l in content.split("\n")
        if l.strip() and not any(p in l for p in skip_patterns)
    ]
    st.markdown("\n".join(summary_lines))
