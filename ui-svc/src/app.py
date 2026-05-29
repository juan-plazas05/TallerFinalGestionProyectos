import logging
import os

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ui-svc")

GRPC_HOST = os.getenv("INFERENCE_GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("INFERENCE_GRPC_PORT", "50051"))

st.set_page_config(page_title="Sign Language Recognition", layout="wide")

st.title("Sign Language Recognition")
st.markdown("Reconocimiento de lenguaje de señas en tiempo real")

st.info(f"Conectando a inference-svc en {GRPC_HOST}:{GRPC_PORT}")

st.subheader("Camera")
st.caption("La integración con cámara estará disponible en el próximo sprint.")

st.markdown("---")
st.subheader("Image Upload")
uploaded_file = st.file_uploader(
    "Sube una imagen para analizar",
    type=["jpg", "jpeg", "png"],
)
if uploaded_file:
    st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
    st.success("Imagen recibida. La inferencia se implementará en el siguiente sprint.")
