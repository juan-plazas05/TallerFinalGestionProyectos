FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY ui-svc/pyproject.toml .
RUN uv lock && uv sync --frozen --no-dev

COPY proto/ ./proto/
RUN mkdir -p src && \
    uv run python -m grpc_tools.protoc \
    -I ./proto \
    --python_out=./src \
    --grpc_python_out=./src \
    ./proto/vision.proto

COPY ui-svc/src/ ./src/

RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["uv", "run", "streamlit", "run", "src/app.py"]
