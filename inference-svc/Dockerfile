FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY inference-svc/pyproject.toml .
RUN uv lock && uv sync --frozen --no-dev

COPY inference-svc/src/ ./src/
COPY proto/ ./proto/
RUN uv run python -m grpc_tools.protoc \
    -I ./proto \
    --python_out=./src \
    --grpc_python_out=./src \
    ./proto/vision.proto

RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 50051

CMD ["uv", "run", "python", "src/server.py"]
