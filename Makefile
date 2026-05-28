.PHONY: proto install lint test clean

PROTO_DIR = proto
PROTO_OUT = proto/gen

proto:
	@mkdir -p $(PROTO_OUT)
	uv run python -m grpc_tools.protoc \
		-I $(PROTO_DIR) \
		--python_out=$(PROTO_OUT) \
		--grpc_python_out=$(PROTO_OUT) \
		$(PROTO_DIR)/vision.proto
	@touch $(PROTO_OUT)/__init__.py
	@echo "Proto files generated in $(PROTO_OUT)/"

install:
	cd inference-svc && uv sync
	cd ui-svc && uv sync

lint:
	uv run ruff check .

test:
	cd inference-svc && uv run pytest -v
	cd ui-svc && uv run pytest -v

clean:
	rm -rf proto/gen/
	rm -rf inference-svc/.venv
	rm -rf ui-svc/.venv
	rm -rf **/__pycache__
	rm -rf **/.pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
