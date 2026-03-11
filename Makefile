.PHONY: setup download-pcb convert-pcb train-pcb evaluate infer app api watch export-onnx docker-build docker-run clean help

# ──────────────────────────────────────────────
# AIT Visual Inspector — Makefile
# ──────────────────────────────────────────────

PYTHON ?= python3
PIP ?= pip3
STREAMLIT ?= streamlit
DOCKER_TAG ?= ait-visual-inspector:latest

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────

setup: ## Install dependencies
	$(PIP) install -r requirements.txt

setup-dev: ## Install dependencies + dev tools
	$(PIP) install -r requirements.txt
	$(PIP) install pytest black isort flake8

# ── Data (PCB) ────────────────────────────────

download-pcb: ## Download PKU-Market-PCB dataset from Kaggle
	$(PYTHON) -m src.data.download_pcb

convert-pcb: ## Convert PCB VOC annotations to YOLO format
	$(PYTHON) -m src.data.convert_to_yolo --dataset pcb --validate

prepare-pcb: download-pcb convert-pcb ## Download + convert PCB data

# ── Data (MVTec — legacy) ─────────────────────

download-mvtec: ## Download MVTec AD dataset
	$(PYTHON) -m src.data.download_mvtec --config configs/data.yaml

convert-mvtec: ## Convert MVTec AD masks to YOLO format
	$(PYTHON) -m src.data.convert_to_yolo --dataset mvtec --validate

# ── Training ──────────────────────────────────

train-pcb: ## Train YOLOv8 on PCB defect data
	$(PYTHON) train.py --config configs/model.yaml --data configs/data.yaml

train-quick: ## Quick training (5 epochs) for testing
	$(PYTHON) train.py --config configs/model.yaml --data configs/data.yaml \
		--epochs 5 --img 320 --batch 4

# ── Evaluation ────────────────────────────────

evaluate: ## Run full evaluation on test set
	$(PYTHON) evaluate.py --data configs/data.yaml --config configs/model.yaml

# ── Inference ─────────────────────────────────

infer: ## Run inference (set SOURCE=path/to/image)
	$(PYTHON) infer.py --source $(SOURCE) --config configs/model.yaml --app-config configs/app.yaml

# ── Export ────────────────────────────────────

export-onnx: ## Export model to ONNX format with benchmarks
	$(PYTHON) -m src.export.onnx_export --weights models/pcb_mixed_best.pt --benchmark

# ── Application ───────────────────────────────

app: ## Launch Streamlit dashboard
	$(STREAMLIT) run app.py -- --config configs/app.yaml

api: ## Launch FastAPI REST API
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

watch: ## Start folder watcher (auto-inspect data/inbox/)
	$(PYTHON) -m src.watcher.watch

# ── Docker ────────────────────────────────────

docker-build: ## Build Docker image
	docker build -t $(DOCKER_TAG) .

docker-run: ## Run Docker container (GPU)
	docker run -p 8501:8501 --gpus all $(DOCKER_TAG)

docker-run-cpu: ## Run Docker container (CPU only)
	docker run -p 8501:8501 $(DOCKER_TAG)

deploy: ## Deploy with Docker Compose (API + Dashboard)
	docker compose up -d

deploy-stop: ## Stop Docker Compose
	docker compose down

# ── Testing ───────────────────────────────────

test: ## Run unit tests
	$(PYTHON) -m pytest tests/ -v

# ── Utilities ─────────────────────────────────

clean: ## Remove generated files
	rm -rf runs/ reports/ logs/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean ## Remove everything including data
	rm -rf data/raw/ data/processed/ data/inbox/*

lint: ## Run linters
	black --check src/ *.py
	isort --check src/ *.py
	flake8 src/ *.py

format: ## Auto-format code
	black src/ *.py
	isort src/ *.py
