.PHONY: help install install-dev data run baselines transformer vizu vizu-transformer \
        app api test lint format mlflow-ui walkforward multiseed significance multi-asset \
        interpret interpret-transformer docker-build docker-up docker-down

PYTHON ?= python3
STREAMLIT ?= streamlit
LOGS ?= data/artifacts/lstm_logs.csv
OUTDIR ?= experiments/figures

help:
	@echo "Targets:"
	@echo "  make install         Install runtime dependencies"
	@echo "  make install-dev     Install runtime + dev dependencies (tests, lint)"
	@echo "  make data            Prepare the dataset"
	@echo "  make run             Train the LSTM model (default settings)"
	@echo "  make baselines       Train and evaluate baseline models"
	@echo "  make transformer     Train the Transformer model"
	@echo "  make vizu            Generate LSTM training plots from logs"
	@echo "  make vizu-transformer  Generate Transformer training plots"
	@echo "  make app             Run the Streamlit dashboard"
	@echo "  make api             Run the FastAPI inference service (uvicorn, port 8000)"
	@echo "  make mlflow-ui       Launch MLflow UI (local backend)"
	@echo "  make walkforward     Walk-forward (purged) cross-validation"
	@echo "  make multiseed       Multi-seed LSTM training with confidence intervals"
	@echo "  make significance    Statistical significance + cost sensitivity for the LSTM"
	@echo "  make multi-asset     Train the LSTM across BTC-USD / ETH-USD / ^GSPC"
	@echo "  make interpret       Permutation feature importance for the LSTM"
	@echo "  make interpret-transformer  Feature importance + attention map for the Transformer"
	@echo "  make docker-build    Build the Docker image"
	@echo "  make docker-up       Start dashboard (8501) + API (8000) containers"
	@echo "  make docker-down     Stop the containers"
	@echo "  make test            Run the test suite"
	@echo "  make lint            Run ruff lint checks"
	@echo "  make format          Auto-format code with ruff"
	@echo ""
	@echo "Overrides:"
	@echo "  make vizu LOGS=data/artifacts/transformer_logs.csv OUTDIR=experiments/figures"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

data:
	$(PYTHON) -m src.data.dataset

run:
	$(PYTHON) -m src.training.run_lstm

baselines:
	$(PYTHON) -m src.training.run_baselines

transformer:
	$(PYTHON) -m src.training.run_transformer

vizu:
	$(PYTHON) -m src.viz.plot_training --logs $(LOGS) --outdir $(OUTDIR)

vizu-transformer:
	$(PYTHON) -m src.viz.plot_training --logs data/artifacts/transformer_logs.csv --outdir $(OUTDIR)

app:
	$(STREAMLIT) run src/app/streamlit_app.py

api:
	uvicorn src.serving.api:app --reload --port 8000

mlflow-ui:
	mlflow ui --backend-store-uri file:experiments/mlruns --port 5000

walkforward:
	$(PYTHON) -m src.validation.walkforward

multiseed:
	$(PYTHON) -m src.validation.multiseed

significance:
	$(PYTHON) -m src.validation.significance

multi-asset:
	$(PYTHON) -m src.training.run_multi_asset

interpret:
	$(PYTHON) -m src.interpret.report --model lstm

interpret-transformer:
	$(PYTHON) -m src.interpret.report --model transformer

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

test:
	$(PYTHON) -m pytest tst/ -v

lint:
	$(PYTHON) -m ruff check src/ tst/

format:
	$(PYTHON) -m ruff format src/ tst/
