.PHONY: help install data run baselines transformer vizu app mlflow-ui

PYTHON ?= python3
STREAMLIT ?= streamlit
LOGS ?= data/artifacts/lstm_logs.csv
OUTDIR ?= experiments/figures

help:
	@echo "Targets:"
	@echo "  make install     Install Python dependencies"
	@echo "  make data        Prepare the dataset"
	@echo "  make run         Train the LSTM model (default settings)"
	@echo "  make baselines   Train and evaluate baseline models"
	@echo "  make transformer Train the Transformer model"
	@echo "  make vizu        Generate training plots from logs"
	@echo "  make app         Run the Streamlit dashboard"
	@echo "  make mlflow-ui   Launch MLflow UI (local backend)"
	@echo ""
	@echo "Overrides:"
	@echo "  make vizu LOGS=data/artifacts/transformer_logs.csv OUTDIR=experiments/figures"

install:
	$(PYTHON) -m pip install -r requirements.txt

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

app:
	$(STREAMLIT) run src/app/streamlit_app.py

mlflow-ui:
	mlflow ui --backend-store-uri file:experiments/mlruns --port 5000
