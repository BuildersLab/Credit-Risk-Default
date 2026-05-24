.PHONY: setup data train evaluate app test clean lint format

setup:
	pip install -e ".[dev]"
	pre-commit install
	@echo "Setup complete. Copy .env.example to .env and fill in your values."

data:
	python -m src.data_pipeline

train:
	python -m src.train

evaluate:
	python -m src.evaluate

app:
	streamlit run app/main.py

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/*
	rm -rf models/*.pkl models/*.joblib
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Cleaned processed data and model artefacts."

lint:
	ruff check src/ app/ tests/
	black --check src/ app/ tests/

format:
	black src/ app/ tests/
	ruff check --fix src/ app/ tests/
