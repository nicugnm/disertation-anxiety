.PHONY: help install install-dev clean test lint format collect preprocess label train evaluate smoke

PY ?= python
ENV ?= .venv

help:
	@echo "Targets:"
	@echo "  install        Install package + runtime deps"
	@echo "  install-dev    Install with dev extras"
	@echo "  test           Run pytest"
	@echo "  lint           Run ruff + mypy"
	@echo "  format         Auto-format with ruff"
	@echo "  smoke          End-to-end smoke run on synthetic data"
	@echo "  collect        Collect Reddit data (uses configs/subreddits.yaml)"
	@echo "  preprocess     Clean + anonymize raw data"
	@echo "  label          Apply 3-tier labeling"
	@echo "  train          Train all models"
	@echo "  evaluate       Run evaluation suite"

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m spacy download en_core_web_sm
	$(PY) -m nltk.downloader punkt stopwords vader_lexicon

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

test:
	pytest -v

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

smoke:
	anxiety smoke-run

collect:
	anxiety collect --config configs/subreddits.yaml

preprocess:
	anxiety preprocess

label:
	anxiety label --tier weak
	anxiety label --tier llm
	anxiety label --aggregate

train:
	anxiety train --model tfidf
	anxiety train --model xgboost
	anxiety train --model transformer

evaluate:
	anxiety evaluate --all

plots:
	anxiety plot --run-dir experiments/runs/tfidf_logreg

docs:
	@echo "Documentation lives under docs/. Read README.md for the entry point."
	@ls docs/*.md
