.PHONY: help venv install install-dev freeze lint format test clean

help:
	@echo "Targets:"
	@echo "  venv         Create .venv and upgrade pip"
	@echo "  install      Install package (editable) + runtime deps"
	@echo "  install-dev  Install dev extras"
	@echo "  freeze       Export pinned requirements.txt and requirements-dev.txt"
	@echo "  format       Format code (black)"
	@echo "  lint         Lint code (ruff)"
	@echo "  test         Run tests (pytest)"
	@echo "  clean        Remove caches and build artifacts"

venv:
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip

install:
	. .venv/bin/activate && pip install -e .

install-dev:
	. .venv/bin/activate && pip install -e ".[dev]"

freeze:
	. .venv/bin/activate && pip freeze > requirements.txt
	. .venv/bin/activate && pip freeze > requirements-dev.txt

format:
	. .venv/bin/activate && black .

lint:
	. .venv/bin/activate && ruff check .

test:
	. .venv/bin/activate && pytest

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__ dist build *.egg-info