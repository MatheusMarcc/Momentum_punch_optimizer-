.PHONY: install test lint demo check

install:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest

lint:
	ruff check .

demo:
	python main.py

check: lint test
