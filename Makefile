# oepnStock Makefile
# Korean Stock Trading System Build Commands

.PHONY: help install install-dev test test-cov lint format type-check clean docs run-tests setup

# Default target
help:
	@echo "oepnStock - Korean Stock Market Trading System"
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run code linting (flake8)"
	@echo "  format       Format code (black)"
	@echo "  type-check   Run type checking (mypy)"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  setup        Initial project setup"
	@echo "  run-backtest Run sample backtest"
	@echo "  check-env    Check environment configuration"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=oepnstock --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/ -m unit -v

test-integration:
	pytest tests/ -m integration -v

test-backtest:
	pytest tests/ -m backtest -v

# Code quality
lint:
	flake8 oepnstock tests

format:
	black oepnstock tests

format-check:
	black --check oepnstock tests

type-check:
	mypy oepnstock

# Combined quality check
quality-check: lint type-check format-check

# Build and packaging
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete

build:
	python setup.py sdist bdist_wheel

# Documentation
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

# Development setup
setup: install-dev
	@echo "Creating necessary directories..."
	mkdir -p logs data/raw data/processed data/backtest config/sessions
	@echo "Copying environment template..."
	cp .env.example .env
	@echo ""
	@echo "Setup complete!"
	@echo "Please edit .env file with your configuration"

# Environment checks
check-env:
	@echo "Checking Python version..."
	@python --version
	@echo "Checking installed packages..."
	@pip list | grep -E "(pandas|numpy|scikit-learn|pytest)"
	@echo "Checking environment variables..."
	@python -c "import os; print('DATABASE_URL:', 'SET' if os.getenv('DATABASE_URL') else 'NOT SET')"
	@python -c "import os; print('PAPER_TRADING:', os.getenv('PAPER_TRADING', 'true'))"

# Development utilities
run-example:
	python -m oepnstock.examples.basic_analysis

run-backtest:
	python -m oepnstock.examples.backtest_example

validate-config:
	python -c "from oepnstock.utils.config import config_manager; print('Config validation:', config_manager.validate_config({}))"

# Database operations (when implemented)
db-init:
	python -m oepnstock.database.init_db

db-migrate:
	python -m oepnstock.database.migrate

# Market data operations (when implemented)
fetch-data:
	python -m oepnstock.data.fetch_market_data

update-data:
	python -m oepnstock.data.update_daily

# Monitoring and logs
view-logs:
	tail -f logs/oepnstock.log

clear-logs:
	rm -f logs/*.log

# Performance testing
perf-test:
	python -m pytest tests/ -m "not slow" --benchmark-only

# Security checks
security-check:
	pip-audit

# All-in-one development check
dev-check: quality-check test-cov

# CI/CD pipeline simulation
ci: clean install-dev quality-check test-cov

# Release preparation
pre-release: clean quality-check test-cov docs build

# Docker operations (when implemented)
docker-build:
	docker build -t oepnstock:latest .

docker-run:
	docker run -it --rm oepnstock:latest

docker-test:
	docker run -it --rm oepnstock:latest make test