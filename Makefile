.PHONY: install test lint format run docker-build docker-up docker-down clean help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using uv
	uv sync --all-extras

install-dev: ## Install development dependencies
	uv sync --all-extras --dev

test: ## Run tests with coverage
	uv run pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	uv run pytest tests/integration/ -v

lint: ## Run linting checks
	uv run ruff check app/ tests/
	uv run mypy app/

format: ## Format code with black and ruff
	uv run black app/ tests/ scripts/
	uv run ruff check --fix app/ tests/ scripts/

format-check: ## Check code formatting without making changes
	uv run black --check app/ tests/ scripts/
	uv run ruff check app/ tests/ scripts/

run: ## Run the FastAPI application
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the FastAPI application in production mode
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

init-db: ## Initialize the database
	uv run python scripts/init_db.py

seed-data: ## Seed test data
	uv run python scripts/seed_data.py

docker-build: ## Build Docker image
	docker build -t quant-trading-intelligence:latest .

docker-up: ## Start all services with docker-compose
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

docker-logs: ## View docker-compose logs
	docker-compose logs -f

docker-clean: ## Remove all containers, volumes, and images
	docker-compose down -v
	docker rmi quant-trading-intelligence:latest

clean: ## Clean generated files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

clean-data: ## Clean data directories (WARNING: deletes all data)
	rm -rf data/
	rm -rf chroma_data/
	rm -rf logs/
	rm -rf models/

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	uv run pre-commit run --all-files

check: format-check lint test ## Run all checks (format, lint, test)

all: clean install format lint test ## Clean, install, format, lint, and test
