# Quantitative Trading Intelligence System

A production-ready local quantitative trading system that ingests financial news, performs NLP analysis using Ollama, prepares features for xLSTM forecasting models, generates predictions, and provides portfolio optimization capabilities.

## Features

- **News Ingestion**: Fetch financial news from multiple sources (Yahoo Finance, Reuters, Bloomberg, etc.)
- **NLP Analysis**: Local LLM-powered sentiment analysis and structured data extraction using Ollama
- **Feature Engineering**: Time-series feature preparation for machine learning models
- **Prediction Service**: Load and run trained xLSTM models for trade score generation
- **Portfolio Optimization**: Modern portfolio theory with PyPortfolioOpt and CVXPY
- **Quantitative Finance**: Option pricing, bond valuation, and yield curve construction
- **Local-First**: All processing runs locally without external API dependencies

## Architecture

The system follows clean architecture principles with clear separation of concerns:

- **API Layer**: FastAPI routers handling HTTP requests/responses
- **Service Layer**: Business logic and orchestration
- **Repository Layer**: Data access abstractions
- **Domain Layer**: Core business entities and value objects

## Technology Stack

- **Python**: 3.11+
- **Web Framework**: FastAPI with async support
- **Package Manager**: uv (fast Python package manager)
- **Database**: SQLite with aiosqlite
- **Vector Store**: ChromaDB for embeddings
- **LLM Engine**: Ollama (llama3, nomic-embed-text)
- **Optimization**: PyPortfolioOpt, CVXPY
- **Quantitative Finance**: QuantLib-Python
- **Code Quality**: Black, Ruff, mypy

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai/) for local LLM inference
- Docker and docker-compose (optional, for containerized deployment)

## Installation

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### 2. Clone the repository

```bash
git clone <repository-url>
cd quant-trading-intelligence
```

### 3. Install dependencies

```bash
make install
# or
uv sync --all-extras
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Install Ollama and download models

```bash
# Install Ollama (see https://ollama.ai/)
# Download required models
ollama pull llama3
ollama pull nomic-embed-text
```

### 6. Initialize the database

```bash
make init-db
# or
uv run python scripts/init_db.py
```

## Usage

### Running the application

```bash
# Development mode with auto-reload
make run

# Production mode
make run-prod
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the application is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Running tests

```bash
# Run all tests with coverage
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration
```

### Code quality

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make check
```

## Docker Deployment

### Build and run with docker-compose

```bash
# Build images
make docker-build

# Start all services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

## Project Structure

```
quant-trading-intelligence/
├── app/                    # Application code
│   ├── api/               # API layer (routers, middleware)
│   ├── services/          # Business logic
│   ├── repositories/      # Data access layer
│   ├── models/            # Domain and API models
│   ├── adapters/          # External service adapters
│   ├── utils/             # Utility functions
│   └── core/              # Core functionality (config, exceptions)
├── config/                # Configuration files
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── scripts/              # Utility scripts
├── data/                 # Data directory (gitignored)
├── models/               # Model storage (gitignored)
├── logs/                 # Log files (gitignored)
└── chroma_data/          # ChromaDB storage (gitignored)
```

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options.

Key configuration sections:
- Database settings
- ChromaDB connection
- Ollama configuration
- News ingestion parameters
- NLP analysis settings
- Feature engineering options
- Portfolio optimization parameters
- Logging configuration

## Development

### Setting up development environment

```bash
# Install development dependencies
make install-dev

# Install pre-commit hooks
make pre-commit-install

# Run pre-commit on all files
make pre-commit-run
```

### Available Make commands

Run `make help` to see all available commands.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
# QuantFlowNLP
