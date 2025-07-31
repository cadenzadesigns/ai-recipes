.PHONY: help install sync ruff format lint type-check test clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies with uv"
	@echo "  make sync       - Sync dependencies with uv"
	@echo "  make ruff       - Run ruff linter"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Run all linters (ruff + black check)"
	@echo "  make type-check - Run mypy type checker"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Remove cache files"

# Install dependencies
install:
	uv sync
	uv add --dev pytest pytest-cov black ruff mypy types-requests

# Sync dependencies
sync:
	uv sync

# Run ruff linter with auto-fix (runs format first)
ruff: format
	uv run ruff check --fix src/ app/

# Format code with black
format:
	uv run black src/ app/

# Check formatting without modifying
format-check:
	uv run black --check src/ app/

# Run all linting
lint: ruff format-check

# Type checking
type-check:
	uv run mypy src/ app/

# Run tests
test:
	uv run pytest

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf recipes/