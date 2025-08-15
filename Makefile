# Makefile for ML Linear Algebra Journey
# 
# Common development tasks for maintaining code quality
# Run 'make help' to see available commands

.PHONY: help install dev-install test format lint typecheck clean all check demo

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install basic dependencies"
	@echo "  dev-install - Install development dependencies"
	@echo "  test        - Run tests with pytest"
	@echo "  format      - Format code with Black"
	@echo "  lint        - Lint code with Ruff"
	@echo "  typecheck   - Check types with MyPy"
	@echo "  clean       - Remove cache files and test artifacts"
	@echo "  check       - Run all quality checks (format, lint, typecheck, test)"
	@echo "  demo        - Run the main demonstration"
	@echo "  all         - Install dev dependencies and run all checks"

# Installation
install:
	pip install numpy

dev-install:
	pip install -e ".[dev]"

# Testing
test:
	python3 -m pytest -v

test-fast:
	python3 -m pytest -v -x

# Code quality
format:
	black .

format-check:
	black --check .

lint:
	ruff check .

lint-fix:
	ruff check --fix .

typecheck:
	mypy .

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf .ruff_cache

# Combined commands
check: format-check lint typecheck test

demo:
	python day6_linalg_utils.py

all: dev-install check

# Development workflow
dev-setup: dev-install
	@echo "Development environment setup complete!"
	@echo "Run 'make check' to verify everything works"

# Quick development cycle
quick-check: format lint test-fast

# Pre-commit checks (lightweight)
pre-commit: format-check lint typecheck
	@echo "Pre-commit checks passed!"
