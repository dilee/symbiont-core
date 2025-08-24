# Makefile for Symbiont Development

.PHONY: help install test lint format type-check clean docs example quality-check all-checks setup dev-setup

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install dependencies"
	@echo "  dev-setup     Complete development setup"
	@echo "  test          Run tests"
	@echo "  test-watch    Run tests in watch mode"
	@echo "  lint          Run linting"
	@echo "  format        Format code"
	@echo "  type-check    Run type checking"
	@echo "  quality-check Run all quality checks"
	@echo "  all-checks    Run all checks including tests"
	@echo "  example       Run toy sequence example"
	@echo "  clean         Clean build artifacts"
	@echo "  docs          Build documentation"
	@echo "  pre-commit    Run pre-commit on all files"
	@echo "  benchmark     Run benchmark tests"
	@echo "  profile       Profile example execution"

# Installation and setup
install:
	poetry install

dev-setup: install
	poetry run pre-commit install
	@echo "Development environment setup complete!"
	@echo "Run 'make example' to test the installation."

setup: dev-setup

# Testing
test:
	poetry run pytest -v --cov=symbiont --cov-report=term-missing

test-watch:
	poetry run pytest-watch --runner "poetry run pytest"

test-fast:
	poetry run pytest -x --ff

test-coverage:
	poetry run pytest --cov=symbiont --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

benchmark:
	poetry run pytest tests/ -m benchmark --benchmark-only

# Code quality
lint:
	poetry run ruff check symbiont/ tests/ examples/

format:
	poetry run black symbiont/ tests/ examples/
	poetry run ruff format symbiont/ tests/ examples/

type-check:
	poetry run mypy symbiont/

quality-check: lint type-check
	poetry run bandit -r symbiont/
	@echo "All quality checks passed!"

all-checks: quality-check test
	@echo "All checks passed!"

pre-commit:
	poetry run pre-commit run --all-files

# Security
security:
	poetry run bandit -r symbiont/
	poetry run safety check

# Examples and documentation
example:
	poetry run python examples/toy_sequence.py

docs:
	@echo "Building documentation..."
	@echo "Documentation build not implemented yet."

# Performance and profiling
profile:
	poetry run python -m cProfile -o profile.out examples/toy_sequence.py
	@echo "Profile saved to profile.out"

memory-profile:
	poetry run mprof run examples/toy_sequence.py
	poetry run mprof plot
	@echo "Memory profile plot generated"

# Maintenance
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf .venv/

# Docker (future)
docker-build:
	@echo "Docker build not implemented yet"

docker-test:
	@echo "Docker test not implemented yet"

# Release (future)
version-patch:
	poetry version patch

version-minor:
	poetry version minor

version-major:
	poetry version major

publish-test:
	poetry build
	poetry config repositories.testpypi https://test.pypi.org/legacy/
	poetry publish -r testpypi

publish:
	poetry build
	poetry publish

# Development utilities
deps-update:
	poetry update

deps-show:
	poetry show

env-info:
	@echo "Python version:"
	poetry run python --version
	@echo "Poetry version:"
	poetry --version
	@echo "Virtual environment:"
	poetry env info

check-deps:
	poetry run pip-audit

# Jupyter
notebook:
	poetry run jupyter notebook

lab:
	poetry run jupyter lab

# Quick development workflow
dev: format lint type-check test
	@echo "Development check complete!"

ci: quality-check test
	@echo "CI checks complete!"

# Show project statistics
stats:
	@echo "Code statistics:"
	@find symbiont -name "*.py" | xargs wc -l | tail -1
	@echo "Test statistics:"
	@find tests -name "*.py" | xargs wc -l | tail -1
	@echo "Test count:"
	@poetry run pytest --collect-only -q | grep "tests collected" || echo "No tests found"
