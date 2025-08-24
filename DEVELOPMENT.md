# Development Guide

## Quick Start

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Run the toy example**:
   ```bash
   poetry run python examples/toy_sequence.py
   ```

4. **Run tests**:
   ```bash
   poetry run pytest -v
   ```

## Development Commands

### Using Make (Recommended)

```bash
# Complete development setup
make dev-setup

# Run example
make example

# Run tests
make test

# Code quality checks
make quality-check

# All checks (quality + tests)
make all-checks

# Format code
make format

# Quick development workflow
make dev
```

### Using Poetry Directly

```bash
# Install dependencies
poetry install

# Run tests with coverage
poetry run pytest --cov=symbiont --cov-report=term-missing

# Type checking
poetry run mypy symbiont/

# Code formatting
poetry run black symbiont/ tests/ examples/

# Linting
poetry run ruff check symbiont/ tests/ examples/

# Install pre-commit hooks
poetry run pre-commit install

# Run all quality checks
poetry run pre-commit run --all-files
```

### Additional Development Commands

```bash
# Security checks
make security

# Performance profiling
make profile

# Memory profiling
make memory-profile

# Clean build artifacts
make clean

# Show project statistics
make stats

# Update dependencies
make deps-update
```

## Project Structure

```
symbiont-core/
├── symbiont/              # Main package
│   ├── core/              # Core constraint system
│   ├── bridge/            # Neuro-symbolic bridge
│   ├── generators/        # Generator interfaces
│   ├── domains/           # Domain-specific constraints
│   └── utils/             # Utilities
├── examples/              # Example scripts
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Running the Demo

The toy example demonstrates:
- Constraint definition using DSL
- Mock DNA sequence generation
- Constraint evaluation and visualization
- Quality comparison between constrained/unconstrained generation

```bash
poetry run python examples/toy_sequence.py
```

Expected output includes:
- Generated DNA sequences with and without constraints
- Satisfaction scores for different constraint types
- Quality metrics comparison
- Advanced DSL feature demonstrations

## Next Steps

1. **Integrate Real Models**: Replace mock generators with actual transformers/diffusion models
2. **Add More Domains**: Implement protein, chemical, and other domain constraints
3. **Performance Optimization**: Optimize constraint compilation and evaluation
4. **Web Interface**: Build Streamlit/Gradio interface for interactive use
5. **Documentation**: Expand API documentation and tutorials

## Testing

Run different test suites:

```bash
# All tests
poetry run pytest

# Specific test file
poetry run pytest tests/test_constraints.py -v

# Tests with specific marker
poetry run pytest -m "not slow"

# Coverage report
poetry run pytest --cov=symbiont --cov-report=html
```

## Code Quality

The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pre-commit** hooks for automated quality checks

All code should pass these checks before committing.

## Architecture

### Core Components

1. **Constraints** (`symbiont.core`): Protocol-based constraint system
2. **DSL** (`symbiont.core.dsl`): Declarative rule definition
3. **Bridge** (`symbiont.bridge`): Differentiable logic compilation
4. **Generators** (`symbiont.generators`): Model interfaces
5. **Domains** (`symbiont.domains`): Domain-specific implementations

### Key Design Principles

- **Protocol-based**: Flexible, extensible interfaces
- **Differentiable**: All constraints compile to differentiable operations
- **Domain-agnostic**: Core framework works across domains
- **Composable**: Constraints combine naturally with logical operators
- **Type-safe**: Full type hints and validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all quality checks pass
5. Submit a pull request

## Performance Notes

- Mock generators are for testing only
- Real performance comes with actual ML model integration
- Constraint compilation happens once, evaluation is fast
- Batch operations are optimized for GPU acceleration
