# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Symbiont** is a neuro-symbolic framework that fuses generative AI with domain-specific scientific constraints. It transforms generative models from probabilistic "black boxes" into steerable, constraint-guided partners for scientific discovery through differentiable logic and fuzzy operations.

## Key Commands

### Development Setup
```bash
# Install dependencies (requires Poetry)
poetry install

# Complete dev setup with pre-commit hooks
make dev-setup
```

### Running Code
```bash
# Run toy DNA sequence generation example
make example
# or
poetry run python examples/toy_sequence.py

# Run progress reporter demo (shows live constraint satisfaction tracking)
poetry run python examples/progress_demo.py

# Run specific test file
poetry run pytest tests/test_constraints.py -v

# Run tests excluding slow tests
poetry run pytest -m "not slow"
```

### Code Quality & Testing
```bash
# Full development check (format + lint + type-check + test)
make dev

# Individual quality checks
make format        # Black + Ruff formatting
make lint          # Ruff linting
make type-check    # MyPy type checking
make test          # Run all tests with coverage

# All quality checks + tests
make all-checks

# Security checks
make security
```

### Performance Analysis
```bash
# Run benchmarks
make benchmark

# Profile example execution
make profile

# Memory profiling
make memory-profile
```

## Architecture & Design Patterns

### Neuro-Symbolic Bridge Pattern

The framework's core innovation is the **differentiable constraint compiler** that bridges symbolic rules with neural networks:

1. **Symbolic Layer** (`symbiont/core/dsl.py`): Scientists define constraints using a fluent DSL
   - `rules.enforce()` - Hard constraints (high weight)
   - `rules.constrain()` - Soft constraints (medium weight)
   - `rules.forbid()` - Negation constraints
   - `rules.prefer()` - Weak preferences (low weight)

2. **Compilation Layer** (`symbiont/bridge/compiler.py`): Converts symbolic constraints to differentiable operations
   - Uses fuzzy logic t-norms for logical operations
   - Creates computational graph supporting backpropagation
   - Generates constraint violation loss for gradient descent

3. **Gradient Guidance** (`symbiont/bridge/compiler.py:215`): Steers generation through constraint gradients
   - Computes gradients of constraint satisfaction
   - Iteratively refines outputs toward constraint compliance
   - Enables real-time constraint-guided generation

### Protocol-Based Constraint System

All constraints implement the `Constraint` protocol (`symbiont/core/constraints.py:14`):
- `satisfaction(x: torch.Tensor) -> torch.Tensor`: Returns scores in [0,1]
- Supports logical composition via operator overloading (`&`, `|`, `~`)
- Enables domain-specific implementations while maintaining unified interface

### Domain Specialization Pattern

Domain modules (`symbiont/domains/`) provide specialized constraints:
- **Base classes** define constraint categories (Pattern, Range, Composition, Structural)
- **Domain implementations** (e.g., `sequence.py`) provide biological constraints
- Extensible to new domains (proteins, molecules, materials)

### Generator Interface Pattern

Generators (`symbiont/generators/base.py`) follow a pluggable backend design:
- `generate()`: Unconstrained generation
- `constrained_generate()`: Constraint-guided generation with gradient refinement
- Mock generator for testing, ready for real model integration (transformers, diffusion)

## Key Technical Concepts

### Differentiable Logic
The framework uses **fuzzy logic t-norms** to make discrete logical operations differentiable:
- AND → t-norm (e.g., product: `a * b`)
- OR → s-norm (e.g., probabilistic sum: `a + b - a*b`)
- NOT → complement: `1 - a`

This enables gradient flow through logical constraint trees.

### Constraint Satisfaction Workflow
1. Parse DSL rules into constraint tree
2. Compile to differentiable constraint loss
3. Generate initial candidates
4. Compute constraint satisfaction scores
5. Backpropagate violation gradients
6. Update generation to improve satisfaction
7. Iterate until threshold met

### Weight Management
Constraints have different weights based on importance:
- Hard constraints: weight=10.0 (must satisfy)
- Soft constraints: weight=1.0 (should satisfy)
- Forbidden: weight=5.0 (must not satisfy)
- Preferences: weight=0.5 (gentle guidance)

## Testing Strategy

- **Unit tests** (`tests/test_*.py`): Test individual components
- **Integration tests** (`tests/test_integration.py`): Test end-to-end workflows
- **Benchmarks**: Performance tests marked with `@pytest.mark.benchmark`
- **Coverage target**: 50% minimum (currently enforced)

## Visualization & Monitoring

The framework includes visualization utilities for tracking constraint satisfaction:

### ProgressReporter
Real-time progress tracking during iterative generation:
- Live progress bars with color-coded satisfaction levels
- Trend indicators (improving ↑, declining ↓, stable →)
- ETA calculation and iteration tracking
- Final summary with success metrics

See `examples/progress_demo.py` for a complete demonstration.

## Future Integration Points

The framework is designed for integration with:
- **Protein generators**: RFdiffusion, Chroma, ProteinMPNN
- **Molecule generators**: Graph neural networks, VAEs
- **Language models**: Transformers for sequence generation
- **UI frameworks**: Streamlit/Gradio for interactive constraint definition

## Important Notes

- Mock generators are for demonstration only - real value comes from actual model integration
- Gradient-guided refinement requires continuous representations (not discrete tokens)
- The framework is model-agnostic through standardized generator interfaces
- Constraint compilation happens once; evaluation is optimized for batched GPU operations
