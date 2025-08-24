"""Core type definitions and data structures with runtime validation."""

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


class ValidationError(Exception):
    """Raised when constraint validation fails."""

    pass


def validate_tensor_shape(
    tensor: torch.Tensor, expected_dims: int | None = None
) -> torch.Tensor:
    """Validate tensor shape and constraints."""
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"Expected torch.Tensor, got {type(tensor)}")

    if expected_dims is not None and tensor.dim() != expected_dims:
        raise ValidationError(f"Expected {expected_dims}D tensor, got {tensor.dim()}D")

    if tensor.numel() == 0:
        raise ValidationError("Empty tensor not allowed for constraint evaluation")

    return tensor


def validate_satisfaction_score(score: torch.Tensor) -> torch.Tensor:
    """Validate satisfaction score is in [0,1] range."""
    if not torch.all((score >= 0.0) & (score <= 1.0)):
        raise ValidationError(
            f"Satisfaction score must be in [0,1], got range [{score.min():.3f}, {score.max():.3f}]"
        )

    if torch.any(torch.isnan(score)) or torch.any(torch.isinf(score)):
        raise ValidationError("Satisfaction score contains NaN or Inf values")

    return score


@dataclass
class GenerationConfig:
    """Configuration for the generation process with validation."""

    max_length: int = 100
    temperature: float = 1.0
    constraint_weight: float = 1.0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    batch_size: int = 1
    num_samples: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_length <= 0:
            raise ValidationError(f"max_length must be positive, got {self.max_length}")

        if self.temperature <= 0.0:
            raise ValidationError(
                f"temperature must be positive, got {self.temperature}"
            )

        if self.constraint_weight < 0.0:
            raise ValidationError(
                f"constraint_weight must be non-negative, got {self.constraint_weight}"
            )

        if self.batch_size <= 0:
            raise ValidationError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_samples <= 0:
            raise ValidationError(
                f"num_samples must be positive, got {self.num_samples}"
            )

        if self.seed is not None and self.seed < 0:
            raise ValidationError(f"seed must be non-negative, got {self.seed}")


@dataclass
class ConstraintResult:
    """Result of constraint evaluation."""

    satisfaction: torch.Tensor  # Satisfaction score [0,1]
    confidence: float = 1.0  # Confidence in the evaluation
    metadata: dict[str, Any] | None = None


TNormType = Literal["product", "lukasiewicz", "godel", "drastic"]
