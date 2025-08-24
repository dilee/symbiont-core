"""Core constraint protocols and base implementations."""

from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable

import torch

from symbiont.core.types import validate_satisfaction_score, validate_tensor_shape


@runtime_checkable
class Constraint(Protocol):
    """Base constraint protocol that all constraints must implement."""

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate constraint satisfaction for input tensor.

        Args:
            x: Input tensor to evaluate

        Returns:
            Satisfaction score in range [0,1] where 1 is fully satisfied
        """
        ...

    def __and__(self, other: Constraint) -> AndConstraint:
        """Logical AND operation between constraints."""
        return AndConstraint(self, other)

    def __or__(self, other: Constraint) -> OrConstraint:
        """Logical OR operation between constraints."""
        return OrConstraint(self, other)

    def __invert__(self) -> NotConstraint:
        """Logical NOT operation for constraint."""
        return NotConstraint(self)


class BaseConstraint(abc.ABC):
    """Abstract base class for concrete constraint implementations with validation."""

    @abc.abstractmethod
    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate constraint satisfaction with input/output validation."""
        pass

    def _validate_and_evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Validate input, evaluate constraint, and validate output."""
        # Validate input tensor
        x = validate_tensor_shape(x)

        # Call actual constraint evaluation
        result = self.satisfaction(x)

        # Validate output satisfaction scores
        result = validate_satisfaction_score(result)

        return result

    def __and__(self, other: Constraint) -> AndConstraint:
        return AndConstraint(self, other)

    def __or__(self, other: Constraint) -> OrConstraint:
        return OrConstraint(self, other)

    def __invert__(self) -> NotConstraint:
        return NotConstraint(self)


class CompositeConstraint(BaseConstraint):
    """Base class for constraints that combine other constraints."""

    def __init__(self, *constraints: Constraint):
        if not constraints:
            raise ValueError("At least one constraint must be provided")
        self.constraints = constraints

    def __repr__(self) -> str:
        constraint_reprs = [repr(c) for c in self.constraints]
        return f"{self.__class__.__name__}({', '.join(constraint_reprs)})"


class AndConstraint(CompositeConstraint):
    """Logical AND of multiple constraints (conjunction)."""

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """AND satisfaction is the minimum of all constraint satisfactions."""
        scores = [c.satisfaction(x) for c in self.constraints]
        return torch.stack(scores, dim=0).min(dim=0)[0]


class OrConstraint(CompositeConstraint):
    """Logical OR of multiple constraints (disjunction)."""

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """OR satisfaction is the maximum of all constraint satisfactions."""
        scores = [c.satisfaction(x) for c in self.constraints]
        return torch.stack(scores, dim=0).max(dim=0)[0]


class NotConstraint(BaseConstraint):
    """Logical NOT of a constraint (negation)."""

    def __init__(self, constraint: Constraint):
        self.constraint = constraint

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """NOT satisfaction is 1 - constraint satisfaction."""
        return torch.tensor(1.0) - self.constraint.satisfaction(x)

    def __repr__(self) -> str:
        return f"NotConstraint({repr(self.constraint)})"


class AlwaysTrue(BaseConstraint):
    """Constraint that is always satisfied (tautology)."""

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0] if x.dim() > 0 else 1, device=x.device)

    def __repr__(self) -> str:
        return "AlwaysTrue()"


class AlwaysFalse(BaseConstraint):
    """Constraint that is never satisfied (contradiction)."""

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0] if x.dim() > 0 else 1, device=x.device)

    def __repr__(self) -> str:
        return "AlwaysFalse()"


class WeightedConstraint(BaseConstraint):
    """Wrapper that applies a weight to constraint satisfaction."""

    def __init__(self, constraint: Constraint, weight: float = 1.0):
        if weight < 0:
            raise ValueError("Weight must be non-negative")
        self.constraint = constraint
        self.weight = weight

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        base_satisfaction = self.constraint.satisfaction(x)
        return torch.clamp(self.weight * base_satisfaction, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"WeightedConstraint({repr(self.constraint)}, weight={self.weight})"
