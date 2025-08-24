"""Logical operators for constraint composition."""

import torch

from symbiont.core.constraints import BaseConstraint, Constraint


def all_of(*constraints: Constraint) -> Constraint:
    """
    Create a constraint that requires ALL of the given constraints to be satisfied.

    Equivalent to constraint1 & constraint2 & ... & constraintN

    Args:
        *constraints: Variable number of constraints to combine

    Returns:
        A constraint that is satisfied when all input constraints are satisfied

    Example:
        >>> constraint = all_of(
        ...     Contains("ATG"),
        ...     Length(10, 20),
        ...     GCContent(0.4, 0.6)
        ... )
    """
    if not constraints:
        raise ValueError("At least one constraint must be provided")

    if len(constraints) == 1:
        return constraints[0]

    result = constraints[0]
    for constraint in constraints[1:]:
        result = result & constraint
    return result


def any_of(*constraints: Constraint) -> Constraint:
    """
    Create a constraint that requires ANY of the given constraints to be satisfied.

    Equivalent to constraint1 | constraint2 | ... | constraintN

    Args:
        *constraints: Variable number of constraints to combine

    Returns:
        A constraint that is satisfied when at least one input constraint is satisfied

    Example:
        >>> constraint = any_of(
        ...     Contains("ATG"),
        ...     Contains("GTG"),
        ...     Contains("TTG")
        ... )
    """
    if not constraints:
        raise ValueError("At least one constraint must be provided")

    if len(constraints) == 1:
        return constraints[0]

    result = constraints[0]
    for constraint in constraints[1:]:
        result = result | constraint
    return result


def none_of(*constraints: Constraint) -> Constraint:
    """
    Create a constraint that requires NONE of the given constraints to be satisfied.

    Equivalent to ~(constraint1 | constraint2 | ... | constraintN)

    Args:
        *constraints: Variable number of constraints to negate

    Returns:
        A constraint that is satisfied when none of the input constraints are satisfied

    Example:
        >>> constraint = none_of(
        ...     Contains("TTT"),  # No runs of T
        ...     Contains("AAA"),  # No runs of A
        ... )
    """
    if not constraints:
        raise ValueError("At least one constraint must be provided")

    return ~any_of(*constraints)


class AtLeast(BaseConstraint):
    """Constraint requiring at least N of the given constraints to be satisfied."""

    def __init__(self, n: int, *constraints: Constraint):
        if n < 1:
            raise ValueError("n must be at least 1")
        if n > len(constraints):
            raise ValueError("n cannot be greater than the number of constraints")

        self.n = n
        self.constraints = constraints

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Satisfaction based on how many constraints are satisfied (optimized)."""
        # Pre-allocate tensor for better memory efficiency
        batch_size = x.shape[0] if x.dim() > 1 else 1
        scores = torch.zeros(len(self.constraints), batch_size, device=x.device)

        # Compute all constraint satisfactions in one vectorized operation
        for i, constraint in enumerate(self.constraints):
            scores[i] = constraint.satisfaction(x)

        # Vectorized counting - much faster than individual operations
        satisfied_count = (scores > 0.5).sum(dim=0).float()

        # Normalize satisfaction based on how close we are to the target
        return torch.clamp(satisfied_count / self.n, 0.0, 1.0)

    def __repr__(self) -> str:
        constraint_reprs = [repr(c) for c in self.constraints]
        return f"AtLeast({self.n}, {', '.join(constraint_reprs)})"


class AtMost(BaseConstraint):
    """Constraint requiring at most N of the given constraints to be satisfied."""

    def __init__(self, n: int, *constraints: Constraint):
        if n < 0:
            raise ValueError("n must be non-negative")
        if n > len(constraints):
            # If n >= number of constraints, this is always satisfied
            n = len(constraints)

        self.n = n
        self.constraints = constraints

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Satisfaction based on not exceeding the constraint limit (optimized)."""
        # Pre-allocate tensor for better memory efficiency
        batch_size = x.shape[0] if x.dim() > 1 else 1
        scores = torch.zeros(len(self.constraints), batch_size, device=x.device)

        # Compute all constraint satisfactions in one vectorized operation
        for i, constraint in enumerate(self.constraints):
            scores[i] = constraint.satisfaction(x)

        # Vectorized counting
        satisfied_count = (scores > 0.5).sum(dim=0).float()

        # We're satisfied if we don't exceed the limit
        return torch.where(
            satisfied_count <= self.n,
            torch.ones_like(satisfied_count),
            torch.clamp(self.n / satisfied_count, 0.0, 1.0),
        )

    def __repr__(self) -> str:
        constraint_reprs = [repr(c) for c in self.constraints]
        return f"AtMost({self.n}, {', '.join(constraint_reprs)})"


class Exactly(BaseConstraint):
    """Constraint requiring exactly N of the given constraints to be satisfied."""

    def __init__(self, n: int, *constraints: Constraint):
        if n < 0 or n > len(constraints):
            raise ValueError(f"n must be between 0 and {len(constraints)}")

        self.n = n
        self.constraints = constraints

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Satisfaction based on exactly matching the target count (optimized)."""
        # Pre-allocate tensor for better memory efficiency
        batch_size = x.shape[0] if x.dim() > 1 else 1
        scores = torch.zeros(len(self.constraints), batch_size, device=x.device)

        # Compute all constraint satisfactions in one vectorized operation
        for i, constraint in enumerate(self.constraints):
            scores[i] = constraint.satisfaction(x)

        # Vectorized counting
        satisfied_count = (scores > 0.5).sum(dim=0).float()

        # Maximum satisfaction when exactly n constraints are satisfied
        distance_from_target = torch.abs(satisfied_count - self.n)
        max_distance = max(self.n, len(self.constraints) - self.n)

        if max_distance == 0:
            return torch.ones_like(satisfied_count)

        return torch.clamp(1.0 - distance_from_target / max_distance, 0.0, 1.0)

    def __repr__(self) -> str:
        constraint_reprs = [repr(c) for c in self.constraints]
        return f"Exactly({self.n}, {', '.join(constraint_reprs)})"
