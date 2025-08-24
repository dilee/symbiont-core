"""Domain-specific language for constraint definition."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from symbiont.core.constraints import Constraint, WeightedConstraint
from symbiont.core.operators import all_of, any_of


class Rules:
    """
    DSL for defining constraints in a declarative manner.

    Provides a fluent interface for building complex constraint systems
    that can be compiled into differentiable loss functions.

    Example:
        >>> rules = Rules()
        >>> rules.enforce(Contains("ATG"))           # Hard constraint
        >>> rules.constrain(Length(10, 20))          # Soft constraint
        >>> rules.forbid(HasPattern("AAA+"))         # Negation constraint
        >>> rules.prefer(HigherScore(metric_fn))     # Optimization objective
    """

    def __init__(self) -> None:
        self._constraints: list[Constraint] = []
        self._hard_constraints: list[Constraint] = []
        self._soft_constraints: list[Constraint] = []
        self._forbidden_constraints: list[Constraint] = []
        self._preferences: list[Constraint] = []
        self._weights: dict[Constraint, float] = defaultdict(lambda: 1.0)

    def enforce(self, constraint: Constraint, weight: float = 10.0) -> Rules:
        """
        Add a hard constraint that must be satisfied.

        Hard constraints have high weights and strongly guide generation.

        Args:
            constraint: The constraint to enforce
            weight: Weight for the constraint (default: 10.0)

        Returns:
            Self for method chaining
        """
        weighted_constraint = WeightedConstraint(constraint, weight)
        self._hard_constraints.append(weighted_constraint)
        self._constraints.append(weighted_constraint)
        return self

    def constrain(self, constraint: Constraint, weight: float = 1.0) -> Rules:
        """
        Add a soft constraint that should preferably be satisfied.

        Soft constraints guide generation but don't strictly enforce rules.

        Args:
            constraint: The constraint to apply
            weight: Weight for the constraint (default: 1.0)

        Returns:
            Self for method chaining
        """
        weighted_constraint = WeightedConstraint(constraint, weight)
        self._soft_constraints.append(weighted_constraint)
        self._constraints.append(weighted_constraint)
        return self

    def forbid(self, constraint: Constraint, weight: float = 5.0) -> Rules:
        """
        Add a constraint that must NOT be satisfied.

        Equivalent to enforce(~constraint).

        Args:
            constraint: The constraint to forbid
            weight: Weight for the negated constraint (default: 5.0)

        Returns:
            Self for method chaining
        """
        forbidden = ~constraint
        weighted_constraint = WeightedConstraint(forbidden, weight)
        self._forbidden_constraints.append(weighted_constraint)
        self._constraints.append(weighted_constraint)
        return self

    def prefer(self, constraint: Constraint, weight: float = 0.5) -> Rules:
        """
        Add a weak preference that guides generation.

        Preferences have low weights and provide gentle nudges.

        Args:
            constraint: The constraint to prefer
            weight: Weight for the constraint (default: 0.5)

        Returns:
            Self for method chaining
        """
        weighted_constraint = WeightedConstraint(constraint, weight)
        self._preferences.append(weighted_constraint)
        self._constraints.append(weighted_constraint)
        return self

    def require_all(self, *constraints: Constraint, weight: float = 1.0) -> Rules:
        """
        Require ALL of the given constraints to be satisfied.

        Equivalent to enforce(constraint1 & constraint2 & ... & constraintN).

        Args:
            *constraints: Constraints that must all be satisfied
            weight: Weight for the combined constraint

        Returns:
            Self for method chaining
        """
        combined = all_of(*constraints)
        return self.enforce(combined, weight)

    def require_any(self, *constraints: Constraint, weight: float = 1.0) -> Rules:
        """
        Require ANY of the given constraints to be satisfied.

        Equivalent to enforce(constraint1 | constraint2 | ... | constraintN).

        Args:
            *constraints: Constraints where at least one must be satisfied
            weight: Weight for the combined constraint

        Returns:
            Self for method chaining
        """
        combined = any_of(*constraints)
        return self.enforce(combined, weight)

    def forbid_all(self, *constraints: Constraint, weight: float = 1.0) -> Rules:
        """
        Forbid ALL of the given constraints from being satisfied.

        Equivalent to forbid(constraint1 | constraint2 | ... | constraintN).

        Args:
            *constraints: Constraints that must not be satisfied
            weight: Weight for the combined constraint

        Returns:
            Self for method chaining
        """
        combined = any_of(*constraints)
        return self.forbid(combined, weight)

    @property
    def constraints(self) -> list[Constraint]:
        """Get all constraints."""
        return self._constraints.copy()

    @property
    def hard_constraints(self) -> list[Constraint]:
        """Get only hard constraints."""
        return self._hard_constraints.copy()

    @property
    def soft_constraints(self) -> list[Constraint]:
        """Get only soft constraints."""
        return self._soft_constraints.copy()

    @property
    def forbidden_constraints(self) -> list[Constraint]:
        """Get only forbidden constraints."""
        return self._forbidden_constraints.copy()

    @property
    def preferences(self) -> list[Constraint]:
        """Get only preference constraints."""
        return self._preferences.copy()

    def evaluate(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Evaluate all constraints against input tensor.

        Args:
            x: Input tensor to evaluate

        Returns:
            Dictionary containing satisfaction scores for different constraint types
        """
        results = {
            "total": torch.zeros(x.shape[0] if x.dim() > 0 else 1, device=x.device),
            "hard": torch.zeros(x.shape[0] if x.dim() > 0 else 1, device=x.device),
            "soft": torch.zeros(x.shape[0] if x.dim() > 0 else 1, device=x.device),
            "forbidden": torch.zeros(x.shape[0] if x.dim() > 0 else 1, device=x.device),
            "preferences": torch.zeros(
                x.shape[0] if x.dim() > 0 else 1, device=x.device
            ),
        }

        # Evaluate each constraint type
        for constraints, key in [
            (self._constraints, "total"),
            (self._hard_constraints, "hard"),
            (self._soft_constraints, "soft"),
            (self._forbidden_constraints, "forbidden"),
            (self._preferences, "preferences"),
        ]:
            if constraints:
                scores = [c.satisfaction(x) for c in constraints]
                results[key] = torch.stack(scores, dim=0).mean(dim=0)

        return results

    def __len__(self) -> int:
        """Get total number of constraints."""
        return len(self._constraints)

    def __bool__(self) -> bool:
        """Check if any constraints are defined."""
        return len(self._constraints) > 0

    def __repr__(self) -> str:
        """String representation of the rules."""
        parts = []
        if self._hard_constraints:
            parts.append(f"hard={len(self._hard_constraints)}")
        if self._soft_constraints:
            parts.append(f"soft={len(self._soft_constraints)}")
        if self._forbidden_constraints:
            parts.append(f"forbidden={len(self._forbidden_constraints)}")
        if self._preferences:
            parts.append(f"preferences={len(self._preferences)}")

        constraint_summary = ", ".join(parts) if parts else "empty"
        return f"Rules({constraint_summary})"


class RuleBuilder:
    """
    Builder pattern for creating complex rule combinations.

    Provides a more structured approach to building constraint systems
    with conditional logic and dynamic rule generation.
    """

    def __init__(self) -> None:
        self._rules = Rules()
        self._conditions: list[tuple[Any, Rules]] = []

    def when(self, condition: Any) -> ConditionBuilder:
        """
        Add conditional constraints.

        Args:
            condition: Condition to evaluate (implementation-specific)

        Returns:
            Builder for adding constraints under this condition
        """
        return ConditionBuilder(self, condition)

    def build(self) -> Rules:
        """
        Build the final Rules object.

        Returns:
            Completed Rules object with all constraints
        """
        # For now, return the base rules
        # In the future, this could evaluate conditions and merge rules
        return self._rules


class ConditionBuilder:
    """Builder for conditional constraint application."""

    def __init__(self, parent: RuleBuilder, condition: Any):
        self.parent = parent
        self.condition = condition
        self._conditional_rules = Rules()

    def enforce(self, constraint: Constraint, weight: float = 10.0) -> ConditionBuilder:
        """Add conditional hard constraint."""
        self._conditional_rules.enforce(constraint, weight)
        return self

    def constrain(
        self, constraint: Constraint, weight: float = 1.0
    ) -> ConditionBuilder:
        """Add conditional soft constraint."""
        self._conditional_rules.constrain(constraint, weight)
        return self

    def forbid(self, constraint: Constraint, weight: float = 5.0) -> ConditionBuilder:
        """Add conditional forbidden constraint."""
        self._conditional_rules.forbid(constraint, weight)
        return self

    def then(self) -> RuleBuilder:
        """
        Complete the conditional rule and return to parent builder.

        Returns:
            Parent RuleBuilder for continued construction
        """
        self.parent._conditions.append((self.condition, self._conditional_rules))
        return self.parent
