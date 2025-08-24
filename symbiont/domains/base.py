"""Base classes for domain-specific constraints."""

import re
from abc import abstractmethod
from re import Pattern
from typing import Any

import torch

from symbiont.core.constraints import BaseConstraint
from symbiont.core.types import ValidationError, validate_tensor_shape


class PatternConstraint(BaseConstraint):
    """Base class for constraints that match patterns in sequences."""

    def __init__(self, pattern: str | Pattern[str], case_sensitive: bool = True):
        if isinstance(pattern, str):
            flags = 0 if case_sensitive else re.IGNORECASE
            self.pattern = re.compile(pattern, flags)
        else:
            self.pattern = pattern

        self.case_sensitive = case_sensitive

    @abstractmethod
    def _sequence_to_string(self, x: torch.Tensor) -> list[str]:
        """Convert tensor sequences to string representation."""
        pass

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Base implementation for pattern matching satisfaction with error handling."""
        try:
            # Validate input tensor
            x = validate_tensor_shape(x)

            # Convert tensor to string representation
            sequences = self._sequence_to_string(x)
            if not sequences:
                raise ValidationError("Failed to convert tensor to sequence strings")

            # Perform pattern matching with error handling
            satisfactions = []
            for seq_str in sequences:
                if not isinstance(seq_str, str):
                    raise ValidationError(
                        f"Expected string sequence, got {type(seq_str)}"
                    )

                try:
                    match_found = self.pattern.search(seq_str) is not None
                    satisfactions.append(1.0 if match_found else 0.0)
                except re.error as e:
                    raise ValidationError(f"Pattern matching failed: {e}") from e

            return torch.tensor(satisfactions, device=x.device, dtype=torch.float32)

        except Exception as e:
            # Wrap unexpected errors with context
            raise ValidationError(f"Pattern constraint evaluation failed: {e}") from e


class RangeConstraint(BaseConstraint):
    """Base class for constraints that enforce numeric ranges."""

    min_value: float | None
    max_value: float | None

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        exact_value: float | None = None,
    ):
        if exact_value is not None:
            if min_value is not None or max_value is not None:
                raise ValueError("Cannot specify exact_value with min/max values")
            self.min_value = exact_value
            self.max_value = exact_value
        else:
            self.min_value = min_value
            self.max_value = max_value

        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must be <= max_value")

    @abstractmethod
    def _compute_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the numeric value to constrain."""
        pass

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate range constraint satisfaction."""
        values = self._compute_value(x)

        # Start with full satisfaction
        satisfaction = torch.ones_like(values)

        # Apply min constraint
        if self.min_value is not None:
            min_satisfaction = torch.sigmoid(10.0 * (values - self.min_value))
            satisfaction = torch.min(satisfaction, min_satisfaction)

        # Apply max constraint
        if self.max_value is not None:
            max_satisfaction = torch.sigmoid(10.0 * (self.max_value - values))
            satisfaction = torch.min(satisfaction, max_satisfaction)

        return satisfaction


class CompositionConstraint(BaseConstraint):
    """Base class for constraints on sequence composition (e.g., GC content)."""

    def __init__(self, min_ratio: float = 0.0, max_ratio: float = 1.0):
        if not 0 <= min_ratio <= max_ratio <= 1:
            raise ValueError("Must have 0 <= min_ratio <= max_ratio <= 1")

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    @abstractmethod
    def _compute_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the composition ratio for each sequence."""
        pass

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate composition constraint satisfaction."""
        ratios = self._compute_ratio(x)

        # Satisfaction is 1 when within range, decreases outside
        # in_range = (ratios >= self.min_ratio) & (ratios <= self.max_ratio)  # Unused, kept for clarity

        # For values outside range, compute distance-based satisfaction
        below_min = ratios < self.min_ratio
        above_max = ratios > self.max_ratio

        satisfaction = torch.ones_like(ratios)

        # Penalty for being below minimum
        if below_min.any():
            distance_below = self.min_ratio - ratios[below_min]
            satisfaction[below_min] = torch.exp(-5.0 * distance_below)

        # Penalty for being above maximum
        if above_max.any():
            distance_above = ratios[above_max] - self.max_ratio
            satisfaction[above_max] = torch.exp(-5.0 * distance_above)

        return satisfaction


class StructuralConstraint(BaseConstraint):
    """Base class for constraints on sequence structure (e.g., secondary structure)."""

    def __init__(self, structure_type: str):
        self.structure_type = structure_type

    @abstractmethod
    def _analyze_structure(self, x: torch.Tensor) -> dict[str, Any]:
        """Analyze structural properties of sequences."""
        pass

    @abstractmethod
    def _evaluate_structure(self, structure_info: dict[str, Any]) -> float:
        """Evaluate satisfaction based on structural analysis."""
        pass

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate structural constraint satisfaction."""
        batch_size = x.shape[0] if x.dim() > 1 else 1
        satisfactions = []

        for i in range(batch_size):
            seq = x[i : i + 1] if x.dim() > 1 else x.unsqueeze(0)
            structure_info = self._analyze_structure(seq)
            sat_value = self._evaluate_structure(structure_info)
            satisfactions.append(sat_value)

        return torch.tensor(satisfactions, device=x.device, dtype=torch.float32)


class FunctionalConstraint(BaseConstraint):
    """Base class for constraints based on predicted function or properties."""

    def __init__(self, target_function: str, threshold: float = 0.5):
        self.target_function = target_function
        self.threshold = threshold

    @abstractmethod
    def _predict_function(self, x: torch.Tensor) -> torch.Tensor:
        """Predict functional properties of sequences."""
        pass

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate functional constraint satisfaction."""
        predictions = self._predict_function(x)

        # Convert predictions to satisfaction scores
        # Higher predictions = higher satisfaction
        return torch.sigmoid(5.0 * (predictions - self.threshold))
