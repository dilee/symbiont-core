"""Triangular norms (t-norms) for fuzzy logic operations."""

from collections.abc import Callable

import torch

from symbiont.core.types import TNormType


class TNorm:
    """
    Triangular norms for fuzzy logic operations.

    T-norms are used to generalize logical AND operations to continuous values.
    They provide different ways to combine constraint satisfactions.
    """

    @staticmethod
    def product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Product t-norm: T(a,b) = a * b

        Most commonly used t-norm. Provides smooth gradients and
        natural interpretation as probabilistic independence.

        Args:
            a: First operand tensor [0,1]
            b: Second operand tensor [0,1]

        Returns:
            Element-wise product of a and b
        """
        return a * b

    @staticmethod
    def lukasiewicz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Łukasiewicz t-norm: T(a,b) = max(0, a + b - 1)

        Provides strong penalty when constraints are not fully satisfied.
        Good for strict logical reasoning.

        Args:
            a: First operand tensor [0,1]
            b: Second operand tensor [0,1]

        Returns:
            Łukasiewicz t-norm of a and b
        """
        return torch.clamp(a + b - 1.0, min=0.0)

    @staticmethod
    def godel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Gödel t-norm: T(a,b) = min(a, b)

        Takes the minimum satisfaction. Conservative approach
        where the weakest constraint dominates.

        Args:
            a: First operand tensor [0,1]
            b: Second operand tensor [0,1]

        Returns:
            Element-wise minimum of a and b
        """
        return torch.min(a, b)

    @staticmethod
    def drastic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Drastic t-norm: T(a,b) = min(a,b) if max(a,b) = 1, else 0

        Very strict t-norm that only gives positive results when
        at least one constraint is fully satisfied.

        Args:
            a: First operand tensor [0,1]
            b: Second operand tensor [0,1]

        Returns:
            Drastic t-norm of a and b
        """
        max_val = torch.max(a, b)
        min_val = torch.min(a, b)
        return torch.where(
            torch.isclose(max_val, torch.ones_like(max_val), atol=1e-6),
            min_val,
            torch.zeros_like(min_val),
        )

    @classmethod
    def get_tnorm(
        cls, tnorm_type: TNormType
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Get t-norm function by name.

        Args:
            tnorm_type: Name of the t-norm to retrieve

        Returns:
            T-norm function

        Raises:
            ValueError: If tnorm_type is not recognized
        """
        tnorm_map = {
            "product": cls.product,
            "lukasiewicz": cls.lukasiewicz,
            "godel": cls.godel,
            "drastic": cls.drastic,
        }

        if tnorm_type not in tnorm_map:
            available = list(tnorm_map.keys())
            raise ValueError(f"Unknown t-norm: {tnorm_type}. Available: {available}")

        return tnorm_map[tnorm_type]


class SoftTNorm:
    """Soft/parameterized versions of t-norms for better gradient flow."""

    @staticmethod
    def soft_min(
        a: torch.Tensor, b: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Soft minimum using LogSumExp trick.

        Approximates min(a,b) with smooth gradients. As temperature approaches 0,
        this converges to the true minimum.

        Args:
            a: First operand tensor [0,1]
            b: Second operand tensor [0,1]
            temperature: Softness parameter (lower = closer to true min)

        Returns:
            Soft minimum of a and b
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        # Convert to log space for numerical stability
        # soft_min(a,b) ≈ -log(exp(-a/T) + exp(-b/T)) * T
        neg_a = -a / temperature
        neg_b = -b / temperature

        # Use logsumexp for numerical stability
        log_sum = torch.logsumexp(torch.stack([neg_a, neg_b], dim=-1), dim=-1)
        return -log_sum * temperature

    @staticmethod
    def soft_product(
        a: torch.Tensor, b: torch.Tensor, sharpness: float = 1.0
    ) -> torch.Tensor:
        """
        Sharpened product t-norm.

        T(a,b) = (a * b)^sharpness

        Higher sharpness makes the product more sensitive to low values,
        approaching the Gödel t-norm as sharpness increases.

        Args:
            a: First operand tensor [0,1]
            b: Second operand tensor [0,1]
            sharpness: Sharpness parameter (higher = more sharp)

        Returns:
            Sharpened product of a and b
        """
        if sharpness <= 0:
            raise ValueError("Sharpness must be positive")

        product = a * b
        return torch.pow(product, sharpness)

    @staticmethod
    def einstein_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Einstein t-norm: T(a,b) = (a * b) / (2 - (a + b - a * b))

        Provides a compromise between product and Łukasiewicz t-norms.

        Args:
            a: First operand tensor [0,1]
            b: Second operand tensor [0,1]

        Returns:
            Einstein product of a and b
        """
        numerator = a * b
        denominator = 2.0 - (a + b - a * b)

        # Handle division by zero
        denominator = torch.clamp(denominator, min=1e-8)

        return torch.div(numerator, denominator)


class TNormCombination:
    """Utilities for combining multiple constraints using t-norms."""

    @staticmethod
    def combine_constraints(
        satisfactions: list[torch.Tensor], tnorm_type: TNormType = "product"
    ) -> torch.Tensor:
        """
        Combine multiple constraint satisfactions using specified t-norm.

        Args:
            satisfactions: List of satisfaction tensors to combine
            tnorm_type: Type of t-norm to use for combination

        Returns:
            Combined satisfaction score

        Raises:
            ValueError: If satisfactions list is empty
        """
        if not satisfactions:
            raise ValueError("Cannot combine empty list of satisfactions")

        if len(satisfactions) == 1:
            return satisfactions[0]

        tnorm_func = TNorm.get_tnorm(tnorm_type)

        # Combine pairwise
        result = satisfactions[0]
        for satisfaction in satisfactions[1:]:
            result = tnorm_func(result, satisfaction)

        return result

    @staticmethod
    def weighted_combination(
        satisfactions: list[torch.Tensor],
        weights: list[float],
        tnorm_type: TNormType = "product",
    ) -> torch.Tensor:
        """
        Combine satisfactions with different weights using t-norm.

        Args:
            satisfactions: List of satisfaction tensors
            weights: List of weights for each satisfaction
            tnorm_type: Type of t-norm to use

        Returns:
            Weighted combined satisfaction score

        Raises:
            ValueError: If lengths don't match or lists are empty
        """
        if len(satisfactions) != len(weights):
            raise ValueError("Satisfactions and weights must have same length")

        if not satisfactions:
            raise ValueError("Cannot combine empty lists")

        # Apply weights to satisfactions
        weighted_satisfactions = [
            torch.clamp(weight * sat, 0.0, 1.0)
            for sat, weight in zip(satisfactions, weights, strict=False)
        ]

        return TNormCombination.combine_constraints(weighted_satisfactions, tnorm_type)
