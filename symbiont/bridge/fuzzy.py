"""Fuzzy logic operations for continuous constraint satisfaction with gradient optimization."""

import torch

from symbiont.bridge.tnorms import TNorm
from symbiont.core.types import TNormType


class GradientOptimizedOperations:
    """Gradient-optimized fuzzy operations for better backpropagation."""

    @staticmethod
    def smooth_step(
        x: torch.Tensor, threshold: float = 0.5, smoothness: float = 10.0
    ) -> torch.Tensor:
        """Smooth step function with better gradient flow than hard thresholding."""
        return torch.sigmoid(smoothness * (x - threshold))

    @staticmethod
    def soft_clamp(
        x: torch.Tensor,
        min_val: float = 0.0,
        max_val: float = 1.0,
        smoothness: float = 10.0,
    ) -> torch.Tensor:
        """Soft clamping with continuous gradients."""
        # Use smooth approximation of clamp for better gradient flow
        x_scaled = smoothness * (x - (min_val + max_val) / 2)
        return (max_val - min_val) * torch.sigmoid(x_scaled) + min_val

    @staticmethod
    def differentiable_count(
        scores: torch.Tensor, threshold: float = 0.5, smoothness: float = 10.0
    ) -> torch.Tensor:
        """Differentiable approximation of counting satisfied constraints."""
        # Replace hard counting with soft counting for gradient flow
        return GradientOptimizedOperations.smooth_step(
            scores, threshold, smoothness
        ).sum(dim=0)


class FuzzyOperations:
    """
    Fuzzy logic operations for constraint evaluation.

    Provides continuous generalizations of Boolean logic operations
    that maintain differentiability for gradient-based optimization.
    """

    @staticmethod
    def fuzzy_and(
        a: torch.Tensor, b: torch.Tensor, tnorm: TNormType = "product"
    ) -> torch.Tensor:
        """
        Fuzzy logical AND using specified t-norm.

        Args:
            a: First operand [0,1]
            b: Second operand [0,1]
            tnorm: T-norm type to use

        Returns:
            Fuzzy AND result
        """
        tnorm_func = TNorm.get_tnorm(tnorm)
        return tnorm_func(a, b)

    @staticmethod
    def fuzzy_or(
        a: torch.Tensor, b: torch.Tensor, snorm: str = "probabilistic"
    ) -> torch.Tensor:
        """
        Fuzzy logical OR using specified s-norm (t-conorm).

        Args:
            a: First operand [0,1]
            b: Second operand [0,1]
            snorm: S-norm type ("max", "probabilistic", "lukasiewicz")

        Returns:
            Fuzzy OR result
        """
        if snorm == "max":
            return torch.max(a, b)
        elif snorm == "probabilistic":
            return a + b - a * b
        elif snorm == "lukasiewicz":
            return torch.clamp(a + b, max=1.0)
        else:
            raise ValueError(f"Unknown s-norm: {snorm}")

    @staticmethod
    def fuzzy_not(a: torch.Tensor) -> torch.Tensor:
        """
        Fuzzy logical NOT (standard negation).

        Args:
            a: Input tensor [0,1]

        Returns:
            1 - a (fuzzy negation)
        """
        return torch.tensor(1.0, device=a.device, dtype=a.dtype) - a

    @staticmethod
    def fuzzy_implies(
        a: torch.Tensor, b: torch.Tensor, impl_type: str = "lukasiewicz"
    ) -> torch.Tensor:
        """
        Fuzzy implication: a → b

        Args:
            a: Antecedent [0,1]
            b: Consequent [0,1]
            impl_type: Implication type ("lukasiewicz", "godel", "goguen")

        Returns:
            Fuzzy implication result
        """
        if impl_type == "lukasiewicz":
            return torch.clamp(1.0 - a + b, max=1.0)
        elif impl_type == "godel":
            return torch.where(a <= b, torch.ones_like(a), b)
        elif impl_type == "goguen":
            return torch.where(a <= b, torch.ones_like(a), b / torch.clamp(a, min=1e-8))
        else:
            raise ValueError(f"Unknown implication type: {impl_type}")


class FuzzyMembership:
    """Fuzzy membership functions for constraint satisfaction."""

    @staticmethod
    def triangular(x: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
        """
        Triangular membership function.

        Returns 0 outside [a,c], peak of 1 at b, linear ramps between.

        Args:
            x: Input values
            a: Left boundary
            b: Peak point
            c: Right boundary

        Returns:
            Membership values [0,1]
        """
        if not (a <= b <= c):
            raise ValueError("Must have a <= b <= c")

        left_ramp = torch.clamp((x - a) / (b - a), 0.0, 1.0)
        right_ramp = torch.clamp((c - x) / (c - b), 0.0, 1.0)

        return torch.min(left_ramp, right_ramp)

    @staticmethod
    def trapezoidal(
        x: torch.Tensor, a: float, b: float, c: float, d: float
    ) -> torch.Tensor:
        """
        Trapezoidal membership function.

        Returns 0 outside [a,d], 1 in [b,c], linear ramps at edges.

        Args:
            x: Input values
            a: Left boundary
            b: Left plateau start
            c: Right plateau end
            d: Right boundary

        Returns:
            Membership values [0,1]
        """
        if not (a <= b <= c <= d):
            raise ValueError("Must have a <= b <= c <= d")

        # Create the trapezoidal shape by combining regions
        result = torch.zeros_like(x)

        # Left ramp: linear from 0 to 1 between a and b
        left_mask = (a <= x) & (x <= b)
        if b != a:  # Avoid division by zero
            result = torch.where(left_mask, (x - a) / (b - a), result)

        # Plateau: value 1 between b and c
        plateau_mask = (b <= x) & (x <= c)
        result = torch.where(plateau_mask, torch.ones_like(x), result)

        # Right ramp: linear from 1 to 0 between c and d
        right_mask = (c <= x) & (x <= d)
        if d != c:  # Avoid division by zero
            result = torch.where(right_mask, (d - x) / (d - c), result)

        return result

    @staticmethod
    def gaussian(x: torch.Tensor, center: float, width: float) -> torch.Tensor:
        """
        Gaussian membership function.

        Args:
            x: Input values
            center: Center of the Gaussian
            width: Width parameter (higher = wider)

        Returns:
            Membership values [0,1]
        """
        if width <= 0:
            raise ValueError("Width must be positive")

        return torch.exp(-0.5 * ((x - center) / width) ** 2)

    @staticmethod
    def sigmoid(x: torch.Tensor, center: float, slope: float = 1.0) -> torch.Tensor:
        """
        Sigmoid membership function.

        Args:
            x: Input values
            center: Inflection point
            slope: Steepness of the sigmoid

        Returns:
            Membership values [0,1]
        """
        return torch.sigmoid(slope * (x - center))


class FuzzyRules:
    """Fuzzy rule evaluation system."""

    def __init__(self, tnorm: TNormType = "product"):
        self.tnorm = tnorm

    def if_then(
        self, antecedent: torch.Tensor, consequent: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate fuzzy if-then rule using min operation (Mamdani inference).

        Args:
            antecedent: Condition satisfaction [0,1]
            consequent: Consequent membership [0,1]

        Returns:
            Rule activation strength
        """
        return torch.min(antecedent, consequent)

    def aggregate_rules(self, rule_outputs: list[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate multiple fuzzy rule outputs using max operation.

        Args:
            rule_outputs: List of rule activation strengths

        Returns:
            Aggregated output
        """
        if not rule_outputs:
            raise ValueError("Cannot aggregate empty rule outputs")

        return torch.stack(rule_outputs, dim=0).max(dim=0)[0]

    def defuzzify_centroid(
        self, membership: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Defuzzify using centroid method.

        Args:
            membership: Membership degrees [0,1]
            values: Corresponding values

        Returns:
            Crisp output value
        """
        numerator = (membership * values).sum()
        denominator = membership.sum()

        return numerator / torch.clamp(denominator, min=1e-8)
