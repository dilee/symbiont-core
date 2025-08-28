"""Adaptive weight management for constraint optimization stability."""

from typing import Any

import torch

from symbiont.optimization.monitor import StabilityMetrics


class AdaptiveWeightManager:
    """
    Dynamically adjust constraint weights to maintain gradient stability.

    Uses gradient health metrics to rebalance weights, preventing gradient
    vanishing/exploding and ensuring all constraints contribute effectively
    to optimization.
    """

    def __init__(
        self,
        initial_weights: list[float] | torch.Tensor | None = None,
        num_constraints: int | None = None,
        adaptation_rate: float = 0.1,
        temperature: float = 1.0,
        min_weight: float = 0.01,
        max_weight: float = 100.0,
        balance_factor: float = 0.5,
        momentum: float = 0.9,
    ):
        """
        Initialize adaptive weight manager.

        Args:
            initial_weights: Initial constraint weights
            num_constraints: Number of constraints (if weights not provided)
            adaptation_rate: Learning rate for weight updates (0-1)
            temperature: Temperature for weight smoothing
            min_weight: Minimum allowed weight value
            max_weight: Maximum allowed weight value
            balance_factor: How much to prioritize balance vs individual performance
            momentum: Momentum for weight updates
        """
        # Initialize weights
        if initial_weights is not None:
            if isinstance(initial_weights, list):
                self.weights = torch.tensor(initial_weights, dtype=torch.float32)
            else:
                self.weights = initial_weights.clone()
            self.num_constraints = len(self.weights)
        elif num_constraints is not None:
            self.num_constraints = num_constraints
            self.weights = torch.ones(num_constraints, dtype=torch.float32)
        else:
            raise ValueError("Must provide either initial_weights or num_constraints")

        # Adaptation parameters
        self.adaptation_rate = adaptation_rate
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.balance_factor = balance_factor
        self.momentum = momentum

        # State tracking
        self.weight_history = [self.weights.clone()]
        self.weight_momentum = torch.zeros_like(self.weights)
        self.constraint_importance = torch.ones_like(self.weights)
        self.constraint_gradients: list[torch.Tensor] = []
        self.iteration = 0

        # Adaptation strategies
        self.strategies = {
            "gradient_magnitude": self._adapt_by_gradient_magnitude,
            "satisfaction_balance": self._adapt_by_satisfaction_balance,
            "gradient_variance": self._adapt_by_gradient_variance,
            "contribution_tracking": self._adapt_by_contribution,
        }

    def adapt(
        self,
        metrics: StabilityMetrics,
        constraint_gradients: list[torch.Tensor] | None = None,
        constraint_satisfactions: torch.Tensor | None = None,
        strategy: str = "gradient_magnitude",
    ) -> torch.Tensor:
        """
        Adapt weights based on gradient health metrics.

        Args:
            metrics: Current stability metrics from GradientMonitor
            constraint_gradients: Optional per-constraint gradients
            constraint_satisfactions: Optional current satisfaction scores
            strategy: Adaptation strategy to use

        Returns:
            Updated weights tensor
        """
        self.iteration += 1

        # Store gradients if provided
        if constraint_gradients is not None:
            self.constraint_gradients = constraint_gradients

        # Apply selected strategy
        if strategy in self.strategies:
            weight_update = self.strategies[strategy](
                metrics, constraint_gradients, constraint_satisfactions
            )
        else:
            # Default: combine multiple strategies
            weight_update = self._combined_adaptation(
                metrics, constraint_gradients, constraint_satisfactions
            )

        # Apply momentum
        self.weight_momentum = (
            self.momentum * self.weight_momentum + (1 - self.momentum) * weight_update
        )

        # Update weights with learning rate
        self.weights = self.weights + self.adaptation_rate * self.weight_momentum

        # Apply temperature scaling
        if self.temperature != 1.0:
            self.weights = self._apply_temperature(self.weights)

        # Clamp to valid range
        self.weights = torch.clamp(self.weights, self.min_weight, self.max_weight)

        # Normalize if needed (optional)
        # self.weights = self.weights / self.weights.sum() * self.num_constraints

        # Track history
        self.weight_history.append(self.weights.clone())
        if len(self.weight_history) > 100:
            self.weight_history.pop(0)

        return self.weights

    def _adapt_by_gradient_magnitude(
        self,
        metrics: StabilityMetrics,
        constraint_gradients: list[torch.Tensor] | None,
        constraint_satisfactions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Adapt weights based on gradient magnitudes."""
        if constraint_gradients is None or len(constraint_gradients) == 0:
            return torch.zeros_like(self.weights)

        # Compute gradient norms for each constraint
        grad_norms = []
        for grad in constraint_gradients:
            if grad is not None:
                norm = torch.norm(grad.view(-1), p=2).item()
            else:
                norm = 0.0
            grad_norms.append(norm)

        grad_norms = torch.tensor(grad_norms, dtype=torch.float32)

        # Target: balanced gradient magnitudes
        if grad_norms.sum() > 0:
            target_norm = grad_norms[grad_norms > 0].mean()
        else:
            target_norm = 1.0

        # Compute weight updates
        weight_update = torch.zeros_like(self.weights)

        for i, norm in enumerate(grad_norms):
            if norm < metrics.vanishing_threshold:
                # Gradient vanishing - increase weight
                weight_update[i] = self.weights[i] * 0.5
            elif norm > metrics.exploding_threshold:
                # Gradient exploding - decrease weight
                weight_update[i] = -self.weights[i] * 0.3
            else:
                # Move toward target norm
                ratio = target_norm / (norm + 1e-8)
                if ratio > 1.5:
                    weight_update[i] = self.weights[i] * 0.2
                elif ratio < 0.67:
                    weight_update[i] = -self.weights[i] * 0.2

        return weight_update

    def _adapt_by_satisfaction_balance(
        self,
        metrics: StabilityMetrics,
        constraint_gradients: list[torch.Tensor] | None,
        constraint_satisfactions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Adapt weights to balance constraint satisfactions."""
        if constraint_satisfactions is None:
            return torch.zeros_like(self.weights)

        # Increase weights for poorly satisfied constraints
        mean_satisfaction = constraint_satisfactions.mean()
        weight_update = torch.zeros_like(self.weights)

        for i, satisfaction in enumerate(constraint_satisfactions):
            if satisfaction < mean_satisfaction - 0.2:
                # Under-satisfied - increase weight
                weight_update[i] = self.weights[i] * 0.3
            elif satisfaction > mean_satisfaction + 0.3:
                # Over-satisfied - can reduce weight
                weight_update[i] = -self.weights[i] * 0.1

        return weight_update

    def _adapt_by_gradient_variance(
        self,
        metrics: StabilityMetrics,
        constraint_gradients: list[torch.Tensor] | None,
        constraint_satisfactions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Adapt weights to reduce gradient variance across constraints."""
        if constraint_gradients is None or len(constraint_gradients) < 2:
            return torch.zeros_like(self.weights)

        # Compute gradient statistics
        grad_norms = []
        for grad in constraint_gradients:
            if grad is not None:
                norm = torch.norm(grad.view(-1), p=2).item()
                grad_norms.append(norm)
            else:
                grad_norms.append(0.0)

        grad_norms = torch.tensor(grad_norms, dtype=torch.float32)

        if grad_norms.sum() == 0:
            return torch.zeros_like(self.weights)

        # Compute deviation from mean
        mean_norm = grad_norms.mean()
        std_norm = grad_norms.std()

        if std_norm < 1e-8:
            return torch.zeros_like(self.weights)

        # Normalize deviations
        z_scores = (grad_norms - mean_norm) / (std_norm + 1e-8)

        # Adjust weights to reduce variance
        weight_update = -self.balance_factor * z_scores * self.weights

        return weight_update

    def _adapt_by_contribution(
        self,
        metrics: StabilityMetrics,
        constraint_gradients: list[torch.Tensor] | None,
        constraint_satisfactions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Adapt weights based on constraint contribution to optimization."""
        if constraint_gradients is None:
            return torch.zeros_like(self.weights)

        # Track which constraints are contributing
        contributions = torch.zeros(self.num_constraints)

        for i, grad in enumerate(constraint_gradients):
            if grad is not None:
                # Contribution = gradient magnitude * (1 - satisfaction)
                norm = torch.norm(grad.view(-1), p=2).item()
                if constraint_satisfactions is not None:
                    contribution = norm * (1.0 - constraint_satisfactions[i].item())
                else:
                    contribution = norm
                contributions[i] = contribution

        # Update importance scores
        self.constraint_importance = (
            0.9 * self.constraint_importance + 0.1 * contributions
        )

        # Adjust weights based on importance
        mean_importance = self.constraint_importance.mean()
        weight_update = torch.zeros_like(self.weights)

        for i in range(self.num_constraints):
            if self.constraint_importance[i] < mean_importance * 0.1:
                # Very low contribution - might be satisfied or incompatible
                weight_update[i] = -self.weights[i] * 0.2
            elif self.constraint_importance[i] > mean_importance * 2:
                # High contribution - important for optimization
                weight_update[i] = self.weights[i] * 0.1

        return weight_update

    def _combined_adaptation(
        self,
        metrics: StabilityMetrics,
        constraint_gradients: list[torch.Tensor] | None,
        constraint_satisfactions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine multiple adaptation strategies."""
        weight_update = torch.zeros_like(self.weights)

        # Weighted combination of strategies
        if metrics.is_vanishing:
            # Focus on gradient magnitude when vanishing
            update = self._adapt_by_gradient_magnitude(
                metrics, constraint_gradients, constraint_satisfactions
            )
            weight_update += 0.6 * update

        if metrics.constraint_balance < 0.7:
            # Focus on variance reduction when imbalanced
            update = self._adapt_by_gradient_variance(
                metrics, constraint_gradients, constraint_satisfactions
            )
            weight_update += 0.3 * update

        # Always consider satisfaction balance
        update = self._adapt_by_satisfaction_balance(
            metrics, constraint_gradients, constraint_satisfactions
        )
        weight_update += 0.2 * update

        # Consider contribution tracking
        update = self._adapt_by_contribution(
            metrics, constraint_gradients, constraint_satisfactions
        )
        weight_update += 0.2 * update

        return weight_update

    def _apply_temperature(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to weights."""
        if self.temperature == 0:
            # Hard selection - only highest weight
            max_idx = torch.argmax(weights)
            result = torch.zeros_like(weights)
            result[max_idx] = weights.sum()
            return result

        # Softmax-style temperature scaling
        scaled = weights / self.temperature
        # Normalize to maintain sum
        return scaled / scaled.mean() * weights.mean()

    def get_weight_summary(self) -> str:
        """Get human-readable summary of weight adaptation."""
        lines = [
            f"Adaptive Weight Status (Iteration {self.iteration})",
            "=" * 50,
        ]

        for i, weight in enumerate(self.weights):
            change = 0.0
            if len(self.weight_history) > 1:
                prev_weight = self.weight_history[-2][i]
                change = (weight - prev_weight) / (prev_weight + 1e-8) * 100

            importance = self.constraint_importance[i].item()

            status = ""
            if abs(change) > 10:
                status = "📈" if change > 0 else "📉"

            lines.append(
                f"Constraint {i}: {weight:.3f} ({change:+.1f}%) "
                f"Importance: {importance:.3f} {status}"
            )

        # Summary statistics
        lines.append("-" * 50)
        lines.append(f"Mean Weight: {self.weights.mean():.3f}")
        lines.append(f"Weight Std: {self.weights.std():.3f}")
        lines.append(f"Min/Max: {self.weights.min():.3f}/{self.weights.max():.3f}")

        return "\n".join(lines)

    def reset(self, keep_weights: bool = False) -> None:
        """
        Reset adaptation state.

        Args:
            keep_weights: Whether to keep current weights or reset to initial
        """
        if not keep_weights:
            self.weights = torch.ones(self.num_constraints, dtype=torch.float32)

        self.weight_history = [self.weights.clone()]
        self.weight_momentum = torch.zeros_like(self.weights)
        self.constraint_importance = torch.ones_like(self.weights)
        self.constraint_gradients = []
        self.iteration = 0

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for saving/loading."""
        return {
            "weights": self.weights,
            "weight_momentum": self.weight_momentum,
            "constraint_importance": self.constraint_importance,
            "iteration": self.iteration,
            "adaptation_rate": self.adaptation_rate,
            "temperature": self.temperature,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load state from dictionary."""
        self.weights = state["weights"]
        self.weight_momentum = state["weight_momentum"]
        self.constraint_importance = state["constraint_importance"]
        self.iteration = state["iteration"]
        self.adaptation_rate = state.get("adaptation_rate", self.adaptation_rate)
        self.temperature = state.get("temperature", self.temperature)
