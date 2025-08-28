"""Gradient monitoring and stability analysis for constraint optimization."""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class StabilityMetrics:
    """Metrics tracking gradient stability and optimization health."""

    gradient_norm: float = 0.0
    gradient_variance: float = 0.0
    vanishing_ratio: float = 0.0  # Ratio of vanishing gradients
    exploding_ratio: float = 0.0  # Ratio of exploding gradients
    stability_score: float = 1.0  # Overall stability [0,1]
    constraint_balance: float = 1.0  # How balanced constraint gradients are
    effective_constraints: int = 0  # Number of constraints with healthy gradients
    total_constraints: int = 0
    iteration: int = 0

    # Thresholds for gradient health assessment
    vanishing_threshold: float = 1e-8
    exploding_threshold: float = 1e3

    # Historical tracking
    gradient_history: list[float] = field(default_factory=list)
    stability_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/visualization."""
        return {
            "gradient_norm": self.gradient_norm,
            "gradient_variance": self.gradient_variance,
            "vanishing_ratio": self.vanishing_ratio,
            "exploding_ratio": self.exploding_ratio,
            "stability_score": self.stability_score,
            "constraint_balance": self.constraint_balance,
            "effective_constraints": self.effective_constraints,
            "total_constraints": self.total_constraints,
            "iteration": self.iteration,
        }

    @property
    def is_stable(self) -> bool:
        """Check if gradients are stable."""
        return self.stability_score > 0.5

    @property
    def is_vanishing(self) -> bool:
        """Check if gradients are vanishing."""
        return self.vanishing_ratio > 0.5

    @property
    def is_exploding(self) -> bool:
        """Check if gradients are exploding."""
        return self.exploding_ratio > 0.3

    @property
    def needs_intervention(self) -> bool:
        """Check if optimization needs intervention."""
        return not self.is_stable or self.is_vanishing or self.is_exploding


class GradientMonitor:
    """
    Monitor gradient health during constraint optimization.

    Tracks gradient magnitudes, detects vanishing/exploding gradients,
    and provides early warning signals for optimization issues.
    """

    def __init__(
        self,
        constraints: list[Any] | None = None,
        vanishing_threshold: float = 1e-8,
        exploding_threshold: float = 1e3,
        history_window: int = 20,
        enable_hooks: bool = False,
    ):
        """
        Initialize gradient monitor.

        Args:
            constraints: Optional list of constraints to monitor
            vanishing_threshold: Gradient magnitude below which is considered vanishing
            exploding_threshold: Gradient magnitude above which is considered exploding
            history_window: Number of iterations to keep in history
            enable_hooks: Whether to enable automatic gradient hooks
        """
        self.constraints = constraints or []
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.history_window = history_window
        self.enable_hooks = enable_hooks

        # State tracking
        self.metrics = StabilityMetrics(
            vanishing_threshold=vanishing_threshold,
            exploding_threshold=exploding_threshold,
        )
        self.constraint_gradients: dict[int, torch.Tensor] = {}
        self.gradient_hooks: list[Any] = []

        # Statistics tracking
        self.gradient_stats = {
            "min": float("inf"),
            "max": float("-inf"),
            "mean": 0.0,
            "std": 0.0,
        }

        if enable_hooks and constraints:
            self._register_hooks()

    def _register_hooks(self) -> None:
        """Register backward hooks on constraints for automatic monitoring."""
        for i, constraint in enumerate(self.constraints):
            if isinstance(constraint, nn.Module):
                hook = constraint.register_backward_hook(
                    lambda module, grad_input, grad_output, idx=i: self._hook_gradient(
                        idx, grad_output
                    )
                )
                self.gradient_hooks.append(hook)

    def _hook_gradient(
        self, constraint_idx: int, grad_output: tuple[torch.Tensor, ...]
    ) -> None:
        """Hook to capture gradients during backward pass."""
        if grad_output and grad_output[0] is not None:
            self.constraint_gradients[constraint_idx] = grad_output[0].detach()

    def analyze_gradients(
        self,
        gradients: torch.Tensor | list[torch.Tensor],
        constraint_wise: bool = False,
    ) -> StabilityMetrics:
        """
        Analyze gradient health and compute stability metrics.

        Args:
            gradients: Gradients to analyze (can be single tensor or list)
            constraint_wise: Whether gradients are per-constraint

        Returns:
            Updated stability metrics
        """
        # Convert to list if single tensor
        if isinstance(gradients, torch.Tensor):
            grad_list = [gradients]
        else:
            grad_list = gradients

        # Compute gradient statistics
        all_norms = []
        vanishing_count = 0
        exploding_count = 0

        for grad in grad_list:
            if grad is None:
                vanishing_count += 1
                continue

            # Compute gradient norm
            grad_norm = torch.norm(grad.view(-1), p=2).item()
            all_norms.append(grad_norm)

            # Check for vanishing/exploding
            if grad_norm < self.vanishing_threshold:
                vanishing_count += 1
            elif grad_norm > self.exploding_threshold:
                exploding_count += 1

        total_count = len(grad_list)
        valid_norms = [n for n in all_norms if n > 0]

        # Update metrics
        self.metrics.iteration += 1
        self.metrics.total_constraints = total_count
        self.metrics.effective_constraints = len(valid_norms)

        if valid_norms:
            self.metrics.gradient_norm = sum(valid_norms) / len(valid_norms)
            self.metrics.gradient_variance = torch.std(torch.tensor(valid_norms)).item()

            # Track history
            self.metrics.gradient_history.append(self.metrics.gradient_norm)
            if len(self.metrics.gradient_history) > self.history_window:
                self.metrics.gradient_history.pop(0)

            # Update statistics
            self.gradient_stats["min"] = min(
                self.gradient_stats["min"], min(valid_norms)
            )
            self.gradient_stats["max"] = max(
                self.gradient_stats["max"], max(valid_norms)
            )
            self.gradient_stats["mean"] = sum(valid_norms) / len(valid_norms)

            if len(valid_norms) > 1:
                self.gradient_stats["std"] = torch.std(torch.tensor(valid_norms)).item()

        # Compute ratios
        self.metrics.vanishing_ratio = (
            vanishing_count / total_count if total_count > 0 else 0
        )
        self.metrics.exploding_ratio = (
            exploding_count / total_count if total_count > 0 else 0
        )

        # Compute constraint balance (how uniform gradients are across constraints)
        if len(valid_norms) > 1:
            norm_tensor = torch.tensor(valid_norms)
            # Coefficient of variation (lower is better)
            cv = torch.std(norm_tensor) / (torch.mean(norm_tensor) + 1e-8)
            self.metrics.constraint_balance = 1.0 / (1.0 + cv.item())
        else:
            self.metrics.constraint_balance = 1.0 if valid_norms else 0.0

        # Compute overall stability score
        self._compute_stability_score()

        # Track stability history
        self.metrics.stability_history.append(self.metrics.stability_score)
        if len(self.metrics.stability_history) > self.history_window:
            self.metrics.stability_history.pop(0)

        return self.metrics

    def _compute_stability_score(self) -> None:
        """Compute overall stability score based on multiple factors."""
        score = 1.0

        # Penalize vanishing gradients heavily
        score *= (1.0 - self.metrics.vanishing_ratio) ** 2

        # Penalize exploding gradients
        score *= 1.0 - self.metrics.exploding_ratio

        # Reward balanced gradients across constraints
        score *= self.metrics.constraint_balance

        # Penalize if too few effective constraints
        if self.metrics.total_constraints > 0:
            effectiveness = (
                self.metrics.effective_constraints / self.metrics.total_constraints
            )
            score *= effectiveness

        # Check gradient magnitude is in healthy range
        if self.metrics.gradient_norm > 0:
            # Ideal range is [1e-4, 1e2]
            log_norm = torch.log10(torch.tensor(self.metrics.gradient_norm)).item()
            if -4 <= log_norm <= 2:
                norm_score = 1.0
            elif log_norm < -4:
                norm_score = max(0, 1.0 + (log_norm + 4) / 4)  # Gradual penalty
            else:
                norm_score = max(0, 1.0 - (log_norm - 2) / 2)  # Gradual penalty
            score *= norm_score

        self.metrics.stability_score = max(0.0, min(1.0, score))

    def suggest_intervention(self) -> dict[str, Any]:
        """
        Suggest interventions based on current gradient health.

        Returns:
            Dictionary with intervention suggestions
        """
        suggestions = {
            "needs_intervention": self.metrics.needs_intervention,
            "actions": [],
            "urgency": "none",
        }

        if self.metrics.is_vanishing:
            suggestions["actions"].append("increase_learning_rate")
            suggestions["actions"].append("reduce_weight_decay")
            suggestions["actions"].append("switch_to_lukasiewicz_tnorm")
            suggestions["urgency"] = "high"

        if self.metrics.is_exploding:
            suggestions["actions"].append("decrease_learning_rate")
            suggestions["actions"].append("add_gradient_clipping")
            suggestions["actions"].append("increase_weight_decay")
            suggestions["urgency"] = "high"

        if self.metrics.constraint_balance < 0.5:
            suggestions["actions"].append("rebalance_constraint_weights")
            suggestions["actions"].append("use_adaptive_weighting")
            if suggestions["urgency"] == "none":
                suggestions["urgency"] = "medium"

        if self.metrics.effective_constraints < self.metrics.total_constraints * 0.5:
            suggestions["actions"].append("relax_inactive_constraints")
            suggestions["actions"].append("check_constraint_compatibility")
            if suggestions["urgency"] == "none":
                suggestions["urgency"] = "medium"

        return suggestions

    def get_gradient_summary(self) -> str:
        """Get human-readable summary of gradient health."""
        lines = [
            f"Gradient Health Report (Iteration {self.metrics.iteration})",
            "=" * 50,
            f"Stability Score: {self.metrics.stability_score:.3f} {'✅' if self.metrics.is_stable else '⚠️'}",
            f"Gradient Norm: {self.metrics.gradient_norm:.2e}",
            f"Effective Constraints: {self.metrics.effective_constraints}/{self.metrics.total_constraints}",
            f"Constraint Balance: {self.metrics.constraint_balance:.3f}",
        ]

        if self.metrics.is_vanishing:
            lines.append(
                f"⚠️  WARNING: {self.metrics.vanishing_ratio:.1%} gradients vanishing!"
            )

        if self.metrics.is_exploding:
            lines.append(
                f"⚠️  WARNING: {self.metrics.exploding_ratio:.1%} gradients exploding!"
            )

        if self.metrics.gradient_history:
            recent_trend = self._compute_trend()
            if recent_trend < -0.5:
                lines.append("📉 Gradient magnitude decreasing rapidly")
            elif recent_trend > 0.5:
                lines.append("📈 Gradient magnitude increasing rapidly")
            else:
                lines.append("➡️  Gradient magnitude stable")

        return "\n".join(lines)

    def _compute_trend(self) -> float:
        """Compute trend in gradient magnitude over recent history."""
        if len(self.metrics.gradient_history) < 3:
            return 0.0

        recent = self.metrics.gradient_history[-5:]
        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        x = torch.arange(len(recent), dtype=torch.float32)
        y = torch.tensor(recent, dtype=torch.float32)

        # Normalize to log scale for better trend detection
        y_log = torch.log10(y + 1e-10)

        # Compute trend coefficient
        x_mean = x.mean()
        y_mean = y_log.mean()

        numerator = ((x - x_mean) * (y_log - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if denominator > 0:
            trend = numerator / denominator
            return trend.item()
        return 0.0

    def reset(self) -> None:
        """Reset monitoring state."""
        self.metrics = StabilityMetrics(
            vanishing_threshold=self.vanishing_threshold,
            exploding_threshold=self.exploding_threshold,
        )
        self.constraint_gradients.clear()

        # Remove hooks if any
        for hook in self.gradient_hooks:
            hook.remove()
        self.gradient_hooks.clear()

        # Re-register if needed
        if self.enable_hooks and self.constraints:
            self._register_hooks()
