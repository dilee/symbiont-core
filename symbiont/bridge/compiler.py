"""Constraint compiler that converts symbolic constraints to differentiable operations."""

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from symbiont.bridge.fuzzy import FuzzyOperations
from symbiont.bridge.tnorms import TNorm, TNormCombination
from symbiont.core.types import TNormType
from symbiont.optimization.adaptive import AdaptiveWeightManager
from symbiont.optimization.monitor import GradientMonitor, StabilityMetrics

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from symbiont.core.constraints import (
        AndConstraint,
        Constraint,
        NotConstraint,
        OrConstraint,
        WeightedConstraint,
    )
    from symbiont.core.dsl import Rules
else:
    # Import at runtime when needed to avoid circular dependencies
    Constraint = "Constraint"
    AndConstraint = "AndConstraint"
    OrConstraint = "OrConstraint"
    NotConstraint = "NotConstraint"
    WeightedConstraint = "WeightedConstraint"


class DifferentiableConstraint(nn.Module):
    """
    Differentiable wrapper for constraints that can be used in loss functions.

    Converts symbolic constraints into PyTorch operations that support
    automatic differentiation.
    """

    def __init__(self, constraint: "Constraint", tnorm: TNormType = "product"):
        super().__init__()
        self.constraint = constraint
        self.tnorm = tnorm
        self.tnorm_func = TNorm.get_tnorm(tnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate constraint satisfaction in a differentiable manner.

        Args:
            x: Input tensor to evaluate

        Returns:
            Satisfaction scores [0,1]
        """
        return self._evaluate_constraint(self.constraint, x)

    def _evaluate_constraint(
        self, constraint: "Constraint", x: torch.Tensor
    ) -> torch.Tensor:
        """Recursively evaluate constraint tree."""

        # Import constraint types at runtime to avoid circular dependencies
        from symbiont.core.constraints import (
            AndConstraint,
            NotConstraint,
            OrConstraint,
            WeightedConstraint,
        )

        # Handle composite constraints
        if isinstance(constraint, AndConstraint):
            scores = [self._evaluate_constraint(c, x) for c in constraint.constraints]
            return TNormCombination.combine_constraints(scores, self.tnorm)

        elif isinstance(constraint, OrConstraint):
            scores = [self._evaluate_constraint(c, x) for c in constraint.constraints]
            # Use fuzzy OR (s-norm) instead of t-norm for disjunction
            result = scores[0]
            for score in scores[1:]:
                result = FuzzyOperations.fuzzy_or(result, score)
            return result

        elif isinstance(constraint, NotConstraint):
            inner_score = self._evaluate_constraint(constraint.constraint, x)
            return FuzzyOperations.fuzzy_not(inner_score)

        elif isinstance(constraint, WeightedConstraint):
            base_score = self._evaluate_constraint(constraint.constraint, x)
            # Apply weight and clamp to [0,1]
            return torch.clamp(constraint.weight * base_score, 0.0, 1.0)

        else:
            # Base constraint - call its satisfaction method
            return constraint.satisfaction(x)


class DifferentiableCompiler:
    """
    Compiler that converts constraint systems to differentiable loss functions.

    Takes a set of symbolic constraints and creates a PyTorch module
    that can be used for gradient-based optimization.
    """

    def __init__(self, tnorm: TNormType = "product"):
        self.tnorm = tnorm
        self.compiled_constraints: dict[str, DifferentiableConstraint] = {}

    def compile(
        self,
        constraints: list["Constraint"],
        loss_type: str = "violation",
        enable_monitoring: bool = False,
        enable_adaptive_weights: bool = False,
    ) -> "ConstraintLoss":
        """
        Compile constraints into a differentiable loss function.

        Args:
            constraints: List of constraints to compile
            loss_type: Type of loss ("violation", "satisfaction", "weighted")
            enable_monitoring: Enable gradient health monitoring
            enable_adaptive_weights: Enable adaptive weight management

        Returns:
            Compiled constraint loss function with optional stability features
        """
        if not constraints:
            raise ValueError("Cannot compile empty constraint list")

        differentiable_constraints = [
            DifferentiableConstraint(c, self.tnorm) for c in constraints
        ]

        return ConstraintLoss(
            differentiable_constraints,
            loss_type,
            enable_monitoring=enable_monitoring,
            enable_adaptive_weights=enable_adaptive_weights,
        )

    def compile_rules(self, rules: "Rules") -> "ConstraintLoss":
        """
        Compile a Rules object into a differentiable loss function.

        Args:
            rules: Rules object containing constraints

        Returns:
            Compiled constraint loss function
        """
        return self.compile(rules.constraints, loss_type="weighted")


class ConstraintLoss(nn.Module):
    """
    Differentiable loss function based on constraint satisfaction.

    Provides various loss formulations for different optimization objectives
    with support for adaptive weight management and gradient monitoring.
    """

    def __init__(
        self,
        constraints: list[DifferentiableConstraint],
        loss_type: str = "violation",
        enable_monitoring: bool = False,
        enable_adaptive_weights: bool = False,
    ):
        super().__init__()
        self.constraints = nn.ModuleList(constraints)
        self.loss_type = loss_type
        self.enable_monitoring = enable_monitoring
        self.enable_adaptive_weights = enable_adaptive_weights

        if loss_type not in ["violation", "satisfaction", "weighted"]:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Initialize monitoring and adaptation if enabled
        self.gradient_monitor: GradientMonitor | None = None
        self.weight_manager: AdaptiveWeightManager | None = None
        self.adaptive_weights: torch.Tensor | None = None

        if enable_monitoring:
            self.gradient_monitor = GradientMonitor(
                constraints=list(self.constraints), enable_hooks=True
            )

        if enable_adaptive_weights:
            self.weight_manager = AdaptiveWeightManager(
                num_constraints=len(self.constraints),
                adaptation_rate=0.1,
                temperature=1.0,
            )
            self.adaptive_weights = self.weight_manager.weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute constraint-based loss for input.

        Args:
            x: Input tensor (batch_size, ...)

        Returns:
            Loss tensor (scalar or per-sample)
        """
        if not self.constraints:
            return torch.zeros(x.shape[0] if x.dim() > 0 else 1, device=x.device)

        # Evaluate all constraints
        satisfactions = [constraint(x) for constraint in self.constraints]

        # Apply adaptive weights if enabled
        if self.enable_adaptive_weights and self.adaptive_weights is not None:
            weighted_satisfactions = []
            for i, sat in enumerate(satisfactions):
                weighted_satisfactions.append(self.adaptive_weights[i] * sat)
            satisfactions = weighted_satisfactions

        if self.loss_type == "violation":
            # Loss = 1 - satisfaction (minimize violation)
            violations = [1.0 - s for s in satisfactions]
            return torch.stack(violations, dim=0).mean(dim=0)

        elif self.loss_type == "satisfaction":
            # Loss = -satisfaction (maximize satisfaction)
            return -torch.stack(satisfactions, dim=0).mean(dim=0)

        elif self.loss_type == "weighted":
            # Weighted combination based on constraint weights
            total_satisfaction = torch.stack(satisfactions, dim=0).mean(dim=0)
            return torch.tensor(1.0, device=x.device) - total_satisfaction

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def satisfaction_scores(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Get detailed satisfaction scores for analysis.

        Args:
            x: Input tensor

        Returns:
            Dictionary with individual and aggregate satisfaction scores
        """
        satisfactions = [constraint(x) for constraint in self.constraints]

        stacked_satisfactions = torch.stack(satisfactions, dim=0)
        return {
            "individual": stacked_satisfactions,
            "mean": stacked_satisfactions.mean(dim=0),
            "min": stacked_satisfactions.min(dim=0)[0],
            "max": stacked_satisfactions.max(dim=0)[0],
        }

    def monitor_gradients(self, x: torch.Tensor) -> StabilityMetrics | None:
        """
        Monitor gradient health during optimization.

        Args:
            x: Input tensor with gradients

        Returns:
            Stability metrics if monitoring enabled, None otherwise
        """
        if not self.enable_monitoring or self.gradient_monitor is None:
            return None

        # Compute gradients for each constraint
        constraint_gradients = []
        for constraint in self.constraints:
            if x.requires_grad:
                # Compute gradient for this constraint
                sat = constraint(x).mean()
                if sat.requires_grad:
                    grad = torch.autograd.grad(
                        sat, x, retain_graph=True, create_graph=False
                    )[0]
                    constraint_gradients.append(grad)
                else:
                    constraint_gradients.append(None)
            else:
                constraint_gradients.append(None)

        # Analyze gradients
        metrics = self.gradient_monitor.analyze_gradients(
            constraint_gradients, constraint_wise=True
        )

        return metrics

    def update_adaptive_weights(
        self,
        metrics: StabilityMetrics | None = None,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        Update adaptive weights based on gradient health.

        Args:
            metrics: Pre-computed stability metrics (optional)
            x: Input tensor for computing metrics if not provided

        Returns:
            Updated weights if adaptation enabled, None otherwise
        """
        if not self.enable_adaptive_weights or self.weight_manager is None:
            return None

        # Get metrics if not provided
        if metrics is None and x is not None:
            metrics = self.monitor_gradients(x)

        if metrics is None:
            # Create default metrics
            metrics = StabilityMetrics()

        # Get current satisfactions if available
        constraint_satisfactions = None
        if x is not None:
            satisfactions = [constraint(x).mean() for constraint in self.constraints]
            constraint_satisfactions = torch.tensor(
                [s.item() if hasattr(s, "item") else s for s in satisfactions]
            )

        # Update weights
        self.adaptive_weights = self.weight_manager.adapt(
            metrics=metrics,
            constraint_satisfactions=constraint_satisfactions,
            strategy="gradient_magnitude",
        )

        return self.adaptive_weights

    def get_optimization_status(self) -> dict[str, Any]:
        """Get comprehensive optimization status including gradient health and weights."""
        status = {
            "num_constraints": len(self.constraints),
            "loss_type": self.loss_type,
            "monitoring_enabled": self.enable_monitoring,
            "adaptive_weights_enabled": self.enable_adaptive_weights,
        }

        if self.gradient_monitor is not None:
            status["gradient_metrics"] = self.gradient_monitor.metrics.to_dict()
            status["gradient_summary"] = self.gradient_monitor.get_gradient_summary()
            status["interventions"] = self.gradient_monitor.suggest_intervention()

        if self.weight_manager is not None:
            status["current_weights"] = (
                self.adaptive_weights.tolist()
                if self.adaptive_weights is not None
                else None
            )
            status["weight_summary"] = self.weight_manager.get_weight_summary()

        return status


class GradientGuide:
    """
    Utility for using constraint gradients to guide generation.

    Provides methods for computing gradients and applying them to
    steer generative models toward constraint satisfaction.
    """

    def __init__(self, constraint_loss: ConstraintLoss):
        self.constraint_loss = constraint_loss

    def compute_gradients(
        self, x: torch.Tensor, create_graph: bool = False
    ) -> torch.Tensor:
        """
        Compute gradients of constraint loss with respect to input.

        Args:
            x: Input tensor (requires_grad=True)
            create_graph: Whether to create computation graph for higher-order gradients

        Returns:
            Gradients with same shape as x
        """
        if not x.requires_grad:
            raise ValueError("Input tensor must require gradients")

        loss = self.constraint_loss(x).mean()  # Scalar loss for backward

        gradients = torch.autograd.grad(
            loss, x, create_graph=create_graph, retain_graph=create_graph
        )[0]

        return gradients

    def guided_step(
        self, x: torch.Tensor, step_size: float = 0.1, normalize: bool = True
    ) -> torch.Tensor:
        """
        Take a gradient step toward better constraint satisfaction.

        Args:
            x: Current input tensor
            step_size: Size of gradient step
            normalize: Whether to normalize gradients

        Returns:
            Updated tensor after gradient step
        """
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)

        gradients = self.compute_gradients(x)

        if normalize:
            grad_norm = torch.norm(gradients, dim=-1, keepdim=True)
            gradients = gradients / torch.clamp(grad_norm, min=1e-8)

        # Step in direction of decreasing loss (increasing satisfaction)
        return x - step_size * gradients

    def iterative_refinement(
        self,
        x: torch.Tensor,
        num_steps: int = 10,
        step_size: float = 0.1,
        tolerance: float = 1e-4,
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Iteratively refine input to improve constraint satisfaction.

        Args:
            x: Initial input tensor
            num_steps: Maximum number of refinement steps
            step_size: Size of each gradient step
            tolerance: Convergence tolerance

        Returns:
            Tuple of (refined_input, loss_history)
        """
        x_current = x.clone().detach().requires_grad_(True)
        loss_history = []

        for step in range(num_steps):
            # Compute current loss
            loss = self.constraint_loss(x_current).mean().item()
            loss_history.append(loss)

            # Check convergence
            if step > 0 and abs(loss_history[-2] - loss_history[-1]) < tolerance:
                break

            # Take gradient step
            x_current = self.guided_step(x_current, step_size)
            x_current = x_current.detach().requires_grad_(True)

        return x_current.detach(), loss_history
