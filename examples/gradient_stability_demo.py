#!/usr/bin/env python3
"""Demonstration of gradient stability monitoring and adaptive weight management."""

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from symbiont.bridge.compiler import DifferentiableCompiler
from symbiont.core.constraints import AlwaysTrue, Constraint

console = Console()


class RangeConstraint(Constraint):
    """Simple constraint that checks if values are in a range."""

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Check if mean of x is in range."""
        mean_vals = x.mean(dim=-1)  # Average over features
        # Smooth satisfaction using sigmoid
        lower = torch.sigmoid(10 * (mean_vals - self.min_val))
        upper = torch.sigmoid(10 * (self.max_val - mean_vals))
        return lower * upper


class NormConstraint(Constraint):
    """Constraint on the norm of the input."""

    def __init__(self, target_norm: float):
        self.target_norm = target_norm

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Check how close norm is to target."""
        norms = torch.norm(x, dim=-1)  # Norm for each sample in batch
        # Gaussian-like satisfaction
        diff = (norms - self.target_norm) ** 2
        return torch.exp(-diff)


def demonstrate_gradient_monitoring():
    """Show gradient stability monitoring in action."""
    console.print("\n[bold cyan]Gradient Stability Monitoring Demo[/bold cyan]\n")

    # Create constraints with potential gradient issues
    constraints = [
        RangeConstraint(-0.5, 0.5),  # Range constraint
        RangeConstraint(-1.0, 1.0),  # Another range
        NormConstraint(5.0),  # Norm constraint
        AlwaysTrue(),  # Control constraint
    ]

    # Compile with monitoring enabled
    compiler = DifferentiableCompiler(tnorm="product")
    loss_fn = compiler.compile(
        constraints,
        loss_type="violation",
        enable_monitoring=True,
        enable_adaptive_weights=True,
    )

    console.print("[yellow]Initial Setup:[/yellow]")
    console.print(f"  • Constraints: {len(constraints)}")
    console.print("  • T-norm: Product (can cause vanishing gradients)")
    console.print("  • Monitoring: Enabled")
    console.print("  • Adaptive Weights: Enabled\n")

    # Create sample input - general tensor data
    batch_size = 10
    feature_dim = 50
    x = torch.randn(batch_size, feature_dim, requires_grad=True)

    # Simulate optimization steps
    console.print("[yellow]Simulating optimization steps:[/yellow]\n")

    for step in range(5):
        # Forward pass
        _ = loss_fn(x).mean()  # Compute loss to build graph

        # Monitor gradients
        metrics = loss_fn.monitor_gradients(x)

        if metrics:
            # Create status table
            table = Table(title=f"Step {step + 1} - Gradient Health")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            table.add_column("Status", style="green")

            # Add metrics
            table.add_row(
                "Gradient Norm",
                f"{metrics.gradient_norm:.2e}",
                "✓" if 1e-4 < metrics.gradient_norm < 1e2 else "⚠️",
            )
            table.add_row(
                "Vanishing Ratio",
                f"{metrics.vanishing_ratio:.1%}",
                "✓" if metrics.vanishing_ratio < 0.3 else "⚠️",
            )
            table.add_row(
                "Exploding Ratio",
                f"{metrics.exploding_ratio:.1%}",
                "✓" if metrics.exploding_ratio < 0.1 else "⚠️",
            )
            table.add_row(
                "Stability Score",
                f"{metrics.stability_score:.2f}",
                "✓" if metrics.stability_score > 0.5 else "⚠️",
            )
            table.add_row(
                "Effective Constraints",
                f"{metrics.effective_constraints}/{metrics.total_constraints}",
                (
                    "✓"
                    if metrics.effective_constraints == metrics.total_constraints
                    else "⚠️"
                ),
            )

            console.print(table)

            # Check if intervention needed
            if metrics.needs_intervention:
                suggestions = loss_fn.gradient_monitor.suggest_intervention()
                console.print("\n[red]Intervention Needed![/red]")
                console.print(f"Suggested actions: {', '.join(suggestions['actions'])}")

                # Update adaptive weights
                new_weights = loss_fn.update_adaptive_weights(metrics=metrics, x=x)
                if new_weights is not None:
                    console.print("\n[green]Adaptive weights updated:[/green]")
                    for i, w in enumerate(new_weights):
                        console.print(f"  Constraint {i}: {w:.3f}")
            else:
                console.print("\n[green]✓ Gradients healthy[/green]")

        # Simulate gradient step (would normally use optimizer)
        with torch.no_grad():
            if x.grad is not None:
                x -= 0.01 * x.grad
                x.grad.zero_()

        console.print("-" * 50 + "\n")

    # Final status
    status = loss_fn.get_optimization_status()

    console.print(
        Panel.fit(
            f"[bold]Final Optimization Status[/bold]\n\n"
            f"Monitoring Enabled: {status['monitoring_enabled']}\n"
            f"Adaptive Weights Enabled: {status['adaptive_weights_enabled']}\n"
            f"Gradient Metrics Available: {'gradient_metrics' in status}\n"
            f"Current Weights: {status.get('current_weights', 'N/A')}",
            title="Summary",
            border_style="green",
        )
    )


def demonstrate_weight_adaptation():
    """Show adaptive weight management in action."""
    console.print("\n[bold cyan]Adaptive Weight Management Demo[/bold cyan]\n")

    from symbiont.optimization.adaptive import AdaptiveWeightManager
    from symbiont.optimization.monitor import StabilityMetrics

    # Initialize manager
    manager = AdaptiveWeightManager(
        num_constraints=3, adaptation_rate=0.2, temperature=1.0
    )

    console.print("[yellow]Simulating constraint imbalance scenario:[/yellow]\n")

    # Simulate imbalanced constraint satisfactions
    for iteration in range(3):
        metrics = StabilityMetrics()

        # Mock imbalanced satisfactions
        satisfactions = torch.tensor(
            [
                0.1,  # Poorly satisfied
                0.9,  # Well satisfied
                0.5,  # Moderately satisfied
            ]
        )

        # Mock gradient norms (imbalanced)
        gradients = [
            torch.randn(10, 20) * 1e-6,  # Very small gradient
            torch.randn(10, 20) * 0.1,  # Normal gradient
            torch.randn(10, 20) * 0.05,  # Moderate gradient
        ]

        console.print(f"[cyan]Iteration {iteration + 1}:[/cyan]")
        console.print(f"  Satisfactions: {satisfactions.tolist()}")
        console.print(f"  Initial weights: {manager.weights.tolist()}")

        # Adapt weights
        new_weights = manager.adapt(
            metrics,
            constraint_gradients=gradients,
            constraint_satisfactions=satisfactions,
            strategy="satisfaction_balance",
        )

        console.print(f"  Updated weights: {new_weights.tolist()}")
        console.print(
            "  [green]→ Weight for poorly satisfied constraint increased[/green]\n"
        )

    # Show weight summary
    summary = manager.get_weight_summary()
    console.print(
        Panel(summary, title="Weight Adaptation Summary", border_style="cyan")
    )


if __name__ == "__main__":
    try:
        demonstrate_gradient_monitoring()
        demonstrate_weight_adaptation()

        console.print(
            "\n[bold green]✅ Gradient stability features demonstration complete![/bold green]\n"
        )

    except Exception as e:
        console.print(f"\n[red]Error during demonstration: {e}[/red]\n")
        raise
