"""Tests for gradient stability monitoring and adaptive weight management."""

import pytest
import torch

from symbiont.bridge.compiler import DifferentiableCompiler
from symbiont.core.constraints import AlwaysTrue
from symbiont.domains.sequence import Contains, GCContent, Length
from symbiont.optimization.adaptive import AdaptiveWeightManager
from symbiont.optimization.monitor import GradientMonitor, StabilityMetrics


class TestStabilityMetrics:
    """Test StabilityMetrics dataclass and methods."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = StabilityMetrics()
        assert metrics.stability_score == 1.0
        assert metrics.vanishing_ratio == 0.0
        assert metrics.exploding_ratio == 0.0
        assert metrics.is_stable
        assert not metrics.is_vanishing
        assert not metrics.is_exploding
        assert not metrics.needs_intervention

    def test_vanishing_detection(self):
        """Test vanishing gradient detection."""
        metrics = StabilityMetrics()
        metrics.vanishing_ratio = 0.6
        metrics.stability_score = 0.3

        assert metrics.is_vanishing
        assert not metrics.is_stable
        assert metrics.needs_intervention

    def test_exploding_detection(self):
        """Test exploding gradient detection."""
        metrics = StabilityMetrics()
        metrics.exploding_ratio = 0.4
        metrics.stability_score = 0.4

        assert metrics.is_exploding
        assert not metrics.is_stable
        assert metrics.needs_intervention

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = StabilityMetrics()
        metrics.gradient_norm = 0.01
        metrics.iteration = 5

        data = metrics.to_dict()
        assert data["gradient_norm"] == 0.01
        assert data["iteration"] == 5
        assert "stability_score" in data


class TestGradientMonitor:
    """Test GradientMonitor functionality."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = GradientMonitor(vanishing_threshold=1e-8, exploding_threshold=1e3)
        assert monitor.vanishing_threshold == 1e-8
        assert monitor.exploding_threshold == 1e3
        assert monitor.metrics.iteration == 0

    def test_gradient_analysis(self):
        """Test gradient analysis with normal gradients."""
        monitor = GradientMonitor()

        # Create mock gradients
        gradients = [
            torch.randn(10, 20) * 0.1,
            torch.randn(10, 20) * 0.15,
            torch.randn(10, 20) * 0.08,
        ]

        metrics = monitor.analyze_gradients(gradients, constraint_wise=True)

        assert metrics.total_constraints == 3
        assert metrics.effective_constraints == 3
        assert metrics.vanishing_ratio == 0.0
        assert metrics.exploding_ratio == 0.0
        assert metrics.stability_score > 0.5

    def test_vanishing_gradient_detection(self):
        """Test detection of vanishing gradients."""
        monitor = GradientMonitor(vanishing_threshold=1e-4)

        # Create gradients with some vanishing
        # Ensure consistent norms by normalizing first then scaling
        grad1 = torch.randn(10, 20)
        grad1 = grad1 / torch.norm(grad1) * 1e-6  # Definitely vanishing

        grad2 = torch.randn(10, 20)
        grad2 = grad2 / torch.norm(grad2) * 1e-5  # Definitely vanishing

        grad3 = torch.randn(10, 20)
        grad3 = grad3 / torch.norm(grad3) * 0.1  # Normal

        gradients = [grad1, grad2, grad3]

        metrics = monitor.analyze_gradients(gradients)

        assert metrics.vanishing_ratio > 0.6
        assert metrics.is_vanishing
        assert metrics.needs_intervention

    def test_exploding_gradient_detection(self):
        """Test detection of exploding gradients."""
        monitor = GradientMonitor(exploding_threshold=10.0)

        # Create gradients with some exploding
        gradients = [
            torch.randn(10, 20) * 100,  # Exploding
            torch.randn(10, 20) * 50,  # Exploding
            torch.randn(10, 20) * 0.1,  # Normal
        ]

        metrics = monitor.analyze_gradients(gradients)

        assert metrics.exploding_ratio > 0.6
        assert metrics.is_exploding
        assert metrics.needs_intervention

    def test_constraint_balance_calculation(self):
        """Test constraint balance metric."""
        monitor = GradientMonitor()

        # Balanced gradients
        balanced = [
            torch.randn(10, 20) * 0.1,
            torch.randn(10, 20) * 0.11,
            torch.randn(10, 20) * 0.09,
        ]

        metrics = monitor.analyze_gradients(balanced)
        assert metrics.constraint_balance > 0.7

        # Imbalanced gradients
        imbalanced = [
            torch.randn(10, 20) * 0.001,
            torch.randn(10, 20) * 1.0,
            torch.randn(10, 20) * 0.01,
        ]

        metrics = monitor.analyze_gradients(imbalanced)
        assert metrics.constraint_balance < 0.5

    def test_intervention_suggestions(self):
        """Test intervention suggestion system."""
        monitor = GradientMonitor()

        # Test vanishing gradient suggestions
        monitor.metrics.vanishing_ratio = 0.7
        monitor.metrics.stability_score = 0.3
        suggestions = monitor.suggest_intervention()

        assert suggestions["needs_intervention"]
        assert "increase_learning_rate" in suggestions["actions"]
        assert suggestions["urgency"] == "high"

        # Test exploding gradient suggestions
        monitor.metrics = StabilityMetrics()
        monitor.metrics.exploding_ratio = 0.5
        monitor.metrics.stability_score = 0.3
        suggestions = monitor.suggest_intervention()

        assert "decrease_learning_rate" in suggestions["actions"]
        assert "add_gradient_clipping" in suggestions["actions"]

    def test_gradient_history_tracking(self):
        """Test gradient history tracking."""
        monitor = GradientMonitor(history_window=5)

        for i in range(10):
            gradients = [torch.randn(10, 20) * (0.1 + i * 0.01)]
            monitor.analyze_gradients(gradients)

        assert len(monitor.metrics.gradient_history) <= 5
        assert monitor.metrics.iteration == 10


class TestAdaptiveWeightManager:
    """Test AdaptiveWeightManager functionality."""

    def test_initialization_with_weights(self):
        """Test initialization with provided weights."""
        initial_weights = [1.0, 2.0, 0.5]
        manager = AdaptiveWeightManager(initial_weights=initial_weights)

        assert manager.num_constraints == 3
        assert torch.allclose(manager.weights, torch.tensor(initial_weights))

    def test_initialization_with_num_constraints(self):
        """Test initialization with constraint count."""
        manager = AdaptiveWeightManager(num_constraints=5)

        assert manager.num_constraints == 5
        assert manager.weights.shape == (5,)
        assert torch.allclose(manager.weights, torch.ones(5))

    def test_gradient_magnitude_adaptation(self):
        """Test adaptation based on gradient magnitudes."""
        manager = AdaptiveWeightManager(num_constraints=3)

        # Create metrics with vanishing gradients
        metrics = StabilityMetrics(vanishing_threshold=1e-4)
        metrics.vanishing_ratio = 0.6

        # Mock constraint gradients
        gradients = [
            torch.randn(10, 20) * 1e-6,  # Vanishing
            torch.randn(10, 20) * 0.1,  # Normal
            torch.randn(10, 20) * 1e-5,  # Vanishing
        ]

        initial_weights = manager.weights.clone()
        new_weights = manager.adapt(
            metrics, constraint_gradients=gradients, strategy="gradient_magnitude"
        )

        # Weights for vanishing gradients should increase
        assert new_weights[0] > initial_weights[0]
        assert new_weights[2] > initial_weights[2]

    def test_satisfaction_balance_adaptation(self):
        """Test adaptation based on satisfaction balance."""
        manager = AdaptiveWeightManager(num_constraints=3)
        metrics = StabilityMetrics()

        # Mock satisfaction scores
        satisfactions = torch.tensor([0.2, 0.9, 0.5])  # Imbalanced

        initial_weights = manager.weights.clone()
        new_weights = manager.adapt(
            metrics,
            constraint_satisfactions=satisfactions,
            strategy="satisfaction_balance",
        )

        # Weight for poorly satisfied constraint should increase
        assert new_weights[0] > initial_weights[0]
        # Weight for well satisfied constraint should decrease
        assert new_weights[1] < initial_weights[1]

    def test_weight_clamping(self):
        """Test weight clamping to valid range."""
        manager = AdaptiveWeightManager(
            num_constraints=2,
            min_weight=0.1,
            max_weight=10.0,
            adaptation_rate=10.0,  # High rate to trigger clamping
        )

        metrics = StabilityMetrics()

        # Force large updates
        for _ in range(10):
            gradients = [
                torch.randn(10, 20) * 100,  # Will try to decrease weight
                torch.randn(10, 20) * 1e-8,  # Will try to increase weight
            ]
            manager.adapt(metrics, constraint_gradients=gradients)

        assert manager.weights[0] >= manager.min_weight
        assert manager.weights[1] <= manager.max_weight

    def test_momentum_application(self):
        """Test momentum in weight updates."""
        manager = AdaptiveWeightManager(
            num_constraints=2, momentum=0.9, adaptation_rate=0.1
        )

        metrics = StabilityMetrics()

        # Apply consistent updates
        for _ in range(5):
            satisfactions = torch.tensor([0.2, 0.8])  # Consistent imbalance
            manager.adapt(
                metrics,
                constraint_satisfactions=satisfactions,
                strategy="satisfaction_balance",
            )

        # Momentum should accumulate changes
        assert not torch.allclose(manager.weight_momentum, torch.zeros(2))

    def test_state_dict_save_load(self):
        """Test saving and loading state."""
        manager = AdaptiveWeightManager(num_constraints=3)

        # Make some adaptations
        metrics = StabilityMetrics()
        satisfactions = torch.tensor([0.3, 0.7, 0.5])
        manager.adapt(metrics, constraint_satisfactions=satisfactions)

        # Save state
        state = manager.state_dict()

        # Create new manager and load state
        new_manager = AdaptiveWeightManager(num_constraints=3)
        new_manager.load_state_dict(state)

        assert torch.allclose(new_manager.weights, manager.weights)
        assert new_manager.iteration == manager.iteration


class TestIntegratedGradientStability:
    """Test integrated gradient stability with compiler."""

    def test_compiler_with_monitoring(self):
        """Test compiler with monitoring enabled."""
        compiler = DifferentiableCompiler(tnorm="product")

        constraints = [Contains("ATG"), GCContent(0.4, 0.6), Length(20, 30)]

        loss_fn = compiler.compile(
            constraints,
            loss_type="violation",
            enable_monitoring=True,
            enable_adaptive_weights=False,
        )

        assert loss_fn.enable_monitoring
        assert loss_fn.gradient_monitor is not None
        assert not loss_fn.enable_adaptive_weights

    def test_compiler_with_adaptive_weights(self):
        """Test compiler with adaptive weights enabled."""
        compiler = DifferentiableCompiler(tnorm="product")

        constraints = [AlwaysTrue(), AlwaysTrue()]

        loss_fn = compiler.compile(
            constraints, enable_monitoring=False, enable_adaptive_weights=True
        )

        assert loss_fn.enable_adaptive_weights
        assert loss_fn.weight_manager is not None
        assert loss_fn.adaptive_weights is not None

    def test_gradient_monitoring_during_optimization(self):
        """Test gradient monitoring during optimization."""
        compiler = DifferentiableCompiler()
        constraints = [AlwaysTrue(), GCContent(0.4, 0.6)]

        loss_fn = compiler.compile(
            constraints, enable_monitoring=True, enable_adaptive_weights=True
        )

        # Create input with gradients
        x = torch.randn(10, 100, requires_grad=True)

        # Forward pass
        _ = loss_fn(x).mean()  # Compute loss to build graph

        # Monitor gradients
        metrics = loss_fn.monitor_gradients(x)

        assert metrics is not None
        assert metrics.total_constraints == 2
        assert metrics.iteration > 0

    def test_adaptive_weight_updates(self):
        """Test adaptive weight updates during optimization."""
        compiler = DifferentiableCompiler()
        constraints = [AlwaysTrue(), AlwaysTrue()]

        loss_fn = compiler.compile(constraints, enable_adaptive_weights=True)

        x = torch.randn(10, 100, requires_grad=True)

        initial_weights = loss_fn.adaptive_weights.clone()

        # Update weights
        new_weights = loss_fn.update_adaptive_weights(x=x)

        assert new_weights is not None
        # Weights might change based on gradient health
        assert new_weights.shape == initial_weights.shape

    def test_optimization_status_reporting(self):
        """Test comprehensive optimization status reporting."""
        compiler = DifferentiableCompiler()
        constraints = [AlwaysTrue(), GCContent(0.4, 0.6)]

        loss_fn = compiler.compile(
            constraints, enable_monitoring=True, enable_adaptive_weights=True
        )

        x = torch.randn(10, 100, requires_grad=True)
        loss_fn(x)
        loss_fn.monitor_gradients(x)

        status = loss_fn.get_optimization_status()

        assert "num_constraints" in status
        assert "gradient_metrics" in status
        assert "current_weights" in status
        assert status["monitoring_enabled"]
        assert status["adaptive_weights_enabled"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
