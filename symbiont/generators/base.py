"""Base generator interfaces and abstract classes."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch

from symbiont.core.constraints import Constraint
from symbiont.core.types import GenerationConfig


class Generator(Protocol):
    """
    Protocol defining the interface for all generative models.

    This protocol ensures compatibility across different generative architectures
    while maintaining flexibility for domain-specific implementations.
    """

    def generate(
        self,
        prompt: str | torch.Tensor | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate samples without constraint guidance.

        Args:
            prompt: Optional prompt or conditioning input
            config: Generation configuration parameters
            **kwargs: Additional generation parameters

        Returns:
            Generated samples as tensor
        """
        ...

    def constrained_generate(
        self,
        constraints: list[Constraint],
        config: GenerationConfig | None = None,
        prompt: str | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate samples with constraint guidance.

        Args:
            constraints: List of constraints to satisfy
            config: Generation configuration parameters
            prompt: Optional prompt or conditioning input
            **kwargs: Additional generation parameters

        Returns:
            Generated samples that aim to satisfy constraints
        """
        ...


class BaseGenerator(ABC):
    """
    Abstract base class for generator implementations.

    Provides common functionality and enforces the generator interface
    through abstract methods.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self._is_initialized = False

    @abstractmethod
    def _initialize(self, **kwargs: Any) -> None:
        """Initialize the generator model and parameters."""
        pass

    @abstractmethod
    def _forward_pass(
        self, input_tensor: torch.Tensor, config: GenerationConfig
    ) -> torch.Tensor:
        """Execute forward pass of the generative model."""
        pass

    @abstractmethod
    def _encode_input(self, input_data: str | torch.Tensor | None) -> torch.Tensor:
        """Encode input data into tensor format."""
        pass

    @abstractmethod
    def _decode_output(self, output_tensor: torch.Tensor) -> Any:
        """Decode tensor output into interpretable format."""
        pass

    def generate(
        self,
        prompt: str | torch.Tensor | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Standard generation without constraints."""
        if not self._is_initialized:
            self._initialize(**kwargs)
            self._is_initialized = True

        config = config or GenerationConfig()

        # Set random seed if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Encode input
        if prompt is not None:
            input_tensor = self._encode_input(prompt)
        else:
            # Generate from random noise
            input_tensor = torch.randn(
                config.batch_size, config.max_length, device=self.device
            )

        # Generate samples
        with torch.no_grad():
            output = self._forward_pass(input_tensor, config)

        return output

    def constrained_generate(
        self,
        constraints: list[Constraint],
        config: GenerationConfig | None = None,
        prompt: str | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate with constraint guidance using rejection sampling as fallback.

        This base implementation uses simple rejection sampling.
        Subclasses should override for more sophisticated constraint integration.
        """
        if not constraints:
            return self.generate(prompt, config, **kwargs)

        config = config or GenerationConfig()
        max_attempts = kwargs.get("max_attempts", 100)

        best_samples = None
        best_satisfaction = -1.0

        for _ in range(max_attempts):
            # Generate candidate samples
            samples = self.generate(prompt, config, **kwargs)

            # Evaluate constraint satisfaction
            total_satisfaction = 0.0
            for constraint in constraints:
                satisfaction = constraint.satisfaction(samples).mean().item()
                total_satisfaction += satisfaction

            avg_satisfaction = total_satisfaction / len(constraints)

            # Keep best samples
            if avg_satisfaction > best_satisfaction:
                best_satisfaction = avg_satisfaction
                best_samples = samples

            # Early stopping if satisfaction threshold met
            satisfaction_threshold = kwargs.get("satisfaction_threshold", 0.8)
            if avg_satisfaction >= satisfaction_threshold:
                break

        if best_samples is None:
            # Fallback to unconstrained generation
            return self.generate(prompt, config, **kwargs)

        return best_samples

    def to(self, device: torch.device) -> "BaseGenerator":
        """Move generator to specified device."""
        self.device = device
        return self


class GeneratorMixin:
    """Mixin providing common generator utilities."""

    @staticmethod
    def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        return logits / temperature

    @staticmethod
    def top_k_sampling(
        logits: torch.Tensor, k: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """Sample from top-k logits with temperature."""
        if k <= 0:
            raise ValueError("k must be positive")

        # Apply temperature
        scaled_logits = GeneratorMixin.apply_temperature(logits, temperature)

        # Get top-k values and indices
        top_k_logits, top_k_indices = torch.topk(scaled_logits, k, dim=-1)

        # Create probability distribution
        top_k_probs = torch.softmax(top_k_logits, dim=-1)

        # Sample from top-k distribution
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1)

        # Map back to original vocabulary
        return torch.gather(top_k_indices, -1, sampled_indices).squeeze(-1)

    @staticmethod
    def nucleus_sampling(
        logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0
    ) -> torch.Tensor:
        """Sample using nucleus (top-p) sampling."""
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")

        # Apply temperature
        scaled_logits = GeneratorMixin.apply_temperature(logits, temperature)

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(
            scaled_logits, descending=True, dim=-1
        )

        # Compute cumulative probabilities
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff for nucleus
        mask = cumulative_probs <= p
        # Ensure at least one token is kept
        mask[..., 0] = True

        # Zero out probabilities outside nucleus
        nucleus_logits = sorted_logits.clone()
        nucleus_logits[~mask] = float("-inf")

        # Sample from nucleus distribution
        nucleus_probs = torch.softmax(nucleus_logits, dim=-1)
        sampled_sorted_indices = torch.multinomial(nucleus_probs, num_samples=1)

        # Map back to original vocabulary
        return torch.gather(sorted_indices, -1, sampled_sorted_indices).squeeze(-1)
