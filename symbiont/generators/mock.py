"""Mock generator for testing and development."""

from typing import Any

import torch
import torch.nn as nn

from symbiont.bridge.compiler import DifferentiableCompiler
from symbiont.core.constraints import Constraint
from symbiont.core.types import GenerationConfig
from symbiont.generators.base import BaseGenerator


class MockSequenceGenerator(BaseGenerator):
    """
    Mock sequence generator for testing constraint systems.

    Generates simple sequences (e.g., DNA, text) that can be used
    to validate constraint satisfaction without heavy ML models.
    """

    def __init__(
        self,
        vocab_size: int = 4,
        sequence_length: int = 20,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        # For DNA sequences: 0=A, 1=T, 2=G, 3=C
        self.vocab_map = {
            4: ["A", "T", "G", "C"],  # DNA
            26: [chr(i + ord("A")) for i in range(26)],  # Uppercase letters
            52: [chr(i + ord("A")) for i in range(26)]
            + [chr(i + ord("a")) for i in range(26)],  # Mixed case
        }

        # Simple embedding layer for sequence representation
        self.embedding = nn.Embedding(vocab_size, 16)

    def _initialize(self, **kwargs: Any) -> None:
        """Initialize mock generator parameters."""
        self.embedding.to(self.device)

    def _forward_pass(
        self, input_tensor: torch.Tensor, config: GenerationConfig
    ) -> torch.Tensor:
        """
        Mock forward pass that generates random sequences.

        In a real generator, this would be the actual model forward pass.
        """
        batch_size = config.batch_size
        seq_length = min(config.max_length, self.sequence_length)

        # Generate random sequences
        sequences = torch.randint(
            0, self.vocab_size, (batch_size, seq_length), device=self.device
        )

        # Apply temperature if specified
        if hasattr(config, "temperature") and config.temperature != 1.0:
            # Simulate temperature by adjusting randomness
            noise_scale = 1.0 / config.temperature
            noise = torch.randn_like(sequences.float()) * noise_scale
            sequences = torch.clamp(
                sequences.float() + noise, 0, self.vocab_size - 1
            ).long()

        return sequences

    def _encode_input(self, input_data: str | torch.Tensor | None) -> torch.Tensor:
        """Encode input into tensor format."""
        if input_data is None:
            return torch.zeros(1, device=self.device)

        if isinstance(input_data, str) and self.vocab_size == 4:
            # Simple character-to-index mapping for DNA
            char_to_idx = {"A": 0, "T": 1, "G": 2, "C": 3}
            indices = [char_to_idx.get(c.upper(), 0) for c in input_data]
            return torch.tensor(indices, device=self.device)

        if isinstance(input_data, torch.Tensor):
            return input_data.to(self.device)

        raise ValueError(f"Cannot encode input of type {type(input_data)}")

    def _decode_output(self, output_tensor: torch.Tensor) -> list[str]:
        """Decode tensor output to string sequences."""
        if self.vocab_size not in self.vocab_map:
            # Fallback to numeric representation
            return [" ".join(map(str, seq.cpu().numpy())) for seq in output_tensor]

        vocab = self.vocab_map[self.vocab_size]
        sequences = []

        for seq in output_tensor:
            char_seq = "".join([vocab[int(idx.item())] for idx in seq])
            sequences.append(char_seq)

        return sequences

    def constrained_generate(
        self,
        constraints: list[Constraint],
        config: GenerationConfig | None = None,
        prompt: str | torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate sequences with constraint guidance using gradient-based steering.

        This mock implementation demonstrates how constraints could be integrated
        during generation rather than just for filtering.
        """
        if not constraints:
            return self.generate(prompt, config, **kwargs)

        config = config or GenerationConfig()

        # Compile constraints to differentiable form
        compiler = DifferentiableCompiler()
        constraint_loss = compiler.compile(constraints, loss_type="violation")

        # Initialize with random sequences
        sequences = self.generate(prompt, config, **kwargs)

        # Optional: Apply gradient-based refinement
        if kwargs.get("use_gradients", False):
            # Remove any conflicting kwargs
            refinement_kwargs = {
                k: v for k, v in kwargs.items() if k not in ["initial_sequences"]
            }
            sequences = self._gradient_based_refinement(
                sequences, constraint_loss, config, **refinement_kwargs
            )

        return sequences

    def _gradient_based_refinement(
        self,
        initial_sequences: torch.Tensor,
        constraint_loss: nn.Module,
        config: GenerationConfig,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Apply gradient-based refinement to improve constraint satisfaction.

        This is a simplified demonstration of how gradients could guide generation.
        """
        # Convert discrete sequences to continuous for gradient computation
        sequences_float = initial_sequences.float().requires_grad_(True)

        num_steps = kwargs.get("refinement_steps", 5)
        step_size = kwargs.get("step_size", 0.1)

        for _ in range(num_steps):
            # Compute constraint violation loss
            loss = constraint_loss(sequences_float).mean()

            # Compute gradients
            loss.backward()

            with torch.no_grad():
                # Update sequences in direction of decreasing loss
                sequences_float -= step_size * sequences_float.grad
                if sequences_float.grad is not None:
                    sequences_float.grad.zero_()

                # Clamp to valid range
                sequences_float.clamp_(0, self.vocab_size - 1)

        # Convert back to discrete sequences
        return torch.round(sequences_float).long().detach()

    def batch_generate(
        self,
        batch_size: int,
        constraints: list[Constraint] | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate multiple sequences in parallel."""
        config = config or GenerationConfig()
        config.batch_size = batch_size

        if constraints:
            return self.constrained_generate(constraints, config, **kwargs)
        else:
            return self.generate(config=config, **kwargs)

    def evaluate_batch(
        self, sequences: torch.Tensor, constraints: list[Constraint]
    ) -> dict[str, torch.Tensor]:
        """Evaluate constraint satisfaction for a batch of sequences."""
        if not constraints:
            return {"satisfaction": torch.ones(sequences.shape[0])}

        individual_scores = []

        # Evaluate each constraint
        for constraint in constraints:
            scores = constraint.satisfaction(sequences)
            individual_scores.append(scores)

        if individual_scores:
            all_scores = torch.stack(individual_scores, dim=0)
            return {
                "individual_scores": all_scores,
                "mean_satisfaction": all_scores.mean(dim=0),
                "min_satisfaction": all_scores.min(dim=0)[0],
            }
        else:
            return {
                "individual_scores": torch.empty((0, sequences.shape[0])),
                "mean_satisfaction": torch.zeros(sequences.shape[0]),
                "min_satisfaction": torch.zeros(sequences.shape[0]),
            }


class MockTextGenerator(MockSequenceGenerator):
    """Mock text generator using character-level generation."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        # Use extended ASCII for text generation
        super().__init__(vocab_size=128, sequence_length=50, device=device)

    def _encode_input(self, input_data: str | torch.Tensor | None) -> torch.Tensor:
        """Encode text input to character indices."""
        if input_data is None:
            return torch.zeros(1, device=self.device)

        if isinstance(input_data, str):
            # Convert characters to ASCII codes (clamped to vocab range)
            indices = [min(ord(c), self.vocab_size - 1) for c in input_data]
            return torch.tensor(indices, device=self.device)

        if isinstance(input_data, torch.Tensor):
            return input_data.to(self.device)

        raise ValueError(f"Cannot encode input of type {type(input_data)}")

    def _decode_output(self, output_tensor: torch.Tensor) -> list[str]:
        """Decode tensor output to text strings."""
        texts = []
        for seq in output_tensor:
            try:
                # Convert indices back to characters
                chars = [chr(idx.item()) for idx in seq if 0 <= idx.item() < 128]
                texts.append("".join(chars))
            except ValueError:
                # Fallback for invalid character codes
                texts.append(" ".join(map(str, seq.cpu().numpy())))

        return texts
