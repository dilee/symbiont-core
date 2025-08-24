"""Domain-specific constraints for biological sequences (DNA, RNA, protein)."""

import torch

from symbiont.core.types import validate_tensor_shape
from symbiont.domains.base import (
    CompositionConstraint,
    PatternConstraint,
    RangeConstraint,
    StructuralConstraint,
)


class Contains(PatternConstraint):
    """Constraint requiring sequences to contain a specific pattern."""

    def __init__(self, pattern: str, case_sensitive: bool = False):
        """
        Args:
            pattern: Pattern to search for (can include IUPAC codes)
            case_sensitive: Whether matching should be case sensitive
        """
        super().__init__(pattern, case_sensitive)
        self.pattern_str = pattern

    def _sequence_to_string(self, x: torch.Tensor) -> list[str]:
        """Convert DNA tensor to string representation."""
        # Assume 0=A, 1=T, 2=G, 3=C encoding
        dna_map = {0: "A", 1: "T", 2: "G", 3: "C"}
        sequences = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            seq_str = "".join([dna_map.get(idx.item(), "N") for idx in seq])
            sequences.append(seq_str)

        return sequences

    def __repr__(self) -> str:
        return f"Contains('{self.pattern_str}')"


class Length(RangeConstraint):
    """Constraint on sequence length."""

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        exactly: int | None = None,
    ):
        """
        Args:
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            exactly: Exact length required (overrides min/max)
        """
        super().__init__(min_length, max_length, exactly)

    def _compute_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sequence lengths."""
        if x.dim() == 1:
            return torch.tensor([x.shape[0]], device=x.device, dtype=torch.float32)
        else:
            return torch.tensor(
                [seq.shape[0] for seq in x], device=x.device, dtype=torch.float32
            )

    def __repr__(self) -> str:
        if self.min_value == self.max_value and self.min_value is not None:
            return f"Length(exactly={int(self.min_value)})"
        return f"Length(min={self.min_value}, max={self.max_value})"


class GCContent(CompositionConstraint):
    """Constraint on GC content for DNA sequences."""

    def __init__(self, min_gc: float = 0.0, max_gc: float = 1.0):
        """
        Args:
            min_gc: Minimum GC ratio (0-1)
            max_gc: Maximum GC ratio (0-1)
        """
        super().__init__(min_gc, max_gc)

    def _compute_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """Compute GC content ratio for each sequence using vectorized operations."""
        # Validate input tensor first
        x = validate_tensor_shape(x)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Vectorized GC counting: much faster than per-sequence iteration
        # Count G (2) and C (3) nucleotides across all sequences at once
        gc_mask = (x == 2) | (x == 3)  # Shape: [batch_size, seq_len]
        gc_counts = gc_mask.sum(dim=1).float()  # Shape: [batch_size]
        total_counts = x.shape[1]  # All sequences have same length

        # Handle edge case of zero-length sequences
        if total_counts == 0:
            return torch.zeros(x.shape[0], device=x.device)

        return gc_counts / total_counts

    def __repr__(self) -> str:
        return f"GCContent(min={self.min_ratio:.2f}, max={self.max_ratio:.2f})"


class NoRepeats(PatternConstraint):
    """Constraint forbidding repetitive sequences."""

    def __init__(self, max_repeat_length: int = 3):
        """
        Args:
            max_repeat_length: Maximum allowed repeat length
        """
        self.max_repeat_length = max_repeat_length
        # Create pattern for detecting repeats of any nucleotide
        pattern = "|".join(
            [f"{base}{{{max_repeat_length + 1},}}" for base in ["A", "T", "G", "C"]]
        )
        super().__init__(pattern, case_sensitive=False)

    def _sequence_to_string(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to DNA string."""
        dna_map = {0: "A", 1: "T", 2: "G", 3: "C"}
        sequences = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            seq_str = "".join([dna_map.get(idx.item(), "N") for idx in seq])
            sequences.append(seq_str)

        return sequences

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1 - (presence of repeats)."""
        base_satisfaction = super().satisfaction(x)
        # Invert because we want to forbid repeats
        return torch.tensor(1.0) - base_satisfaction

    def __repr__(self) -> str:
        return f"NoRepeats(max_length={self.max_repeat_length})"


class HasMotif(PatternConstraint):
    """Constraint requiring specific sequence motifs (using IUPAC codes)."""

    def __init__(self, motif: str, case_sensitive: bool = False):
        """
        Args:
            motif: Motif pattern using IUPAC nucleotide codes
        """
        # Convert IUPAC codes to regex
        iupac_map = {
            "R": "[AG]",  # Purine
            "Y": "[CT]",  # Pyrimidine
            "S": "[GC]",  # Strong
            "W": "[AT]",  # Weak
            "K": "[GT]",  # Keto
            "M": "[AC]",  # Amino
            "B": "[CGT]",  # Not A
            "D": "[AGT]",  # Not C
            "H": "[ACT]",  # Not G
            "V": "[ACG]",  # Not T
            "N": "[ACGT]",  # Any
        }

        # Convert motif to regex pattern
        regex_pattern = motif
        for iupac, regex in iupac_map.items():
            regex_pattern = regex_pattern.replace(iupac, regex)

        super().__init__(regex_pattern, case_sensitive)
        self.motif = motif

    def _sequence_to_string(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to DNA string."""
        dna_map = {0: "A", 1: "T", 2: "G", 3: "C"}
        sequences = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            seq_str = "".join([dna_map.get(idx.item(), "N") for idx in seq])
            sequences.append(seq_str)

        return sequences

    def __repr__(self) -> str:
        return f"HasMotif('{self.motif}')"


class StartCodon(Contains):
    """Constraint requiring presence of start codon (ATG)."""

    def __init__(self) -> None:
        super().__init__("ATG", case_sensitive=False)

    def __repr__(self) -> str:
        return "StartCodon()"


class StopCodon(PatternConstraint):
    """Constraint requiring presence of stop codons."""

    def __init__(self, codon_type: str = "any"):
        """
        Args:
            codon_type: "any", "amber" (TAG), "ochre" (TAA), "opal" (TGA)
        """
        patterns = {"any": "TAG|TAA|TGA", "amber": "TAG", "ochre": "TAA", "opal": "TGA"}

        if codon_type not in patterns:
            raise ValueError(f"Unknown codon type: {codon_type}")

        super().__init__(patterns[codon_type], case_sensitive=False)
        self.codon_type = codon_type

    def _sequence_to_string(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to DNA string."""
        dna_map = {0: "A", 1: "T", 2: "G", 3: "C"}
        sequences = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            seq_str = "".join([dna_map.get(idx.item(), "N") for idx in seq])
            sequences.append(seq_str)

        return sequences

    def __repr__(self) -> str:
        return f"StopCodon('{self.codon_type}')"


class ReadingFrame(StructuralConstraint):
    """Constraint ensuring valid reading frame structure."""

    def __init__(self, frame: int = 0):
        """
        Args:
            frame: Reading frame (0, 1, or 2)
        """
        if frame not in [0, 1, 2]:
            raise ValueError("Frame must be 0, 1, or 2")

        super().__init__("reading_frame")
        self.frame = frame

    def _analyze_structure(self, x: torch.Tensor) -> dict:
        """Analyze reading frame structure."""
        seq = x.squeeze()
        seq_length = seq.shape[0]

        # Check if sequence length is compatible with reading frame
        effective_length = seq_length - self.frame
        has_complete_codons = effective_length % 3 == 0

        return {
            "sequence_length": seq_length,
            "effective_length": effective_length,
            "has_complete_codons": has_complete_codons,
            "num_complete_codons": effective_length // 3,
        }

    def _evaluate_structure(self, structure_info: dict) -> float:
        """Evaluate reading frame satisfaction."""
        if structure_info["has_complete_codons"]:
            return 1.0
        else:
            # Partial satisfaction based on how close to complete codons
            remainder = structure_info["effective_length"] % 3
            return float(1.0 - (remainder / 3.0))

    def __repr__(self) -> str:
        return f"ReadingFrame(frame={self.frame})"


class CodonUsage(CompositionConstraint):
    """Constraint on codon usage bias."""

    def __init__(self, preferred_codons: dict[str, float]):
        """
        Args:
            preferred_codons: Mapping of codon to preference weight (0-1)
        """
        super().__init__(0.0, 1.0)  # Will override _compute_ratio
        self.preferred_codons = preferred_codons

    def _compute_ratio(self, x: torch.Tensor) -> torch.Tensor:
        """Compute codon usage preference ratio."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        dna_map = {0: "A", 1: "T", 2: "G", 3: "C"}
        ratios = []

        for seq in x:
            seq_str = "".join([dna_map.get(idx.item(), "N") for idx in seq])

            # Extract codons
            codons = [seq_str[i : i + 3] for i in range(0, len(seq_str) - 2, 3)]

            if not codons:
                ratios.append(0.0)
                continue

            # Calculate preference score
            total_preference = 0.0
            for codon in codons:
                total_preference += self.preferred_codons.get(
                    codon, 0.5
                )  # Default neutral

            avg_preference = total_preference / len(codons)
            ratios.append(avg_preference)

        return torch.tensor(ratios, device=x.device)

    def __repr__(self) -> str:
        return f"CodonUsage({len(self.preferred_codons)} codons)"
