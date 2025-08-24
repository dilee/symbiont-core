"""Utility functions for evaluating constraint satisfaction and generation quality."""

from typing import Any

import numpy as np
import torch

from symbiont.core.constraints import Constraint


def satisfaction_score(
    sequences: torch.Tensor, constraints: list[Constraint], aggregation: str = "mean"
) -> torch.Tensor:
    """
    Compute overall satisfaction score for sequences against constraints.

    Args:
        sequences: Generated sequences to evaluate
        constraints: List of constraints to check
        aggregation: How to aggregate scores ("mean", "min", "max", "weighted")

    Returns:
        Satisfaction scores for each sequence
    """
    if not constraints:
        return torch.ones(sequences.shape[0] if sequences.dim() > 1 else 1)

    # Evaluate each constraint
    scores = []
    for constraint in constraints:
        score = constraint.satisfaction(sequences)
        scores.append(score)

    # Stack scores [num_constraints, batch_size]
    all_scores = torch.stack(scores, dim=0)

    # Aggregate scores
    if aggregation == "mean":
        return all_scores.mean(dim=0)
    elif aggregation == "min":
        return all_scores.min(dim=0)[0]
    elif aggregation == "max":
        return all_scores.max(dim=0)[0]
    elif aggregation == "weighted":
        # Use geometric mean for stricter aggregation
        return torch.prod(all_scores, dim=0) ** (1.0 / len(constraints))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def constraint_violation(
    sequences: torch.Tensor, constraints: list[Constraint], threshold: float = 0.5
) -> dict[str, torch.Tensor]:
    """
    Compute constraint violations for analysis.

    Args:
        sequences: Generated sequences to evaluate
        constraints: List of constraints to check
        threshold: Satisfaction threshold below which constraint is violated

    Returns:
        Dictionary with violation statistics
    """
    if not constraints:
        batch_size = sequences.shape[0] if sequences.dim() > 1 else 1
        return {
            "total_violations": torch.zeros(batch_size),
            "violation_rate": torch.zeros(batch_size),
            "violated_constraints": torch.tensor([], dtype=torch.int),
            "individual_violations": torch.zeros((0, batch_size)),
        }

    # Evaluate constraints
    violations = []
    violated_indices = []

    for i, constraint in enumerate(constraints):
        score = constraint.satisfaction(sequences)
        violation = (score < threshold).float()
        violations.append(violation)

        if violation.any():
            violated_indices.append(i)

    violations_tensor = torch.stack(violations, dim=0)

    return {
        "total_violations": violations_tensor.sum(dim=0),
        "violation_rate": violations_tensor.mean(dim=0),
        "violated_constraints": torch.tensor(violated_indices, dtype=torch.int),
        "individual_violations": violations_tensor,
    }


def generation_quality_metrics(
    sequences: torch.Tensor,
    constraints: list[Constraint],
    reference_sequences: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Compute comprehensive quality metrics for generated sequences.

    Args:
        sequences: Generated sequences
        constraints: Constraints to evaluate against
        reference_sequences: Optional reference sequences for comparison

    Returns:
        Dictionary of quality metrics
    """
    metrics = {}

    # Constraint satisfaction metrics
    if constraints:
        satisfaction_scores = satisfaction_score(sequences, constraints, "mean")

        metrics.update(
            {
                "mean_satisfaction": satisfaction_scores.mean().item(),
                "min_satisfaction": satisfaction_scores.min().item(),
                "max_satisfaction": satisfaction_scores.max().item(),
                "satisfaction_std": satisfaction_scores.std().item(),
                "high_satisfaction_rate": (satisfaction_scores > 0.8)
                .float()
                .mean()
                .item(),
            }
        )

        # Violation analysis
        violations = constraint_violation(sequences, constraints)
        metrics.update(
            {
                "mean_violations": violations["total_violations"].float().mean().item(),
                "violation_rate": violations["violation_rate"].mean().item(),
                "num_violated_constraints": len(violations["violated_constraints"]),
            }
        )

    # Diversity metrics
    metrics.update(_compute_diversity_metrics(sequences))

    # Comparison with reference if provided
    if reference_sequences is not None:
        metrics.update(_compute_reference_metrics(sequences, reference_sequences))

    return metrics


def _compute_diversity_metrics(sequences: torch.Tensor) -> dict[str, float]:
    """Compute diversity metrics for generated sequences."""
    metrics = {}

    if sequences.dim() == 1:
        sequences = sequences.unsqueeze(0)

    batch_size = sequences.shape[0]

    if batch_size < 2:
        return {
            "pairwise_distance_mean": 0.0,
            "pairwise_distance_std": 0.0,
            "unique_sequences": 1.0,
        }

    # Compute pairwise Hamming distances
    distances = []
    unique_sequences = set()

    for i in range(batch_size):
        seq_i = sequences[i]
        seq_str = "_".join(map(str, seq_i.cpu().numpy()))
        unique_sequences.add(seq_str)

        for j in range(i + 1, batch_size):
            seq_j = sequences[j]
            # Hamming distance
            distance = (seq_i != seq_j).float().mean().item()
            distances.append(distance)

    if distances:
        metrics.update(
            {
                "pairwise_distance_mean": float(np.mean(distances)),
                "pairwise_distance_std": float(np.std(distances)),
            }
        )
    else:
        metrics.update(
            {
                "pairwise_distance_mean": 0.0,
                "pairwise_distance_std": 0.0,
            }
        )

    metrics["unique_sequences"] = float(len(unique_sequences) / batch_size)

    return metrics


def _compute_reference_metrics(
    sequences: torch.Tensor, reference_sequences: torch.Tensor
) -> dict[str, float]:
    """Compute metrics comparing generated sequences to references."""
    metrics = {}

    if sequences.dim() == 1:
        sequences = sequences.unsqueeze(0)
    if reference_sequences.dim() == 1:
        reference_sequences = reference_sequences.unsqueeze(0)

    # Find closest reference for each generated sequence
    min_distances = []

    for gen_seq in sequences:
        distances_to_refs = []
        for ref_seq in reference_sequences:
            # Handle different lengths
            min_len = min(len(gen_seq), len(ref_seq))
            gen_truncated = gen_seq[:min_len]
            ref_truncated = ref_seq[:min_len]

            # Hamming distance
            if min_len > 0:
                distance = (gen_truncated != ref_truncated).float().mean().item()
            else:
                distance = 1.0

            distances_to_refs.append(distance)

        min_distances.append(min(distances_to_refs))

    metrics.update(
        {
            "mean_distance_to_reference": np.mean(min_distances),
            "min_distance_to_reference": np.min(min_distances),
            "novelty_rate": np.mean([d > 0.1 for d in min_distances]),  # >10% different
        }
    )

    return metrics


def constraint_analysis(
    sequences: torch.Tensor, constraints: list[Constraint], detailed: bool = False
) -> dict[str, Any]:
    """
    Detailed analysis of constraint satisfaction patterns.

    Args:
        sequences: Generated sequences
        constraints: Constraints to analyze
        detailed: Whether to include per-constraint detailed analysis

    Returns:
        Analysis results dictionary
    """
    if not constraints:
        return {"summary": "No constraints to analyze"}

    analysis: dict[str, Any] = {
        "num_constraints": len(constraints),
        "num_sequences": sequences.shape[0] if sequences.dim() > 1 else 1,
        "constraint_performance": {},
    }

    # Analyze each constraint individually
    for i, constraint in enumerate(constraints):
        constraint_name = f"{constraint.__class__.__name__}_{i}"
        scores = constraint.satisfaction(sequences)

        perf: dict[str, Any] = {
            "mean_satisfaction": scores.mean().item(),
            "std_satisfaction": scores.std().item(),
            "min_satisfaction": scores.min().item(),
            "max_satisfaction": scores.max().item(),
            "satisfaction_rate": (scores > 0.5).float().mean().item(),
            "high_satisfaction_rate": (scores > 0.8).float().mean().item(),
        }

        if detailed:
            perf.update(
                {
                    "score_distribution": {
                        "0.0-0.2": (scores <= 0.2).float().mean().item(),
                        "0.2-0.4": ((scores > 0.2) & (scores <= 0.4))
                        .float()
                        .mean()
                        .item(),
                        "0.4-0.6": ((scores > 0.4) & (scores <= 0.6))
                        .float()
                        .mean()
                        .item(),
                        "0.6-0.8": ((scores > 0.6) & (scores <= 0.8))
                        .float()
                        .mean()
                        .item(),
                        "0.8-1.0": (scores > 0.8).float().mean().item(),
                    },
                    "individual_scores": (
                        scores.cpu().numpy().tolist() if detailed else None
                    ),
                }
            )

        analysis["constraint_performance"][constraint_name] = perf

    # Overall analysis
    all_scores = [
        constraints[i].satisfaction(sequences) for i in range(len(constraints))
    ]
    overall_scores = torch.stack(all_scores, dim=0).mean(dim=0)

    analysis["overall"] = {
        "mean_overall_satisfaction": overall_scores.mean().item(),
        "sequences_satisfying_all": (overall_scores > 0.8).float().mean().item(),
        "sequences_failing_any": (overall_scores < 0.5).float().mean().item(),
    }

    return analysis
