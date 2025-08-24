"""Visualization utilities for constraint satisfaction and generation results."""

from typing import Any

import numpy as np
import torch

# Optional matplotlib imports - will gracefully handle if not available
try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from symbiont.core.constraints import Constraint


def plot_satisfaction(
    sequences: torch.Tensor,
    constraints: list[Constraint],
    sequence_names: list[str] | None = None,
    constraint_names: list[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    save_path: str | None = None,
) -> Any | None:
    """
    Plot constraint satisfaction heatmap.

    Args:
        sequences: Generated sequences to evaluate
        constraints: List of constraints to check
        sequence_names: Optional names for sequences
        constraint_names: Optional names for constraints
        figsize: Figure size tuple
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object if matplotlib available, None otherwise
    """
    if not HAS_MATPLOTLIB:
        print(
            "Matplotlib not available for plotting. Install with: pip install matplotlib"
        )
        return None

    if not constraints:
        print("No constraints to visualize")
        return None

    # Compute satisfaction scores
    satisfaction_scores = []
    for constraint in constraints:
        scores = constraint.satisfaction(sequences)
        satisfaction_scores.append(scores.cpu().numpy())

    satisfaction_matrix = np.array(satisfaction_scores)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(satisfaction_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Set labels
    num_sequences = satisfaction_matrix.shape[1]
    num_constraints = satisfaction_matrix.shape[0]

    if sequence_names is None:
        sequence_names = [f"Seq_{i}" for i in range(num_sequences)]
    if constraint_names is None:
        constraint_names = [f"C_{i}" for i in range(num_constraints)]

    ax.set_xticks(range(num_sequences))
    ax.set_yticks(range(num_constraints))
    ax.set_xticklabels(sequence_names, rotation=45, ha="right")
    ax.set_yticklabels(constraint_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Satisfaction Score", rotation=270, labelpad=15)

    # Add text annotations
    for i in range(num_constraints):
        for j in range(num_sequences):
            ax.text(
                j,
                i,
                f"{satisfaction_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_title("Constraint Satisfaction Heatmap")
    ax.set_xlabel("Generated Sequences")
    ax.set_ylabel("Constraints")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_satisfaction_distribution(
    satisfaction_scores: torch.Tensor,
    title: str = "Satisfaction Score Distribution",
    bins: int = 20,
    figsize: tuple[int, int] = (10, 6),
    save_path: str | None = None,
) -> Any | None:
    """
    Plot distribution of satisfaction scores.

    Args:
        satisfaction_scores: Tensor of satisfaction scores [0,1]
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size tuple
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object if available, None otherwise
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    scores_np = satisfaction_scores.cpu().numpy()

    # Create histogram
    n, bins_edges, patches = ax.hist(scores_np, bins=bins, alpha=0.7, edgecolor="black")

    # Color bars based on satisfaction level
    for _i, (patch, edge) in enumerate(zip(patches, bins_edges[:-1], strict=False)):  # type: ignore[arg-type,unused-ignore]
        if edge < 0.3:
            patch.set_facecolor("red")
        elif edge < 0.7:
            patch.set_facecolor("orange")
        else:
            patch.set_facecolor("green")

    # Add statistics
    mean_score = scores_np.mean()
    std_score = scores_np.std()

    ax.axvline(
        mean_score,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_score:.3f}",
    )
    ax.axvline(
        mean_score + std_score,
        color="blue",
        linestyle=":",
        alpha=0.7,
        label=f"±1σ: {std_score:.3f}",
    )
    ax.axvline(mean_score - std_score, color="blue", linestyle=":", alpha=0.7)

    ax.set_xlabel("Satisfaction Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_constraint_performance(
    analysis_results: dict[str, Any],
    figsize: tuple[int, int] = (12, 8),
    save_path: str | None = None,
) -> Any | None:
    """
    Plot individual constraint performance comparison.

    Args:
        analysis_results: Results from constraint_analysis()
        figsize: Figure size tuple
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object if available, None otherwise
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return None

    if "constraint_performance" not in analysis_results:
        print("No constraint performance data found")
        return None

    perf_data = analysis_results["constraint_performance"]
    constraint_names = list(perf_data.keys())

    if not constraint_names:
        print("No constraints found in analysis results")
        return None

    # Extract metrics
    metrics = ["mean_satisfaction", "satisfaction_rate", "high_satisfaction_rate"]
    metric_labels = [
        "Mean Satisfaction",
        "Satisfaction Rate (>0.5)",
        "High Satisfaction Rate (>0.8)",
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels, strict=False)):
        values = [perf_data[name][metric] for name in constraint_names]

        bars = axes[i].bar(range(len(constraint_names)), values, alpha=0.7)

        # Color bars based on performance
        for bar, value in zip(bars, values, strict=False):
            if value < 0.3:
                bar.set_color("red")
            elif value < 0.7:
                bar.set_color("orange")
            else:
                bar.set_color("green")

        axes[i].set_xlabel("Constraints")
        axes[i].set_ylabel(label)
        axes[i].set_title(f"{label} by Constraint")
        axes[i].set_xticks(range(len(constraint_names)))
        axes[i].set_xticklabels(constraint_names, rotation=45, ha="right")
        axes[i].set_ylim(0, 1)
        axes[i].grid(True, alpha=0.3)

        # Add value labels on bars
        for j, (_, value) in enumerate(zip(bars, values, strict=False)):
            axes[i].text(
                j, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_generation_quality_trends(
    quality_metrics_history: list[dict[str, float]],
    figsize: tuple[int, int] = (12, 8),
    save_path: str | None = None,
) -> Any | None:
    """
    Plot generation quality metrics over time/iterations.

    Args:
        quality_metrics_history: List of quality metrics from multiple generations
        figsize: Figure size tuple
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure object if available, None otherwise
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return None

    if not quality_metrics_history:
        print("No quality metrics history provided")
        return None

    # Extract time series data
    iterations = list(range(len(quality_metrics_history)))

    # Key metrics to plot
    key_metrics = [
        "mean_satisfaction",
        "violation_rate",
        "pairwise_distance_mean",
        "high_satisfaction_rate",
        "unique_sequences",
    ]

    # Filter metrics that exist in the data
    available_metrics = []
    for metric in key_metrics:
        if all(metric in data for data in quality_metrics_history):
            available_metrics.append(metric)

    if not available_metrics:
        print("No common metrics found across all iterations")
        return None

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, metric in enumerate(available_metrics[:6]):  # Plot up to 6 metrics
        values = [data[metric] for data in quality_metrics_history]

        axes[i].plot(iterations, values, marker="o", linewidth=2, markersize=4)
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
        axes[i].grid(True, alpha=0.3)

        # Add trend line
        if len(iterations) > 2:
            z = np.polyfit(iterations, values, 1)
            p = np.poly1d(z)
            axes[i].plot(iterations, p(iterations), "--", alpha=0.7, color="red")

    # Hide empty subplots
    for i in range(len(available_metrics), 6):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def print_satisfaction_summary(
    sequences: torch.Tensor,
    constraints: list[Constraint],
    constraint_names: list[str] | None = None,
) -> None:
    """
    Print a text summary of constraint satisfaction.

    Args:
        sequences: Generated sequences to evaluate
        constraints: List of constraints to check
        constraint_names: Optional names for constraints
    """
    if not constraints:
        print("No constraints to evaluate")
        return

    print("=" * 60)
    print("CONSTRAINT SATISFACTION SUMMARY")
    print("=" * 60)

    if constraint_names is None:
        constraint_names = [f"Constraint_{i}" for i in range(len(constraints))]

    total_scores = []

    for _i, (constraint, name) in enumerate(
        zip(constraints, constraint_names, strict=False)
    ):
        scores = constraint.satisfaction(sequences)
        total_scores.append(scores)

        mean_score = scores.mean().item()
        std_score = scores.std().item()
        min_score = scores.min().item()
        max_score = scores.max().item()
        satisfaction_rate = (scores > 0.5).float().mean().item()

        print(f"\n{name}:")
        print(f"  Mean satisfaction: {mean_score:.3f} (±{std_score:.3f})")
        print(f"  Range: [{min_score:.3f}, {max_score:.3f}]")
        print(f"  Satisfaction rate (>0.5): {satisfaction_rate:.1%}")

        if mean_score < 0.3:
            status = "❌ POOR"
        elif mean_score < 0.7:
            status = "⚠️  FAIR"
        else:
            status = "✅ GOOD"
        print(f"  Status: {status}")

    # Overall summary
    if total_scores:
        overall_scores = torch.stack(total_scores, dim=0).mean(dim=0)
        overall_mean = overall_scores.mean().item()
        sequences_passing_all = (overall_scores > 0.8).float().mean().item()
        sequences_failing_any = (overall_scores < 0.5).float().mean().item()

        print("\n" + "=" * 60)
        print("OVERALL PERFORMANCE")
        print("=" * 60)
        print(f"Overall satisfaction: {overall_mean:.3f}")
        print(f"Sequences passing all constraints (>0.8): {sequences_passing_all:.1%}")
        print(f"Sequences failing any constraint (<0.5): {sequences_failing_any:.1%}")

        if overall_mean > 0.8:
            print("🎉 Excellent constraint satisfaction!")
        elif overall_mean > 0.6:
            print("👍 Good constraint satisfaction")
        elif overall_mean > 0.4:
            print("⚠️  Moderate constraint satisfaction - room for improvement")
        else:
            print("❌ Poor constraint satisfaction - significant issues")


def visualize_sequence_alignment(
    sequences: torch.Tensor,
    vocab_map: dict[int, str] | None = None,
    max_sequences: int = 10,
    figsize: tuple[int, int] = (15, 8),
) -> Any | None:
    """
    Visualize sequence alignment for comparison.

    Args:
        sequences: Generated sequences to visualize
        vocab_map: Mapping from indices to characters/symbols
        max_sequences: Maximum number of sequences to display
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object if available, None otherwise
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for plotting")
        return None

    if sequences.dim() == 1:
        sequences = sequences.unsqueeze(0)

    # Limit number of sequences
    num_seqs = min(sequences.shape[0], max_sequences)
    sequences = sequences[:num_seqs]

    if vocab_map is None:
        vocab_map = {0: "A", 1: "T", 2: "G", 3: "C"}  # Default DNA mapping

    # Create visualization matrix
    sequences.shape[1]
    vis_matrix = sequences.cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create custom colormap for DNA bases
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    cmap = ListedColormap(colors[: len(vocab_map)])

    ax.imshow(vis_matrix, cmap=cmap, aspect="auto")

    # Set labels
    ax.set_xlabel("Position")
    ax.set_ylabel("Sequence")
    ax.set_title("Sequence Alignment Visualization")

    # Create custom legend
    legend_elements = []
    for idx, symbol in vocab_map.items():
        if idx < len(colors):
            legend_elements.append(
                patches.Patch(color=colors[idx], label=f"{symbol} ({idx})")
            )

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add sequence labels
    ax.set_yticks(range(num_seqs))
    ax.set_yticklabels([f"Seq_{i}" for i in range(num_seqs)])

    plt.tight_layout()

    return fig
