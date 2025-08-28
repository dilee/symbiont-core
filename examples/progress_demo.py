#!/usr/bin/env python3
"""Demo of ProgressReporter for tracking constraint satisfaction during generation."""

import time

from symbiont import GenerationConfig, Rules
from symbiont.domains.sequence import Contains, GCContent, Length, NoRepeats, StartCodon
from symbiont.generators.mock import MockSequenceGenerator
from symbiont.utils.visualization import ProgressReporter


def simulate_iterative_generation():
    """
    Simulate iterative constraint-guided generation with progress reporting.

    This demonstrates how ProgressReporter tracks satisfaction scores
    as the generator iteratively improves outputs to meet constraints.
    """
    print("=" * 80)
    print("SYMBIONT PROGRESS REPORTER DEMO")
    print("Tracking constraint satisfaction during iterative generation")
    print("=" * 80)
    print()

    # Define biological constraints for DNA sequences
    rules = Rules()
    rules.enforce(StartCodon())  # Must start with ATG
    rules.enforce(Length(21, 30))  # Length between 21-30
    rules.constrain(GCContent(0.4, 0.6))  # GC content 40-60%
    rules.forbid(NoRepeats(max_repeat_length=3))  # No repeats longer than 3
    rules.prefer(Contains("GAATTC"))  # Prefer EcoRI restriction site

    # Name our constraints for clearer output
    constraint_names = [
        "Start Codon (ATG)",
        "Length (21-30)",
        "GC Content (40-60%)",
        "No Long Repeats",
        "EcoRI Site",
    ]

    # Initialize generator
    generator = MockSequenceGenerator(vocab_size=4, sequence_length=25)
    config = GenerationConfig(batch_size=10, max_length=25, temperature=1.0, seed=42)

    # Create progress reporter
    progress_reporter = ProgressReporter(
        constraints=rules.constraints,
        constraint_names=constraint_names,
        target_threshold=0.7,
        update_interval=0.3,  # Update every 0.3 seconds
        show_eta=True,
        use_color=True,
    )

    # Generate initial sequences
    sequences = generator.generate(config=config)

    # Initialize progress tracking
    max_iterations = 15
    progress_reporter.initialize(sequences, max_iterations=max_iterations)

    print("\nStarting iterative refinement...")
    print("Watch as constraint satisfaction improves over iterations!\n")
    time.sleep(1)

    # Simulate iterative refinement
    for iteration in range(1, max_iterations + 1):
        # In a real scenario, this would be gradient-guided refinement
        # For demo, we'll just generate new sequences with slight improvements

        # In a real scenario, we would calculate current satisfaction scores
        # for feedback and use them to guide refinement through gradients

        # Simulate refinement (in reality, this would use gradients)
        # Mock generator doesn't actually refine, but we add small improvements
        # to simulate the effect
        if iteration % 3 == 0:
            # Every 3rd iteration, generate slightly better sequences
            sequences = generator.generate(
                config=GenerationConfig(
                    batch_size=10,
                    max_length=25,
                    temperature=max(0.5, 1.0 - iteration * 0.05),  # Lower temperature
                    seed=42 + iteration,
                )
            )

        # Update progress display
        progress_reporter.update(sequences, iteration=iteration)

        # Add slight delay to see progress animation
        time.sleep(0.4)

        # Check if all constraints are satisfied
        all_satisfied = all(
            constraint.satisfaction(sequences).mean().item() >= 0.7
            for constraint in rules.constraints
        )

        if all_satisfied and iteration > 5:
            print("\n🎉 All constraints satisfied early! Stopping refinement.")
            break

    # Finalize and show summary
    print()
    summary = progress_reporter.finalize()

    # Additional analysis
    print("\n" + "=" * 80)
    print("GENERATION ANALYSIS")
    print("=" * 80)

    if summary["success_rate"] == 1.0:
        print("✅ Perfect! All constraints met target threshold.")
    elif summary["success_rate"] >= 0.8:
        print("👍 Good! Most constraints satisfied.")
    elif summary["success_rate"] >= 0.6:
        print("⚠️  Moderate success. Some constraints need more work.")
    else:
        print("❌ Poor constraint satisfaction. Consider adjusting parameters.")

    # Show which constraints were hardest to satisfy
    print("\nConstraint Difficulty Ranking (by final score):")
    sorted_constraints = sorted(
        summary["constraints"].items(), key=lambda x: x[1]["final_score"]
    )

    for i, (name, data) in enumerate(sorted_constraints, 1):
        score = data["final_score"]
        improvement = data["improvement"]
        difficulty = "HARD" if score < 0.5 else "MEDIUM" if score < 0.8 else "EASY"
        print(f"{i}. {name:25} Score: {score:.3f} ({improvement:+.3f}) - {difficulty}")

    print("\n" + "=" * 80)
    print("Demo complete! ProgressReporter helps track and visualize")
    print("constraint satisfaction during iterative generation.")
    print("=" * 80)


if __name__ == "__main__":
    simulate_iterative_generation()
