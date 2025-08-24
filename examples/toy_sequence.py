#!/usr/bin/env python3
"""
Toy DNA Sequence Generation Example

This example demonstrates the core Symbiont framework using a simple
mock generator to create DNA sequences with various constraints.

This is a proof-of-concept that shows:
1. Constraint definition using the DSL
2. Constraint compilation to differentiable form
3. Constraint-guided generation
4. Evaluation and visualization of results
"""

import numpy as np
import torch

from symbiont import GenerationConfig, Rules
from symbiont.domains.sequence import (
    Contains,
    GCContent,
    Length,
    NoRepeats,
    StartCodon,
    StopCodon,
)
from symbiont.generators.mock import MockSequenceGenerator
from symbiont.utils.metrics import constraint_analysis, generation_quality_metrics
from symbiont.utils.visualization import print_satisfaction_summary


def create_dna_constraints() -> Rules:
    """Create a set of realistic DNA sequence constraints."""
    rules = Rules()

    # Hard constraints - must be satisfied
    rules.enforce(StartCodon())  # Must have ATG start codon
    rules.enforce(StopCodon())  # Must have stop codon
    rules.enforce(Length(21, 30))  # Length between 21-30 bp

    # Soft constraints - should be satisfied
    rules.constrain(GCContent(0.4, 0.6))  # GC content 40-60%
    rules.constrain(NoRepeats(max_repeat_length=2))  # Avoid long repeats

    # Preferences - gentle guidance
    rules.prefer(Contains("GAATTC"))  # EcoRI recognition site

    return rules


def demonstrate_basic_generation():
    """Demonstrate basic generation without constraints."""
    print("=" * 70)
    print("BASIC GENERATION (NO CONSTRAINTS)")
    print("=" * 70)

    generator = MockSequenceGenerator(vocab_size=4, sequence_length=25)
    config = GenerationConfig(batch_size=5, max_length=25, seed=42)

    sequences = generator.generate(config=config)
    decoded_sequences = generator._decode_output(sequences)

    print("Generated sequences:")
    for i, seq in enumerate(decoded_sequences):
        print(f"  {i+1}: {seq}")

    return sequences


def demonstrate_constrained_generation():
    """Demonstrate constraint-guided generation."""
    print("\n" + "=" * 70)
    print("CONSTRAINT-GUIDED GENERATION")
    print("=" * 70)

    # Create constraints
    rules = create_dna_constraints()
    print("Constraints defined:")
    for i, constraint in enumerate(rules.constraints):
        print(f"  {i+1}: {constraint}")

    # Initialize generator and compiler
    generator = MockSequenceGenerator(vocab_size=4, sequence_length=25)
    config = GenerationConfig(
        batch_size=10, max_length=25, seed=42, constraint_weight=2.0
    )

    # Generate with constraints
    print(f"\nGenerating {config.batch_size} sequences with constraints...")
    constrained_sequences = generator.constrained_generate(
        constraints=rules.constraints,
        config=config,
        max_attempts=50,
        satisfaction_threshold=0.7,
    )

    decoded_constrained = generator._decode_output(constrained_sequences)

    print("\nConstraint-guided sequences:")
    for i, seq in enumerate(decoded_constrained):
        print(f"  {i+1}: {seq}")

    return constrained_sequences, rules


def demonstrate_constraint_evaluation(sequences: torch.Tensor, rules: Rules):
    """Demonstrate constraint evaluation and analysis."""
    print("\n" + "=" * 70)
    print("CONSTRAINT EVALUATION & ANALYSIS")
    print("=" * 70)

    # Basic satisfaction evaluation
    satisfaction_results = rules.evaluate(sequences)

    print("Constraint satisfaction summary:")
    for constraint_type, scores in satisfaction_results.items():
        mean_score = scores.mean().item()
        print(f"  {constraint_type.capitalize()}: {mean_score:.3f}")

    # Detailed constraint analysis
    analysis = constraint_analysis(sequences, rules.constraints, detailed=False)

    print("\nDetailed analysis:")
    print(f"  Number of constraints: {analysis['num_constraints']}")
    print(f"  Number of sequences: {analysis['num_sequences']}")
    print(
        f"  Overall mean satisfaction: {analysis['overall']['mean_overall_satisfaction']:.3f}"
    )
    print(
        f"  Sequences satisfying all (>0.8): {analysis['overall']['sequences_satisfying_all']:.1%}"
    )
    print(
        f"  Sequences failing any (<0.5): {analysis['overall']['sequences_failing_any']:.1%}"
    )

    # Print detailed satisfaction summary
    constraint_names = [str(c) for c in rules.constraints]
    print_satisfaction_summary(sequences, rules.constraints, constraint_names)


def demonstrate_quality_comparison():
    """Compare quality between constrained and unconstrained generation."""
    print("\n" + "=" * 70)
    print("QUALITY COMPARISON")
    print("=" * 70)

    generator = MockSequenceGenerator(vocab_size=4, sequence_length=25)
    rules = create_dna_constraints()
    config = GenerationConfig(batch_size=20, max_length=25, seed=42)

    # Generate unconstrained sequences
    unconstrained = generator.generate(config=config)
    unconstrained_metrics = generation_quality_metrics(unconstrained, rules.constraints)

    # Generate constrained sequences
    constrained = generator.constrained_generate(
        constraints=rules.constraints, config=config, satisfaction_threshold=0.6
    )
    constrained_metrics = generation_quality_metrics(constrained, rules.constraints)

    print("Quality Comparison:")
    print(
        f"{'Metric':<30} {'Unconstrained':<15} {'Constrained':<15} {'Improvement':<15}"
    )
    print("-" * 75)

    key_metrics = [
        "mean_satisfaction",
        "violation_rate",
        "high_satisfaction_rate",
        "unique_sequences",
        "pairwise_distance_mean",
    ]

    for metric in key_metrics:
        if metric in unconstrained_metrics and metric in constrained_metrics:
            uncon_val = unconstrained_metrics[metric]
            con_val = constrained_metrics[metric]

            # Calculate improvement (higher is better for most metrics, except violation_rate)
            if metric == "violation_rate":
                improvement = (uncon_val - con_val) / uncon_val if uncon_val > 0 else 0
            else:
                improvement = (con_val - uncon_val) / uncon_val if uncon_val > 0 else 0

            print(
                f"{metric:<30} {uncon_val:<15.3f} {con_val:<15.3f} {improvement:+.1%}"
            )


def demonstrate_dsl_features():
    """Demonstrate advanced DSL features and constraint composition."""
    print("\n" + "=" * 70)
    print("ADVANCED DSL FEATURES")
    print("=" * 70)

    # Create more complex constraint combinations
    rules = Rules()

    # Conditional-like constraints using logical operators
    start_or_alt = StartCodon() | Contains("GTG")  # ATG or alternative start GTG
    rules.enforce(start_or_alt, weight=8.0)

    # Multiple stop codons
    rules.require_any(
        StopCodon("amber"),  # TAG
        StopCodon("ochre"),  # TAA
        StopCodon("opal"),  # TGA
        weight=5.0,
    )

    # Combined constraints
    rules.require_all(Length(24, 36), GCContent(0.3, 0.7), NoRepeats(3), weight=3.0)

    # Forbidden patterns
    rules.forbid_all(
        Contains("AAAA"),  # No poly-A
        Contains("TTTT"),  # No poly-T
        Contains("GGGG"),  # No poly-G
        Contains("CCCC"),  # No poly-C
        weight=2.0,
    )

    print("Complex constraint system created:")
    print(f"  Total constraints: {len(rules.constraints)}")
    print(f"  Hard constraints: {len(rules.hard_constraints)}")
    print(f"  Soft constraints: {len(rules.soft_constraints)}")
    print(f"  Forbidden constraints: {len(rules.forbidden_constraints)}")
    print(f"  Preferences: {len(rules.preferences)}")

    # Test with mock generator
    generator = MockSequenceGenerator(vocab_size=4, sequence_length=30)
    config = GenerationConfig(batch_size=5, max_length=30, seed=123)

    sequences = generator.constrained_generate(
        constraints=rules.constraints, config=config, satisfaction_threshold=0.5
    )

    decoded = generator._decode_output(sequences)
    print("\nGenerated sequences with complex constraints:")
    for i, seq in enumerate(decoded):
        print(f"  {i+1}: {seq}")

    # Evaluate complex constraints
    results = rules.evaluate(sequences)
    print("\nComplex constraint satisfaction:")
    for constraint_type, scores in results.items():
        print(f"  {constraint_type}: {scores.mean().item():.3f}")


def demonstrate_gradient_guided_refinement():
    """Demonstrate gradient-based constraint satisfaction improvement."""
    print("\n" + "=" * 70)
    print("GRADIENT-GUIDED REFINEMENT")
    print("=" * 70)

    rules = create_dna_constraints()
    generator = MockSequenceGenerator(vocab_size=4, sequence_length=25)

    # Generate initial sequences
    config = GenerationConfig(batch_size=3, max_length=25, seed=999)
    initial_sequences = generator.generate(config=config).float().requires_grad_(True)

    print("Initial sequences (before refinement):")
    initial_decoded = generator._decode_output(initial_sequences)
    for i, seq in enumerate(initial_decoded):
        print(f"  {i+1}: {seq}")

    # Evaluate initial satisfaction
    initial_results = rules.evaluate(initial_sequences)
    initial_satisfaction = initial_results["total"].mean().item()
    print(f"Initial mean satisfaction: {initial_satisfaction:.3f}")

    # Apply gradient-based refinement
    print("\nApplying gradient-based refinement...")
    refined_sequences = generator.constrained_generate(
        constraints=rules.constraints,
        config=config,
        initial_sequences=initial_sequences,  # Pass initial sequences for refinement
        use_gradients=True,
        refinement_steps=10,
        step_size=0.05,
    )

    print("Refined sequences (after gradient guidance):")
    refined_decoded = generator._decode_output(refined_sequences)
    for i, seq in enumerate(refined_decoded):
        print(f"  {i+1}: {seq}")

    # Evaluate refined satisfaction
    refined_results = rules.evaluate(refined_sequences)
    refined_satisfaction = refined_results["total"].mean().item()
    print(f"Refined mean satisfaction: {refined_satisfaction:.3f}")

    improvement = refined_satisfaction - initial_satisfaction
    print(f"Satisfaction improvement: {improvement:+.3f}")


def main():
    """Run the complete toy sequence generation demonstration."""
    print("🧬 SYMBIONT TOY DNA SEQUENCE GENERATION DEMO")
    print("Demonstrating constraint-guided generation with mock DNA sequences")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Basic generation
        _ = demonstrate_basic_generation()  # Show basic generation capability

        # 2. Constrained generation
        constrained_sequences, rules = demonstrate_constrained_generation()

        # 3. Constraint evaluation
        demonstrate_constraint_evaluation(constrained_sequences, rules)

        # 4. Quality comparison
        demonstrate_quality_comparison()

        # 5. Advanced DSL features
        demonstrate_dsl_features()

        # 6. Gradient-guided refinement (disabled for mock generator)
        # Note: This would work with real continuous generative models
        print("\n" + "=" * 70)
        print("GRADIENT-GUIDED REFINEMENT")
        print("=" * 70)
        print("⚠️  Gradient-guided refinement is disabled for the mock generator")
        print("   since it uses discrete tokens. This feature would work with")
        print(
            "   real continuous generative models like transformers or diffusion models."
        )
        # demonstrate_gradient_guided_refinement()

        print("\n" + "=" * 70)
        print("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 70)
        print("\nKey takeaways:")
        print("• Constraint-guided generation improves satisfaction vs. unconstrained")
        print("• DSL provides intuitive way to express biological constraints")
        print("• Fuzzy logic enables differentiable constraint evaluation")
        print("• Framework is extensible to different domains and generators")
        print("\nNext steps:")
        print("• Integrate with real generative models (transformers, diffusion)")
        print("• Add more sophisticated constraint types")
        print("• Implement web interface for interactive constraint definition")
        print("• Optimize performance for large-scale generation")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
