#!/usr/bin/env python3
"""
Constraint Template Library Demonstration

This example showcases the new template system that provides pre-built
constraint patterns for common molecular biology applications.

Demonstrates:
1. Using built-in templates for common scenarios
2. Customizing template parameters
3. Combining multiple templates
4. Template discovery and validation
"""

import numpy as np
import torch

from symbiont import GenerationConfig
from symbiont.generators.mock import MockSequenceGenerator
from symbiont.templates import (
    CodonOptimizedTemplate,
    CRISPRGuideTemplate,
    PrimerDesignTemplate,
    PromoterTemplate,
    registry,
)
from symbiont.utils.metrics import constraint_analysis
from symbiont.utils.visualization import print_satisfaction_summary


def demonstrate_template_discovery():
    """Show template discovery and catalog features."""
    print("=" * 80)
    print("TEMPLATE DISCOVERY AND CATALOG")
    print("=" * 80)

    print(f"Available templates: {len(registry)} registered")
    print("\nAll templates:")
    for name in registry.list_templates():
        info = registry.get_info(name)
        print(f"  • {name}: {info.description}")

    print(f"\nDomains: {', '.join(registry.list_domains())}")
    print(f"All tags: {', '.join(registry.list_tags())}")

    # Search examples
    print("\nSearch examples:")
    pcr_templates = registry.search(query="primer")
    print(f"  Templates containing 'primer': {pcr_templates}")

    editing_templates = registry.search(domain="genome_editing")
    print(f"  Genome editing templates: {editing_templates}")

    dna_templates = registry.search(tags=["dna"])
    print(f"  DNA-related templates: {dna_templates}")


def demonstrate_primer_design():
    """Demonstrate PCR primer design template."""
    print("\n" + "=" * 80)
    print("PCR PRIMER DESIGN TEMPLATE")
    print("=" * 80)

    # Basic primer design
    primer_template = PrimerDesignTemplate(
        target_length=(18, 25),
        gc_content=(0.4, 0.6),
        avoid_hairpins=True,
        gc_clamp=True,
    )

    print("Template configuration:")
    print(primer_template.describe())

    # Validate parameters
    errors = primer_template.validate()
    if errors:
        print(f"\nValidation errors: {errors}")
    else:
        print("\n✅ Template parameters are valid")

    # Build constraint rules
    rules = primer_template.build()
    print(f"\nGenerated {len(rules.constraints)} constraints:")
    for i, constraint in enumerate(rules.constraints):
        print(f"  {i+1}: {constraint}")

    # Generate primer sequences
    generator = MockSequenceGenerator(vocab_size=4, sequence_length=22)
    config = GenerationConfig(batch_size=8, max_length=22, seed=123)

    print(f"\nGenerating {config.batch_size} primer candidates...")
    sequences = generator.constrained_generate(
        constraints=rules.constraints,
        config=config,
        satisfaction_threshold=0.7,
        max_attempts=50,
    )

    # Show results
    decoded = generator._decode_output(sequences)
    print("\nGenerated primer candidates:")
    for i, seq in enumerate(decoded):
        print(f"  {i+1}: {seq}")

    # Analyze constraint satisfaction
    analysis = constraint_analysis(sequences, rules.constraints)
    print("\nConstraint satisfaction analysis:")
    print(
        f"  Mean satisfaction: {analysis['overall']['mean_overall_satisfaction']:.3f}"
    )
    print(
        f"  Sequences passing all: {analysis['overall']['sequences_satisfying_all']:.1%}"
    )


def demonstrate_crispr_design():
    """Demonstrate CRISPR guide RNA design template."""
    print("\n" + "=" * 80)
    print("CRISPR GUIDE RNA DESIGN TEMPLATE")
    print("=" * 80)

    # CRISPR guide for SpCas9
    crispr_template = CRISPRGuideTemplate(
        pam_type="NGG", length=20, gc_content=(0.3, 0.7), avoid_poly_t=True
    )

    print("Designing guides for SpCas9 (PAM: NGG)")
    print("Guide length: 20 bp")
    print("GC content range: 30-70%")

    rules = crispr_template.build()
    print(f"\nConstraints: {len(rules.constraints)} defined")

    # Generate guide sequences
    generator = MockSequenceGenerator(vocab_size=4, sequence_length=20)
    config = GenerationConfig(batch_size=6, max_length=20, seed=456)

    sequences = generator.constrained_generate(
        constraints=rules.constraints, config=config, satisfaction_threshold=0.6
    )

    decoded = generator._decode_output(sequences)
    print("\nGenerated CRISPR guide candidates:")
    for i, seq in enumerate(decoded):
        print(f"  Guide {i+1}: {seq}")

    # Show constraint satisfaction details
    constraint_names = [
        "Length (20 bp)",
        "GC Content (30-70%)",
        "No Poly-T",
        "No Long Repeats",
        "PAM Compatibility",
        "Avoid GC Extremes",
    ]
    print_satisfaction_summary(sequences, rules.constraints, constraint_names)


def demonstrate_codon_optimization():
    """Demonstrate codon optimization template."""
    print("\n" + "=" * 80)
    print("CODON OPTIMIZATION TEMPLATE")
    print("=" * 80)

    # Codon optimization for E. coli expression
    codon_template = CodonOptimizedTemplate(
        organism="e_coli",
        length=(90, 120),  # Short gene segment
        avoid_repeats=True,
        avoid_common_sites=True,
    )

    print("Optimizing for E. coli expression")
    print("Target length: 90-120 bp")
    print("Avoiding common restriction sites")

    rules = codon_template.build()

    # Generate optimized sequence
    generator = MockSequenceGenerator(vocab_size=4, sequence_length=102)
    config = GenerationConfig(batch_size=5, max_length=102, seed=789)

    sequences = generator.constrained_generate(
        constraints=rules.constraints, config=config, satisfaction_threshold=0.6
    )

    decoded = generator._decode_output(sequences)
    print("\nCodon-optimized sequences:")
    for i, seq in enumerate(decoded):
        print(f"  {i+1}: {seq[:30]}...{seq[-30:]} ({len(seq)} bp)")

    # Detailed analysis
    analysis = constraint_analysis(sequences, rules.constraints, detailed=True)
    print("\nOptimization quality:")
    print(
        f"  Overall satisfaction: {analysis['overall']['mean_overall_satisfaction']:.3f}"
    )

    # Show available constraint performance
    if "constraints" in analysis:
        constraint_results = analysis["constraints"]
        if len(constraint_results) >= 3:
            print(
                f"  Start/stop codons: {constraint_results[0]['mean_satisfaction']:.3f}"
            )
            print(f"  GC content: {constraint_results[2]['mean_satisfaction']:.3f}")
    else:
        # Fallback: use the rules to evaluate directly
        satisfaction_results = rules.evaluate(sequences)
        print(
            f"  Hard constraints: {satisfaction_results.get('hard', satisfaction_results['total']).mean():.3f}"
        )
        print(
            f"  Soft constraints: {satisfaction_results.get('soft', satisfaction_results['total']).mean():.3f}"
        )


def demonstrate_template_customization():
    """Show how to customize templates for specific needs."""
    print("\n" + "=" * 80)
    print("TEMPLATE CUSTOMIZATION")
    print("=" * 80)

    # Start with base primer template
    base_template = PrimerDesignTemplate(target_length=(20, 25))

    # Customize for high-GC template
    high_gc_template = base_template.customize(
        gc_content=(0.6, 0.8),
        avoid_restriction_sites=["GAATTC", "AAGCTT"],
        gc_clamp=True,
    )

    # Customize for low-GC template
    low_gc_template = base_template.customize(
        gc_content=(0.2, 0.4), max_repeat_length=2, avoid_hairpins=False
    )

    print("Template variations:")
    print("1. Base template: GC content not specified")
    print("2. High-GC variant: GC 60-80%, avoids EcoRI/HindIII")
    print("3. Low-GC variant: GC 20-40%, stricter repeat limits")

    # Compare constraint counts
    base_rules = base_template.build()
    high_gc_rules = high_gc_template.build()
    low_gc_rules = low_gc_template.build()

    print("\nConstraint counts:")
    print(f"  Base: {len(base_rules.constraints)} constraints")
    print(f"  High-GC: {len(high_gc_rules.constraints)} constraints")
    print(f"  Low-GC: {len(low_gc_rules.constraints)} constraints")


def demonstrate_template_combination():
    """Show how to combine multiple templates."""
    print("\n" + "=" * 80)
    print("TEMPLATE COMBINATION")
    print("=" * 80)

    # Create individual templates
    promoter_template = PromoterTemplate(promoter_type="bacterial", length=(60, 100))

    codon_template = CodonOptimizedTemplate(organism="e_coli", length=(150, 200))

    print("Combining templates:")
    print("1. Bacterial promoter (60-100 bp)")
    print("2. E. coli codon optimization (150-200 bp)")

    # Build combined rules
    promoter_rules = promoter_template.build()
    codon_rules = codon_template.build()

    # Simple combination approach
    from symbiont import Rules

    combined_rules = Rules()

    # Add promoter constraints with modified weights
    for constraint in promoter_rules.constraints:
        combined_rules.constrain(constraint, weight=1.5)

    # Add codon constraints
    for constraint in codon_rules.constraints:
        combined_rules.constrain(constraint, weight=1.0)

    print("\nCombined constraint system:")
    print(f"  Total constraints: {len(combined_rules.constraints)}")
    print(f"  From promoter: {len(promoter_rules.constraints)}")
    print(f"  From codon opt: {len(codon_rules.constraints)}")

    # This would be used for generating promoter + gene constructs
    print("\nThis combined template could generate promoter + gene constructs")
    print("suitable for bacterial expression systems.")


def main():
    """Run the complete template system demonstration."""
    print("🧬 SYMBIONT CONSTRAINT TEMPLATE LIBRARY DEMO")
    print("Demonstrating pre-built templates for molecular biology applications")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Template discovery
        demonstrate_template_discovery()

        # 2. Primer design
        demonstrate_primer_design()

        # 3. CRISPR guide design
        demonstrate_crispr_design()

        # 4. Codon optimization
        demonstrate_codon_optimization()

        # 5. Template customization
        demonstrate_template_customization()

        # 6. Template combination
        demonstrate_template_combination()

        print("\n" + "=" * 80)
        print("TEMPLATE DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 80)
        print("\nKey benefits of the template system:")
        print("• Pre-built constraints for common molecular biology tasks")
        print("• Easy parameter customization for specific requirements")
        print("• Template discovery and search capabilities")
        print("• Scientific best practices encoded in reusable templates")
        print("• Extensible system for adding new domains and use cases")

        print("\nNext steps:")
        print("• Add protein design templates")
        print("• Include thermodynamic calculations")
        print("• Add template validation with experimental data")
        print("• Create web interface for template browsing")

    except Exception as e:
        print(f"❌ Template demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
