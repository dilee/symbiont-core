"""
Constraint Template Library for Common Scientific Use Cases.

This module provides pre-built constraint templates for common scientific
applications, making it easier for researchers to get started with Symbiont
without needing to implement constraints from scratch.

Example:
    >>> from symbiont.templates import PrimerDesignTemplate
    >>> template = PrimerDesignTemplate(
    ...     target_length=(18, 25),
    ...     target_tm=(55, 65),
    ...     max_hairpin_temp=40
    ... )
    >>> rules = template.build()
    >>> # Use rules with your generator
"""

from symbiont.templates.base import ConstraintTemplate, TemplateConfig
from symbiont.templates.dna_templates import (
    CodonOptimizedTemplate,
    CRISPRGuideTemplate,
    PrimerDesignTemplate,
    PromoterTemplate,
)
from symbiont.templates.registry import TemplateRegistry

# Initialize global registry
registry = TemplateRegistry()

# Register built-in templates
registry.register("primer_design", PrimerDesignTemplate)
registry.register("crispr_guide", CRISPRGuideTemplate)
registry.register("codon_optimized", CodonOptimizedTemplate)
registry.register("promoter", PromoterTemplate)

__all__ = [
    "ConstraintTemplate",
    "TemplateConfig",
    "PrimerDesignTemplate",
    "CRISPRGuideTemplate",
    "CodonOptimizedTemplate",
    "PromoterTemplate",
    "TemplateRegistry",
    "registry",
]
