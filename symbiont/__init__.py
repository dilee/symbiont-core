"""
Symbiont: Neuro-symbolic framework for constraint-guided generation.

A framework that adds a "constraint layer" to any generative AI model,
allowing users to guide generation with explicit rules instead of
filtering outputs post-generation.
"""

__version__ = "0.1.0"
__author__ = "Symbiont Team"
__email__ = "hello@symbiont.dev"

# Core exports
from symbiont.core.constraints import Constraint
from symbiont.core.dsl import Rules
from symbiont.core.types import GenerationConfig
from symbiont.generators.base import Generator

__all__ = [
    "Constraint",
    "Rules",
    "GenerationConfig",
    "Generator",
]
