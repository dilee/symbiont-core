"""DNA and RNA sequence constraint templates for common molecular biology applications."""

from __future__ import annotations

from typing import Any

from symbiont.core.constraints import Constraint
from symbiont.domains.sequence import (
    Contains,
    GCContent,
    HasMotif,
    Length,
    NoRepeats,
    StartCodon,
    StopCodon,
)
from symbiont.templates.base import ConstraintTemplate, TemplateConfig


class PrimerDesignTemplate(ConstraintTemplate):
    """
    Template for PCR primer design with standard molecular biology constraints.

    Implements constraints for:
    - Optimal length range
    - GC content for stable hybridization
    - Melting temperature considerations
    - Avoiding problematic sequences
    """

    required_parameters = ["target_length"]

    def _get_config(self) -> TemplateConfig:
        return TemplateConfig(
            name="PCR Primer Design",
            description="Constraints for designing effective PCR primers",
            domain="molecular_biology",
            tags=["pcr", "primer", "amplification", "dna"],
            author="Symbiont Team",
            examples=[
                {"target_length": (18, 25), "gc_content": (0.4, 0.6)},
                {
                    "target_length": (20, 30),
                    "gc_content": (0.45, 0.65),
                    "avoid_hairpins": True,
                },
            ],
            references=[
                "Primer Design Guidelines - NEB",
                "PCR: A Practical Approach - Oxford University Press",
            ],
        )

    def _validate_parameter_values(self) -> list[str]:
        """Validate primer design parameters."""
        errors = []

        if "target_length" in self.parameters:
            length = self.parameters["target_length"]
            if isinstance(length, tuple) and len(length) == 2:
                min_len, max_len = length
                if min_len < 10 or max_len > 50:
                    errors.append(
                        "Primer length should be between 10-50 bp for practical use"
                    )
                if min_len >= max_len:
                    errors.append("Minimum length must be less than maximum length")
            elif isinstance(length, int):
                if length < 10 or length > 50:
                    errors.append("Primer length should be between 10-50 bp")
            else:
                errors.append("target_length must be int or tuple of (min, max)")

        if "gc_content" in self.parameters:
            gc = self.parameters["gc_content"]
            if isinstance(gc, tuple) and len(gc) == 2:
                min_gc, max_gc = gc
                if not (0 <= min_gc <= max_gc <= 1):
                    errors.append("GC content must be between 0 and 1")
            else:
                errors.append("gc_content must be tuple of (min, max)")

        return errors

    def _build_constraints(
        self, **parameters: Any
    ) -> list[tuple[Constraint, str, float]]:
        """Build primer design constraints."""
        constraints = []

        # Length constraint (required)
        target_length = parameters["target_length"]
        if isinstance(target_length, tuple):
            min_len, max_len = target_length
            constraints.append((Length(min_len, max_len), "enforce", 8.0))
        else:
            constraints.append((Length(exactly=target_length), "enforce", 8.0))

        # GC content for stability
        gc_content = parameters.get("gc_content", (0.4, 0.65))
        min_gc, max_gc = gc_content
        constraints.append((GCContent(min_gc, max_gc), "constrain", 3.0))

        # Avoid long repeats that can cause mispriming
        max_repeat = parameters.get("max_repeat_length", 3)
        constraints.append((NoRepeats(max_repeat), "constrain", 2.0))

        # Avoid problematic sequences if specified
        if parameters.get("avoid_hairpins", True):
            # Avoid sequences that can form strong hairpins
            constraints.append((Contains("AAAA"), "forbid", 2.0))
            constraints.append((Contains("TTTT"), "forbid", 2.0))
            constraints.append((Contains("GGGG"), "forbid", 2.0))
            constraints.append((Contains("CCCC"), "forbid", 2.0))

        # Prefer 3' GC clamp for stability
        if parameters.get("gc_clamp", True):
            constraints.append((HasMotif("[GC][GC]$"), "prefer", 1.0))

        # Avoid restriction sites if specified
        avoid_sites = parameters.get("avoid_restriction_sites", [])
        for site in avoid_sites:
            constraints.append((Contains(site), "forbid", 3.0))

        return constraints


class CRISPRGuideTemplate(ConstraintTemplate):
    """
    Template for CRISPR guide RNA design.

    Implements constraints for:
    - Proper length (typically 20 bp)
    - GC content for activity
    - Avoiding problematic sequences
    - PAM site compatibility
    """

    required_parameters = ["pam_type"]

    def _get_config(self) -> TemplateConfig:
        return TemplateConfig(
            name="CRISPR Guide RNA Design",
            description="Constraints for designing effective CRISPR guide RNAs",
            domain="genome_editing",
            tags=["crispr", "guide_rna", "genome_editing", "cas9"],
            examples=[
                {"pam_type": "NGG", "length": 20},
                {"pam_type": "NNGRRT", "length": 20, "avoid_poly_t": True},
            ],
        )

    def _build_constraints(
        self, **parameters: Any
    ) -> list[tuple[Constraint, str, float]]:
        """Build CRISPR guide RNA constraints."""
        constraints = []

        # Standard guide RNA length
        guide_length = parameters.get("length", 20)
        constraints.append((Length(exactly=guide_length), "enforce", 10.0))

        # Optimal GC content for activity (typically 30-70%)
        gc_content = parameters.get("gc_content", (0.3, 0.7))
        min_gc, max_gc = gc_content
        constraints.append((GCContent(min_gc, max_gc), "constrain", 4.0))

        # Avoid poly-T sequences (can cause premature termination)
        if parameters.get("avoid_poly_t", True):
            constraints.append((Contains("TTTT"), "forbid", 5.0))

        # Avoid long repeats
        constraints.append((NoRepeats(3), "constrain", 2.0))

        # PAM-specific constraints
        pam_type = parameters["pam_type"]
        if pam_type == "NGG":  # SpCas9
            # No specific sequence constraint for target, but prefer certain features
            constraints.append(
                (Contains("GG$"), "prefer", 1.0)
            )  # Ends with GG part of NGG
        elif pam_type == "NNGRRT":  # Cpf1/Cas12a
            # Cpf1 preferences
            constraints.append((Contains("TTT"), "prefer", 1.0))  # TTTV PAM preference

        # Avoid sequences with extreme GC skew
        constraints.append((Contains("GGGGGG"), "forbid", 3.0))
        constraints.append((Contains("CCCCCC"), "forbid", 3.0))

        return constraints


class CodonOptimizedTemplate(ConstraintTemplate):
    """
    Template for codon-optimized gene sequences.

    Implements constraints for:
    - Start/stop codons
    - Optimal codon usage
    - Avoiding problematic sequences
    - Proper reading frame
    """

    required_parameters = ["organism"]

    def _get_config(self) -> TemplateConfig:
        return TemplateConfig(
            name="Codon Optimization",
            description="Constraints for codon-optimized gene expression",
            domain="protein_expression",
            tags=["codon_optimization", "expression", "protein", "translation"],
            examples=[
                {"organism": "e_coli", "length": (300, 1500)},
                {"organism": "human", "length": (150, 2000), "avoid_repeats": True},
            ],
        )

    def _build_constraints(
        self, **parameters: Any
    ) -> list[tuple[Constraint, str, float]]:
        """Build codon optimization constraints."""
        constraints = []

        # Must have start codon
        constraints.append((StartCodon(), "enforce", 10.0))

        # Must have stop codon
        constraints.append((StopCodon(), "enforce", 10.0))

        # Length constraint if provided
        if "length" in parameters:
            length = parameters["length"]
            if isinstance(length, tuple):
                min_len, max_len = length
                constraints.append((Length(min_len, max_len), "constrain", 3.0))
            else:
                constraints.append((Length(exactly=length), "constrain", 3.0))

        # GC content optimization based on organism
        organism = parameters["organism"]
        if organism == "e_coli":
            constraints.append((GCContent(0.45, 0.65), "constrain", 2.0))
        elif organism == "human":
            constraints.append((GCContent(0.40, 0.60), "constrain", 2.0))
        elif organism == "yeast":
            constraints.append((GCContent(0.35, 0.55), "constrain", 2.0))

        # Avoid problematic sequences
        if parameters.get("avoid_repeats", True):
            constraints.append((NoRepeats(4), "constrain", 2.0))

        # Avoid restriction sites commonly used in cloning
        common_sites = parameters.get("avoid_common_sites", True)
        if common_sites:
            for site in [
                "GAATTC",
                "AAGCTT",
                "GGATCC",
                "CTCGAG",
            ]:  # EcoRI, HindIII, BamHI, XhoI
                constraints.append((Contains(site), "forbid", 1.5))

        # Avoid poly-A sequences (can cause issues in some systems)
        constraints.append((Contains("AAAAA"), "forbid", 2.0))

        return constraints


class PromoterTemplate(ConstraintTemplate):
    """
    Template for promoter sequence design.

    Implements constraints for:
    - Core promoter elements
    - Transcription factor binding sites
    - Proper spacing and positioning
    """

    required_parameters = ["promoter_type"]

    def _get_config(self) -> TemplateConfig:
        return TemplateConfig(
            name="Promoter Design",
            description="Constraints for functional promoter sequences",
            domain="gene_regulation",
            tags=["promoter", "transcription", "regulation", "expression"],
            examples=[
                {"promoter_type": "bacterial", "strength": "strong"},
                {"promoter_type": "mammalian", "elements": ["TATA", "CAAT"]},
            ],
        )

    def _build_constraints(
        self, **parameters: Any
    ) -> list[tuple[Constraint, str, float]]:
        """Build promoter design constraints."""
        constraints = []

        promoter_type = parameters["promoter_type"]

        if promoter_type == "bacterial":
            # Bacterial promoter elements
            constraints.append(
                (HasMotif("TTGACA"), "prefer", 3.0)
            )  # -35 box (consensus)
            constraints.append(
                (HasMotif("TATAAT"), "prefer", 4.0)
            )  # -10 box (Pribnow box)

            # Length constraint for typical bacterial promoter
            length = parameters.get("length", (50, 150))
            if isinstance(length, tuple):
                min_len, max_len = length
                constraints.append((Length(min_len, max_len), "constrain", 2.0))

        elif promoter_type == "mammalian":
            # Mammalian promoter elements
            elements = parameters.get("elements", ["TATA"])

            if "TATA" in elements:
                constraints.append((HasMotif("TATAAA"), "prefer", 4.0))  # TATA box
            if "CAAT" in elements:
                constraints.append((HasMotif("GGCCAATCT"), "prefer", 2.0))  # CAAT box
            if "GC" in elements:
                constraints.append((HasMotif("GGGCGG"), "prefer", 2.0))  # GC box

            # Mammalian promoters are typically longer
            length = parameters.get("length", (100, 500))
            if isinstance(length, tuple):
                min_len, max_len = length
                constraints.append((Length(min_len, max_len), "constrain", 2.0))

        # General constraints
        gc_content = parameters.get("gc_content", (0.3, 0.7))
        min_gc, max_gc = gc_content
        constraints.append((GCContent(min_gc, max_gc), "constrain", 1.0))

        # Avoid long repeats that might interfere with binding
        constraints.append((NoRepeats(5), "constrain", 1.0))

        return constraints


class RiboswitchTemplate(ConstraintTemplate):
    """
    Template for riboswitch design (RNA regulatory elements).

    Implements constraints for:
    - Aptamer domain structure
    - Expression platform elements
    - Secondary structure considerations
    """

    required_parameters = ["target_ligand"]

    def _get_config(self) -> TemplateConfig:
        return TemplateConfig(
            name="Riboswitch Design",
            description="Constraints for functional riboswitch sequences",
            domain="rna_regulation",
            tags=["riboswitch", "rna", "regulation", "aptamer"],
            examples=[
                {"target_ligand": "theophylline", "length": (80, 120)},
                {"target_ligand": "ATP", "include_terminator": True},
            ],
        )

    def _build_constraints(
        self, **parameters: Any
    ) -> list[tuple[Constraint, str, float]]:
        """Build riboswitch design constraints."""
        constraints = []

        # Length constraint for typical riboswitch
        length = parameters.get("length", (60, 150))
        if isinstance(length, tuple):
            min_len, max_len = length
            constraints.append((Length(min_len, max_len), "constrain", 3.0))

        # GC content suitable for stable secondary structure
        gc_content = parameters.get("gc_content", (0.45, 0.65))
        min_gc, max_gc = gc_content
        constraints.append((GCContent(min_gc, max_gc), "constrain", 2.0))

        # Avoid sequences that prevent proper folding
        constraints.append((NoRepeats(4), "constrain", 2.0))
        constraints.append((Contains("UUUUUU"), "forbid", 3.0))  # RNA poly-U terminator

        # Ligand-specific aptamer constraints
        target_ligand = parameters["target_ligand"]
        if target_ligand == "theophylline":
            # Known theophylline aptamer motifs
            constraints.append((HasMotif("GAUACCAG"), "prefer", 3.0))
        elif target_ligand == "ATP":
            # ATP binding motifs
            constraints.append((HasMotif("GGGAGA"), "prefer", 2.0))

        # Include terminator if specified
        if parameters.get("include_terminator", False):
            constraints.append(
                (HasMotif("UUUUUU"), "prefer", 2.0)
            )  # Rho-independent terminator

        return constraints
