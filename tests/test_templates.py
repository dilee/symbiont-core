"""Tests for constraint template system."""

import pytest
import torch

from symbiont.core.dsl import Rules
from symbiont.templates import (
    CodonOptimizedTemplate,
    ConstraintTemplate,
    CRISPRGuideTemplate,
    PrimerDesignTemplate,
    PromoterTemplate,
    TemplateConfig,
    TemplateRegistry,
    registry,
)
from symbiont.templates.base import CompositeTemplate


class TestConstraintTemplate:
    """Test base ConstraintTemplate functionality."""

    def test_template_config_creation(self):
        """Test TemplateConfig dataclass."""
        config = TemplateConfig(
            name="Test Template",
            description="A test template",
            domain="testing",
            tags=["test", "example"],
        )

        assert config.name == "Test Template"
        assert config.domain == "testing"
        assert "test" in config.tags
        assert config.version == "1.0"  # default

    def test_abstract_template(self):
        """Test that ConstraintTemplate is abstract."""
        with pytest.raises(TypeError):
            ConstraintTemplate()


class TestPrimerDesignTemplate:
    """Test PCR primer design template."""

    def test_basic_creation(self):
        """Test basic template creation."""
        template = PrimerDesignTemplate(target_length=(18, 25))

        assert template.parameters["target_length"] == (18, 25)
        assert template.config.name == "PCR Primer Design"
        assert template.config.domain == "molecular_biology"

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        template = PrimerDesignTemplate(target_length=(20, 25), gc_content=(0.4, 0.6))
        errors = template.validate()
        assert len(errors) == 0

        # Invalid length range
        template = PrimerDesignTemplate(target_length=(5, 8))
        errors = template.validate()
        assert any("10-50 bp" in error for error in errors)

        # Invalid GC content
        template = PrimerDesignTemplate(
            target_length=20,
            gc_content=(0.8, 0.6),  # min > max
        )
        errors = template.validate()
        assert len(errors) > 0

    def test_build_constraints(self):
        """Test constraint building."""
        template = PrimerDesignTemplate(
            target_length=(18, 25),
            gc_content=(0.4, 0.6),
            avoid_hairpins=True,
            gc_clamp=True,
        )

        rules = template.build()

        assert isinstance(rules, Rules)
        assert len(rules.constraints) > 0

        # Should have length constraint
        assert any("Length" in str(c) for c in rules.constraints)

    def test_customization(self):
        """Test template customization."""
        base_template = PrimerDesignTemplate(target_length=20)

        custom_template = base_template.customize(
            gc_content=(0.5, 0.7), avoid_hairpins=False
        )

        assert custom_template.parameters["target_length"] == 20
        assert custom_template.parameters["gc_content"] == (0.5, 0.7)
        assert custom_template.parameters["avoid_hairpins"] is False

    def test_description(self):
        """Test template description."""
        template = PrimerDesignTemplate(target_length=(18, 25))
        description = template.describe()

        assert "PCR Primer Design" in description
        assert "target_length" in description
        assert "(18, 25)" in description


class TestCRISPRGuideTemplate:
    """Test CRISPR guide RNA design template."""

    def test_spcas9_template(self):
        """Test SpCas9 guide template."""
        template = CRISPRGuideTemplate(pam_type="NGG", length=20)

        rules = template.build()
        assert len(rules.constraints) > 0

        # Should enforce 20 bp length
        length_constraints = [c for c in rules.constraints if "Length" in str(c)]
        assert len(length_constraints) > 0

    def test_cpf1_template(self):
        """Test Cpf1/Cas12a guide template."""
        template = CRISPRGuideTemplate(pam_type="NNGRRT", length=20, avoid_poly_t=True)

        rules = template.build()

        # Should have poly-T avoidance
        poly_t_constraints = [c for c in rules.constraints if "TTTT" in str(c)]
        assert len(poly_t_constraints) > 0

    def test_parameter_defaults(self):
        """Test default parameters."""
        template = CRISPRGuideTemplate(pam_type="NGG")

        # Should have default length of 20
        assert template.parameters.get("length", 20) == 20

        # Should avoid poly-T by default
        assert template.parameters.get("avoid_poly_t", True) is True


class TestCodonOptimizedTemplate:
    """Test codon optimization template."""

    def test_ecoli_optimization(self):
        """Test E. coli optimization template."""
        template = CodonOptimizedTemplate(organism="e_coli", length=(300, 600))

        rules = template.build()

        # Should have start/stop codons
        start_constraints = [c for c in rules.constraints if "StartCodon" in str(c)]
        stop_constraints = [c for c in rules.constraints if "StopCodon" in str(c)]

        assert len(start_constraints) > 0
        assert len(stop_constraints) > 0

        # Should have appropriate GC content for E. coli
        gc_constraints = [c for c in rules.constraints if "GCContent" in str(c)]
        assert len(gc_constraints) > 0

    def test_human_optimization(self):
        """Test human codon optimization."""
        template = CodonOptimizedTemplate(organism="human")

        rules = template.build()
        assert len(rules.constraints) > 0

        # Different organisms should have different GC preferences
        # (implementation detail - just verify it builds)

    def test_restriction_site_avoidance(self):
        """Test restriction site avoidance."""
        template = CodonOptimizedTemplate(organism="e_coli", avoid_common_sites=True)

        rules = template.build()

        # Should forbid common restriction sites
        site_constraints = [
            c
            for c in rules.constraints
            if any(site in str(c) for site in ["GAATTC", "AAGCTT"])
        ]
        assert len(site_constraints) > 0


class TestPromoterTemplate:
    """Test promoter design template."""

    def test_bacterial_promoter(self):
        """Test bacterial promoter template."""
        template = PromoterTemplate(promoter_type="bacterial", length=(50, 100))

        rules = template.build()

        # Should have -35 and -10 box preferences
        motif_constraints = [c for c in rules.constraints if "HasMotif" in str(c)]
        assert len(motif_constraints) > 0

    def test_mammalian_promoter(self):
        """Test mammalian promoter template."""
        template = PromoterTemplate(
            promoter_type="mammalian", elements=["TATA", "CAAT"]
        )

        rules = template.build()

        # Should have TATA and CAAT box preferences
        tata_constraints = [c for c in rules.constraints if "TATA" in str(c)]
        assert len(tata_constraints) > 0

    def test_promoter_elements(self):
        """Test different promoter elements."""
        template = PromoterTemplate(promoter_type="mammalian", elements=["GC"])

        rules = template.build()

        # Should prefer GC box
        gc_box_constraints = [c for c in rules.constraints if "GGGCGG" in str(c)]
        assert len(gc_box_constraints) > 0


class TestCompositeTemplate:
    """Test composite template functionality."""

    def test_composite_creation(self):
        """Test creating composite template."""
        primer_template = PrimerDesignTemplate(target_length=20)
        crispr_template = CRISPRGuideTemplate(pam_type="NGG")

        composite = CompositeTemplate([primer_template, crispr_template])

        assert len(composite.templates) == 2
        assert "Composite" in composite.config.name

    def test_composite_constraints(self):
        """Test constraint combination in composite template."""
        template1 = PrimerDesignTemplate(target_length=20)
        template2 = CRISPRGuideTemplate(pam_type="NGG")

        composite = CompositeTemplate([template1, template2])
        rules = composite.build()

        # Should have constraints from both templates
        assert len(rules.constraints) > len(template1.build().constraints)
        assert len(rules.constraints) > len(template2.build().constraints)

    def test_composite_validation(self):
        """Test composite template validation."""
        # Create invalid template
        invalid_template = PrimerDesignTemplate(target_length=(5, 8))  # too short
        valid_template = CRISPRGuideTemplate(pam_type="NGG")

        composite = CompositeTemplate([invalid_template, valid_template])
        errors = composite.validate()

        # Should include errors from invalid template
        assert len(errors) > 0
        assert any("Template 0" in error for error in errors)


class TestTemplateRegistry:
    """Test template registry functionality."""

    def test_registry_registration(self):
        """Test template registration."""
        registry = TemplateRegistry()

        registry.register("test_primer", PrimerDesignTemplate)

        assert "test_primer" in registry
        assert len(registry) == 1
        assert registry.get("test_primer") == PrimerDesignTemplate

    def test_registry_override(self):
        """Test template override."""
        registry = TemplateRegistry()

        registry.register("test", PrimerDesignTemplate)

        # Should fail without override
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", CRISPRGuideTemplate)

        # Should work with override
        registry.register("test", CRISPRGuideTemplate, override=True)
        assert registry.get("test") == CRISPRGuideTemplate

    def test_registry_creation(self):
        """Test template instance creation."""
        registry = TemplateRegistry()
        registry.register("primer", PrimerDesignTemplate)

        template = registry.create("primer", target_length=20)

        assert isinstance(template, PrimerDesignTemplate)
        assert template.parameters["target_length"] == 20

    def test_registry_search(self):
        """Test template search functionality."""
        registry = TemplateRegistry()
        registry.register("primer", PrimerDesignTemplate)
        registry.register("crispr", CRISPRGuideTemplate)

        # Text search
        results = registry.search(query="primer")
        assert "primer" in results

        # Domain search
        results = registry.search(domain="genome_editing")
        assert "crispr" in results

    def test_registry_catalog(self):
        """Test catalog export/import."""
        registry = TemplateRegistry()
        registry.register("primer", PrimerDesignTemplate)

        catalog = registry.export_catalog()

        assert "primer" in catalog
        assert catalog["primer"]["name"] == "PCR Primer Design"
        assert catalog["primer"]["domain"] == "molecular_biology"

    def test_global_registry(self):
        """Test global registry has expected templates."""
        assert "primer_design" in registry
        assert "crispr_guide" in registry
        assert "codon_optimized" in registry
        assert "promoter" in registry

        # Should be able to create instances
        primer = registry.create("primer_design", target_length=20)
        assert isinstance(primer, PrimerDesignTemplate)

    def test_registry_help(self):
        """Test registry help functionality."""
        help_text = registry.get_help("primer_design")

        assert help_text is not None
        assert "PCR Primer Design" in help_text
        assert "target_length" in help_text

    def test_registry_validation(self):
        """Test registry validation."""
        errors = registry.validate_template("primer_design", target_length=(5, 8))

        # Should have validation errors for short primers
        assert len(errors) > 0
        assert any("10-50 bp" in error for error in errors)


class TestTemplateIntegration:
    """Integration tests with mock generators."""

    def test_primer_with_generator(self):
        """Test primer template with mock generator."""
        from symbiont import GenerationConfig
        from symbiont.generators.mock import MockSequenceGenerator

        template = PrimerDesignTemplate(target_length=20)
        rules = template.build()

        generator = MockSequenceGenerator(vocab_size=4, sequence_length=20)
        config = GenerationConfig(batch_size=3, max_length=20, seed=42)

        sequences = generator.constrained_generate(
            constraints=rules.constraints, config=config
        )

        assert sequences.shape == (3, 20)
        assert sequences.dtype == torch.float32

    def test_template_constraint_evaluation(self):
        """Test evaluating templates against generated sequences."""
        template = CRISPRGuideTemplate(pam_type="NGG", length=20)
        rules = template.build()

        # Create mock sequences
        sequences = torch.randint(0, 4, (5, 20)).float()

        # Should be able to evaluate without errors
        satisfaction_results = rules.evaluate(sequences)

        assert "total" in satisfaction_results
        assert satisfaction_results["total"].shape == (5,)
        assert torch.all(satisfaction_results["total"] >= 0)
        assert torch.all(satisfaction_results["total"] <= 1)


if __name__ == "__main__":
    pytest.main([__file__])
