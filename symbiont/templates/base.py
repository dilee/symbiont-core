"""Base classes for constraint templates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from symbiont.core.constraints import Constraint
from symbiont.core.dsl import Rules


@dataclass
class TemplateConfig:
    """Configuration for constraint templates."""

    name: str
    description: str
    domain: str = "general"
    version: str = "1.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


class ConstraintTemplate(ABC):
    """
    Abstract base class for constraint templates.

    Templates provide pre-configured constraint combinations for common
    scientific use cases, making the framework more accessible to domain experts.
    """

    def __init__(self, **kwargs: Any):
        """Initialize template with parameters."""
        self.config = self._get_config()
        self.parameters = self._validate_parameters(kwargs)

    @abstractmethod
    def _get_config(self) -> TemplateConfig:
        """Return template configuration metadata."""
        pass

    @abstractmethod
    def _build_constraints(
        self, **parameters: Any
    ) -> list[Constraint | tuple[Constraint, str, float]]:
        """Build the constraint list based on parameters."""
        pass

    def _validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Validate and set default values for parameters."""
        # Base implementation - subclasses can override for specific validation
        validated = {}
        for key, value in parameters.items():
            validated[key] = value
        return validated

    def build(self) -> Rules:
        """
        Build the complete Rules object with all constraints.

        Returns:
            Rules object ready to use with generators
        """
        constraints = self._build_constraints(**self.parameters)

        rules = Rules()

        # Apply constraints with appropriate weights based on template logic
        for constraint_spec in constraints:
            if isinstance(constraint_spec, tuple):
                constraint, constraint_type, weight = constraint_spec
                if constraint_type == "enforce":
                    rules.enforce(constraint, weight=weight)
                elif constraint_type == "constrain":
                    rules.constrain(constraint, weight=weight)
                elif constraint_type == "forbid":
                    rules.forbid(constraint, weight=weight)
                elif constraint_type == "prefer":
                    rules.prefer(constraint, weight=weight)
            else:
                # Default to soft constraint
                rules.constrain(constraint_spec)

        return rules

    def customize(self, **new_parameters: Any) -> ConstraintTemplate:
        """
        Create a customized version of the template with new parameters.

        Args:
            **new_parameters: Parameters to override

        Returns:
            New template instance with updated parameters
        """
        combined_params = {**self.parameters, **new_parameters}
        return self.__class__(**combined_params)

    def describe(self) -> str:
        """
        Return a human-readable description of the template and its configuration.

        Returns:
            Formatted description string
        """
        lines = [
            f"Template: {self.config.name}",
            f"Domain: {self.config.domain}",
            f"Description: {self.config.description}",
            "",
            "Current Parameters:",
        ]

        for key, value in self.parameters.items():
            lines.append(f"  {key}: {value}")

        if self.config.examples:
            lines.append("\nExample Usage:")
            for example in self.config.examples:
                lines.append(f"  {example}")

        return "\n".join(lines)

    def validate(self) -> list[str]:
        """
        Validate template parameters and return any issues found.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required parameters are present
        required = getattr(self, "required_parameters", [])
        for param in required:
            if param not in self.parameters:
                errors.append(f"Missing required parameter: {param}")

        # Validate parameter types and ranges
        errors.extend(self._validate_parameter_values())

        return errors

    def _validate_parameter_values(self) -> list[str]:
        """
        Validate parameter values. Override in subclasses for specific validation.

        Returns:
            List of validation errors
        """
        return []

    @property
    def info(self) -> TemplateConfig:
        """Get template configuration info."""
        return self.config

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}({param_str})"


class CompositeTemplate(ConstraintTemplate):
    """
    Template that combines multiple other templates.

    Useful for complex scenarios that require constraints from multiple domains.
    """

    def __init__(self, templates: list[ConstraintTemplate], **kwargs: Any):
        """
        Initialize composite template.

        Args:
            templates: List of templates to combine
            **kwargs: Additional parameters
        """
        self.templates = templates
        super().__init__(**kwargs)

    def _get_config(self) -> TemplateConfig:
        template_names = [t.config.name for t in self.templates]
        return TemplateConfig(
            name=f"Composite({', '.join(template_names)})",
            description=f"Combination of: {', '.join(template_names)}",
            domain="composite",
            tags=["composite", "multi-domain"],
        )

    def _build_constraints(
        self, **parameters: Any
    ) -> list[Constraint | tuple[Constraint, str, float]]:
        """Combine constraints from all templates."""
        all_constraints: list[Constraint | tuple[Constraint, str, float]] = []

        for template in self.templates:
            template_constraints = template._build_constraints(**parameters)
            all_constraints.extend(template_constraints)

        return all_constraints

    def validate(self) -> list[str]:
        """Validate all component templates."""
        errors = super().validate()

        for i, template in enumerate(self.templates):
            template_errors = template.validate()
            for error in template_errors:
                errors.append(f"Template {i} ({template.config.name}): {error}")

        return errors
