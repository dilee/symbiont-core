"""Template registry for discovery and management of constraint templates."""

from __future__ import annotations

from typing import Any

from symbiont.templates.base import ConstraintTemplate, TemplateConfig


class TemplateRegistry:
    """
    Registry for managing and discovering constraint templates.

    Provides functionality to:
    - Register new templates
    - Search and filter templates
    - Get template information
    - Validate template compatibility
    """

    def __init__(self):
        """Initialize empty registry."""
        self._templates: dict[str, type[ConstraintTemplate]] = {}
        self._metadata: dict[str, TemplateConfig] = {}

    def register(
        self,
        name: str,
        template_class: type[ConstraintTemplate],
        override: bool = False,
    ) -> None:
        """
        Register a template class.

        Args:
            name: Unique name for the template
            template_class: Template class to register
            override: Whether to override existing registration

        Raises:
            ValueError: If name already exists and override=False
        """
        if name in self._templates and not override:
            raise ValueError(
                f"Template '{name}' already registered. Use override=True to replace."
            )

        if not issubclass(template_class, ConstraintTemplate):
            raise TypeError("template_class must be a subclass of ConstraintTemplate")

        # Get metadata by instantiating with minimal parameters
        try:
            temp_instance = template_class()
            metadata = temp_instance._get_config()
        except Exception:
            # If can't instantiate without parameters, create minimal metadata
            metadata = TemplateConfig(
                name=name,
                description=f"Template class: {template_class.__name__}",
                domain="unknown",
            )

        self._templates[name] = template_class
        self._metadata[name] = metadata

    def unregister(self, name: str) -> bool:
        """
        Unregister a template.

        Args:
            name: Name of template to unregister

        Returns:
            True if template was found and removed, False otherwise
        """
        if name in self._templates:
            del self._templates[name]
            del self._metadata[name]
            return True
        return False

    def get(self, name: str) -> type[ConstraintTemplate] | None:
        """
        Get template class by name.

        Args:
            name: Template name

        Returns:
            Template class if found, None otherwise
        """
        return self._templates.get(name)

    def create(self, name: str, **parameters: Any) -> ConstraintTemplate | None:
        """
        Create template instance by name.

        Args:
            name: Template name
            **parameters: Parameters to pass to template constructor

        Returns:
            Template instance if found, None otherwise
        """
        template_class = self.get(name)
        if template_class is None:
            return None

        return template_class(**parameters)

    def list_templates(self) -> list[str]:
        """
        List all registered template names.

        Returns:
            List of template names
        """
        return list(self._templates.keys())

    def search(
        self,
        query: str | None = None,
        domain: str | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """
        Search for templates matching criteria.

        Args:
            query: Text to search in name/description
            domain: Domain to filter by
            tags: Tags to filter by (all must be present)

        Returns:
            List of matching template names
        """
        results = []

        for name, metadata in self._metadata.items():
            # Text search
            if query is not None:
                query_lower = query.lower()
                if (
                    query_lower not in metadata.name.lower()
                    and query_lower not in metadata.description.lower()
                ):
                    continue

            # Domain filter
            if domain is not None and metadata.domain != domain:
                continue

            # Tags filter
            if tags is not None and not all(tag in metadata.tags for tag in tags):
                continue

            results.append(name)

        return results

    def get_info(self, name: str) -> TemplateConfig | None:
        """
        Get template metadata.

        Args:
            name: Template name

        Returns:
            Template configuration if found, None otherwise
        """
        return self._metadata.get(name)

    def list_domains(self) -> list[str]:
        """
        List all unique domains.

        Returns:
            List of domain names
        """
        domains = {metadata.domain for metadata in self._metadata.values()}
        return sorted(domains)

    def list_tags(self) -> list[str]:
        """
        List all unique tags.

        Returns:
            List of tag names
        """
        tags = set()
        for metadata in self._metadata.values():
            tags.update(metadata.tags)
        return sorted(tags)

    def validate_template(self, name: str, **parameters: Any) -> list[str]:
        """
        Validate template with given parameters.

        Args:
            name: Template name
            **parameters: Parameters to validate

        Returns:
            List of validation errors (empty if valid)
        """
        template = self.create(name, **parameters)
        if template is None:
            return [f"Template '{name}' not found"]

        return template.validate()

    def get_help(self, name: str) -> str | None:
        """
        Get help text for a template.

        Args:
            name: Template name

        Returns:
            Help text if template found, None otherwise
        """
        template_class = self.get(name)
        if template_class is None:
            return None

        try:
            temp_instance = template_class()
            return temp_instance.describe()
        except Exception:
            metadata = self.get_info(name)
            if metadata:
                return f"Template: {metadata.name}\nDescription: {metadata.description}"
            return f"Template: {name} (no description available)"

    def export_catalog(self) -> dict[str, dict[str, Any]]:
        """
        Export complete template catalog.

        Returns:
            Dictionary mapping template names to their metadata
        """
        catalog = {}

        for name, metadata in self._metadata.items():
            catalog[name] = {
                "name": metadata.name,
                "description": metadata.description,
                "domain": metadata.domain,
                "version": metadata.version,
                "author": metadata.author,
                "tags": metadata.tags,
                "parameters": metadata.parameters,
                "examples": metadata.examples,
                "references": metadata.references,
            }

        return catalog

    def import_catalog(self, catalog: dict[str, dict[str, Any]]) -> list[str]:
        """
        Import template metadata from catalog (does not register classes).

        Args:
            catalog: Template catalog dictionary

        Returns:
            List of imported template names
        """
        imported = []

        for name, data in catalog.items():
            if name not in self._templates:
                # Create placeholder metadata for external templates
                metadata = TemplateConfig(
                    name=data.get("name", name),
                    description=data.get("description", ""),
                    domain=data.get("domain", "unknown"),
                    version=data.get("version", "unknown"),
                    author=data.get("author", ""),
                    tags=data.get("tags", []),
                    parameters=data.get("parameters", {}),
                    examples=data.get("examples", []),
                    references=data.get("references", []),
                )
                self._metadata[name] = metadata
                imported.append(name)

        return imported

    def __len__(self) -> int:
        """Return number of registered templates."""
        return len(self._templates)

    def __contains__(self, name: str) -> bool:
        """Check if template is registered."""
        return name in self._templates

    def __iter__(self):
        """Iterate over template names."""
        return iter(self._templates.keys())

    def __repr__(self) -> str:
        return f"TemplateRegistry({len(self._templates)} templates)"
