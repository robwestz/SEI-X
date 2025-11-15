"""
Plugin system for extending SIE-X functionality.
"""

from typing import List, Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod
import importlib
import inspect
from pathlib import Path
import yaml
import ast


class PluginInterface(ABC):
    """Base interface for all SIE-X plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass

    @abstractmethod
    def initialize(self, engine: 'SemanticIntelligenceEngine', config: Dict[str, Any]):
        """Initialize plugin with engine reference."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up plugin resources."""
        pass


class ExtractorPlugin(PluginInterface):
    """Base class for custom keyword extractors."""

    @abstractmethod
    async def extract(
            self,
            text: str,
            options: Dict[str, Any]
    ) -> List['Keyword']:
        """Extract keywords using custom logic."""
        pass

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ['en']  # Default to English

    def get_capabilities(self) -> Dict[str, Any]:
        """Get plugin capabilities."""
        return {
            'async': True,
            'batch_processing': False,
            'streaming': False
        }


class ProcessorPlugin(PluginInterface):
    """Base class for post-processing plugins."""

    @abstractmethod
    async def process(
            self,
            keywords: List['Keyword'],
            context: Dict[str, Any]
    ) -> List['Keyword']:
        """Process extracted keywords."""
        pass


class PluginManager:
    """Manages plugin lifecycle and execution."""

    def __init__(self, plugin_dir: Path = Path("plugins")):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, PluginInterface] = {}
        self.extractors: Dict[str, ExtractorPlugin] = {}
        self.processors: Dict[str, ProcessorPlugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}

    def discover_plugins(self):
        """Discover and load plugins from plugin directory."""
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory {self.plugin_dir} does not exist")
            return

        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                self._load_plugin_from_file(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")

    def _load_plugin_from_file(self, file_path: Path):
        """Load plugin from Python file."""
        # Read plugin metadata
        metadata = self._extract_plugin_metadata(file_path)
        if not metadata:
            return

        # Validate plugin
        if not self._validate_plugin_metadata(metadata):
            logger.warning(f"Invalid plugin metadata in {file_path}")
            return

        # Load module
        spec = importlib.util.spec_from_file_location(
            f"sie_x_plugin_{file_path.stem}",
            file_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find plugin classes
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                    issubclass(obj, PluginInterface) and
                    obj != PluginInterface):
                # Instantiate plugin
                plugin = obj()
                self.register_plugin(plugin)

    def register_plugin(self, plugin: PluginInterface):
        """Register a plugin instance."""
        plugin_id = f"{plugin.name}:{plugin.version}"

        if plugin_id in self.plugins:
            logger.warning(f"Plugin {plugin_id} already registered")
            return

        self.plugins[plugin_id] = plugin

        # Categorize plugin
        if isinstance(plugin, ExtractorPlugin):
            self.extractors[plugin.name] = plugin
        elif isinstance(plugin, ProcessorPlugin):
            self.processors[plugin.name] = plugin

        logger.info(f"Registered plugin: {plugin_id}")

    def get_plugin(self, name: str, version: Optional[str] = None) -> Optional[PluginInterface]:
        """Get plugin by name and optional version."""
        if version:
            return self.plugins.get(f"{name}:{version}")

        # Return latest version
        matches = [p for pid, p in self.plugins.items() if pid.startswith(f"{name}:")]
        if matches:
            return matches[-1]  # Assume last is latest

        return None

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        return [
            {
                'name': plugin.name,
                'version': plugin.version,
                'description': plugin.description,
                'type': type(plugin).__name__
            }
            for plugin in self.plugins.values()
        ]

    def initialize_all(self, engine: 'SemanticIntelligenceEngine', config: Dict[str, Any]):
        """Initialize all plugins."""
        for plugin in self.plugins.values():
            try:
                plugin.initialize(engine, config.get(plugin.name, {}))
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin.name}: {e}")

    def cleanup_all(self):
        """Clean up all plugins."""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup plugin {plugin.name}: {e}")

    def register_hook(self, hook_name: str, callback: Callable):
        """Register a hook callback."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)

    async def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all callbacks for a hook."""
        if hook_name not in self.hooks:
            return []

        results = []
        for callback in self.hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook {hook_name} callback failed: {e}")

        return results

    def _extract_plugin_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract plugin metadata from file."""
        with open(file_path, 'r') as f:
            content = f.read()

        # Look for metadata in docstring
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if docstring and "PLUGIN_METADATA:" in docstring:
                    # Parse YAML metadata
                    metadata_str = docstring.split("PLUGIN_METADATA:")[1].strip()
                    return yaml.safe_load(metadata_str)

        return None

    def _validate_plugin_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate plugin metadata."""
        required_fields = ['name', 'version', 'type', 'author']
        return all(field in metadata for field in required_fields)


# Example Custom Plugins

class DomainSpecificExtractor(ExtractorPlugin):
    """
    Example domain-specific keyword extractor.

    PLUGIN_METADATA:
        name: domain_extractor
        version: 1.0.0
        type: extractor
        author: SIE-X Team
        description: Extract domain-specific keywords using custom rules
    """

    @property
    def name(self) -> str:
        return "domain_extractor"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Extract domain-specific keywords using custom rules"

    def initialize(self, engine: 'SemanticIntelligenceEngine', config: Dict[str, Any]):
        """Initialize with domain configuration."""
        self.engine = engine
        self.domain_rules = config.get('rules', {})
        self.domain_vocabulary = config.get('vocabulary', [])

    async def extract(
            self,
            text: str,
            options: Dict[str, Any]
    ) -> List['Keyword']:
        """Extract keywords using domain rules."""
        keywords = []

        # Use base engine for initial extraction
        base_keywords = await self.engine.extract_async(text, **options)

        # Apply domain-specific filtering
        for kw in base_keywords:
            if self._is_domain_relevant(kw):
                # Boost score for domain terms
                if kw.text in self.domain_vocabulary:
                    kw.score *= 1.5
                keywords.append(kw)

        # Add domain-specific terms not caught by base extractor
        domain_terms = self._extract_domain_terms(text)
        for term in domain_terms:
            if not any(kw.text == term for kw in keywords):
                keywords.append(Keyword(
                    text=term,
                    score=0.8,
                    type='DOMAIN',
                    count=1,
                    related_terms=[]
                ))

        return sorted(keywords, key=lambda k: k.score, reverse=True)

    def _is_domain_relevant(self, keyword: 'Keyword') -> bool:
        """Check if keyword is domain relevant."""
        # Implement domain-specific logic
        return True

    def _extract_domain_terms(self, text: str) -> List[str]:
        """Extract domain-specific terms using rules."""
        # Implement rule-based extraction
        return []

    def cleanup(self):
        """Clean up resources."""
        pass


class AcademicCitationProcessor(ProcessorPlugin):
    """
    Process keywords to extract academic citations.

    PLUGIN_METADATA:
        name: citation_processor
        version: 1.0.0
        type: processor
        author: SIE-X Team
        description: Extract and process academic citations from keywords
    """

    @property
    def name(self) -> str:
        return "citation_processor"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Extract and process academic citations from keywords"

    def initialize(self, engine: 'SemanticIntelligenceEngine', config: Dict[str, Any]):
        """Initialize processor."""
        self.engine = engine
        self.citation_patterns = config.get('patterns', [])

    async def process(
            self,
            keywords: List['Keyword'],
            context: Dict[str, Any]
    ) -> List['Keyword']:
        """Process keywords to identify citations."""
        processed = []

        for kw in keywords:
            # Check if keyword looks like a citation
            if self._is_citation(kw.text):
                kw.type = 'CITATION'

                # Extract citation components
                components = self._parse_citation(kw.text)
                kw.metadata = components

                # Find related citations
                kw.related_terms = self._find_related_citations(kw, keywords)

            processed.append(kw)

        return processed

    def _is_citation(self, text: str) -> bool:
        """Check if text is a citation."""
        # Simple heuristic - contains year in parentheses
        import re
        return bool(re.search(r'\(\d{4}\)', text))

    def _parse_citation(self, citation: str) -> Dict[str, str]:
        """Parse citation components."""
        # Simplified parsing
        import re

        year_match = re.search(r'\((\d{4})\)', citation)
        year = year_match.group(1) if year_match else None

        authors = citation.split('(')[0].strip() if '(' in citation else citation

        return {
            'authors': authors,
            'year': year,
            'full_citation': citation
        }

    def _find_related_citations(
            self,
            citation: 'Keyword',
            all_keywords: List['Keyword']
    ) -> List[str]:
        """Find related citations."""
        related = []

        citation_year = citation.metadata.get('year')
        if citation_year:
            for kw in all_keywords:
                if (kw != citation and
                        hasattr(kw, 'metadata') and
                        kw.metadata.get('year') == citation_year):
                    related.append(kw.text)

        return related

    def cleanup(self):
        """Clean up resources."""
        pass


# Plugin Configuration Schema
PLUGIN_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "plugins": {
            "type": "object",
            "patternProperties": {
                "^.*$": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "version": {"type": "string"},
                        "config": {"type": "object"}
                    }
                }
            }
        },
        "plugin_directory": {"type": "string"},
        "auto_discover": {"type": "boolean"}
    }
}