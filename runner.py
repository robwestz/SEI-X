"""
SIE-X SystemRunner — Agnostic pipeline driven by swappable SYSTEM.md files.

Architecture pattern (from BACOWR v5):
    SYSTEM.md  = "personligheten" — VAD systemet gör, vilka regler som gäller
    runner.py  = agnostisk motor  — HUR det exekveras
    models.py  = typade strukturer — dataflödet genom pipelinen

Usage:
    # CLI
    python runner.py seo-bridges --input data.csv
    python runner.py medical-dx --input patient_note.txt
    python runner.py legal-review --input contract.pdf

    # Python
    from sie_x.runner import SystemRunner
    runner = SystemRunner("systems/seo-bridges.md")
    result = await runner.run({"text": "..."})

    # Swap personality, keep engine
    runner.load_system("systems/medical-dx.md")
    result = await runner.run({"text": "..."})
"""

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ============================================================
# SYSTEM DEFINITION — parsed from SYSTEM.md
# ============================================================

@dataclass
class SystemDefinition:
    """Parsed representation of a SYSTEM.md file."""

    # Identity
    name: str = ""
    version: str = "1.0"
    description: str = ""

    # Engine configuration
    engine_mode: str = "balanced"
    embedding_model: str = "all-MiniLM-L6-v2"
    min_confidence: float = 0.3
    top_k: int = 20

    # Which transformers to load
    transformers: List[str] = field(default_factory=list)

    # Pipeline stages (ordered)
    stages: List[str] = field(default_factory=list)

    # Input/output spec
    input_format: str = "text"          # text, csv, json, url
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_format: str = "json"         # json, markdown, html
    output_directory: str = "output/"

    # Domain rules (markdown body — for LLM agents)
    rules_markdown: str = ""

    # Raw frontmatter for stage-specific config
    raw_config: Dict[str, Any] = field(default_factory=dict)


def parse_system_file(path: str) -> SystemDefinition:
    """Parse a SYSTEM.md file into a SystemDefinition.

    Format: YAML frontmatter between --- delimiters, then markdown body.

    ---
    name: SEO Bridge Analyzer
    engine:
      mode: balanced
    stages:
      - extract_keywords
      - find_bridges
    ---

    # Domain Rules
    (markdown body — read by LLM agents)
    """
    text = Path(path).read_text(encoding="utf-8")

    # Split frontmatter from body
    frontmatter = {}
    body = text

    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', text, re.DOTALL)
    if fm_match and YAML_AVAILABLE:
        frontmatter = yaml.safe_load(fm_match.group(1)) or {}
        body = fm_match.group(2)
    elif fm_match:
        # Fallback: basic key: value parsing if yaml not installed
        for line in fm_match.group(1).split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                frontmatter[key.strip()] = val.strip()
        body = fm_match.group(2)

    engine_cfg = frontmatter.get("engine", {})
    input_cfg = frontmatter.get("input", {})
    output_cfg = frontmatter.get("output", {})

    return SystemDefinition(
        name=frontmatter.get("name", Path(path).stem),
        version=str(frontmatter.get("version", "1.0")),
        description=frontmatter.get("description", ""),
        engine_mode=engine_cfg.get("mode", "balanced") if isinstance(engine_cfg, dict) else "balanced",
        embedding_model=engine_cfg.get("embedding_model", "all-MiniLM-L6-v2") if isinstance(engine_cfg, dict) else "all-MiniLM-L6-v2",
        min_confidence=float(engine_cfg.get("min_confidence", 0.3)) if isinstance(engine_cfg, dict) else 0.3,
        top_k=int(engine_cfg.get("top_k", 20)) if isinstance(engine_cfg, dict) else 20,
        transformers=frontmatter.get("transformers", []),
        stages=frontmatter.get("stages", []),
        input_format=input_cfg.get("format", "text") if isinstance(input_cfg, dict) else "text",
        input_schema=input_cfg.get("schema", {}) if isinstance(input_cfg, dict) else {},
        output_format=output_cfg.get("format", "json") if isinstance(output_cfg, dict) else "json",
        output_directory=output_cfg.get("directory", "output/") if isinstance(output_cfg, dict) else "output/",
        rules_markdown=body.strip(),
        raw_config=frontmatter,
    )


# ============================================================
# PIPELINE CONTEXT — accumulates results as it flows through stages
# ============================================================

@dataclass
class PipelineContext:
    """Mutable context passed through all pipeline stages.

    Each stage reads what it needs, writes its results.
    The context IS the pipeline's state.
    """
    # Input
    input_data: Any = None
    input_text: str = ""

    # System reference
    system: Optional[SystemDefinition] = None

    # Engine reference (set by runner)
    engine: Any = None

    # Stage results (each stage adds its key)
    results: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    started_at: Optional[datetime] = None
    stage_timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def get(self, key: str, default=None):
        """Get a result from a previous stage."""
        return self.results.get(key, default)

    def set(self, key: str, value: Any):
        """Set a result for downstream stages."""
        self.results[key] = value


# ============================================================
# STAGE REGISTRY — available pipeline stages
# ============================================================

# Type signature: async (context, config) -> None
# Stages mutate context.results in-place.
StageFunc = Callable[[PipelineContext, Dict[str, Any]], Any]

_STAGE_REGISTRY: Dict[str, StageFunc] = {}


def stage(name: str):
    """Decorator to register a pipeline stage."""
    def decorator(func: StageFunc):
        _STAGE_REGISTRY[name] = func
        return func
    return decorator


# --- Core stages ---

@stage("extract_keywords")
async def extract_keywords(ctx: PipelineContext, config: Dict):
    """Extract keywords from input text using the configured engine."""
    text = ctx.input_text
    if not text:
        ctx.warnings.append("extract_keywords: no input text")
        return

    top_k = config.get("top_k", ctx.system.top_k)
    min_conf = config.get("min_confidence", ctx.system.min_confidence)

    # Use engine (SimpleSemanticEngine or SemanticIntelligenceEngine)
    if hasattr(ctx.engine, "extract_async"):
        keywords = await ctx.engine.extract_async(
            text, top_k=top_k, min_confidence=min_conf
        )
    else:
        keywords = ctx.engine.extract(text, top_k=top_k, min_confidence=min_conf)

    ctx.set("keywords", keywords)


@stage("cluster_keywords")
async def cluster_keywords(ctx: PipelineContext, config: Dict):
    """Cluster extracted keywords by semantic similarity."""
    keywords = ctx.get("keywords")
    if not keywords:
        return

    if hasattr(ctx.engine, "extract_async"):
        # Production engine has clustering built in
        result = await ctx.engine.extract_async(
            ctx.input_text,
            top_k=ctx.system.top_k,
            enable_clustering=True
        )
        ctx.set("clusters", result)
    else:
        ctx.set("clusters", keywords)


@stage("cross_document_analysis")
async def cross_document_analysis(ctx: PipelineContext, config: Dict):
    """Analyze multiple documents for common/distinctive keywords."""
    texts = ctx.input_data
    if not isinstance(texts, list) or len(texts) < 2:
        ctx.warnings.append("cross_document_analysis: need list of 2+ texts")
        return

    if hasattr(ctx.engine, "extract_multiple_advanced"):
        result = await ctx.engine.extract_multiple_advanced(texts)
        ctx.set("cross_analysis", result)


# --- SEO stages ---

@stage("find_bridges")
async def find_bridges(ctx: PipelineContext, config: Dict):
    """Find semantic bridge topics between publisher and target."""
    publisher_data = ctx.get("publisher_profile")
    target_data = ctx.get("target_fingerprint")

    if not publisher_data or not target_data:
        ctx.warnings.append("find_bridges: need publisher_profile and target_fingerprint")
        return

    if hasattr(ctx.engine, "find_bridge_topics"):
        bridges = ctx.engine.find_bridge_topics(publisher_data, target_data)
        ctx.set("bridges", bridges)


@stage("assess_risk")
async def assess_risk(ctx: PipelineContext, config: Dict):
    """Assess risk level based on semantic distance and target type."""
    bridges = ctx.get("bridges")
    if not bridges:
        return

    # Risk logic (from BACOWR pattern)
    best = bridges[0] if bridges else None
    if best:
        score = best.get("score", 0.5) if isinstance(best, dict) else 0.5
        if score < 0.3:
            risk = "HIGH"
        elif score < 0.5:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        ctx.set("risk_level", risk)
        ctx.set("bridge_score", score)


@stage("generate_constraints")
async def generate_constraints(ctx: PipelineContext, config: Dict):
    """Generate writer constraints from bridge analysis (BACOWR-style)."""
    bridges = ctx.get("bridges")
    keywords = ctx.get("keywords")

    constraints = {
        "required_entities": [],
        "forbidden_entities": [],
        "recommended_angle": None,
        "trust_link_topics": [],
    }

    if bridges:
        best = bridges[0] if bridges else {}
        if isinstance(best, dict):
            constraints["recommended_angle"] = best.get("content_angle")
            constraints["required_entities"] = best.get("entities", [])

    ctx.set("constraints", constraints)


# --- Medical stages ---

@stage("extract_medical_entities")
async def extract_medical_entities(ctx: PipelineContext, config: Dict):
    """Extract medical entities (symptoms, conditions, medications)."""
    keywords = ctx.get("keywords", [])

    # MedicalTransformer should be loaded — engine has .diagnose()
    if hasattr(ctx.engine, "diagnose"):
        ctx.set("medical_entities", keywords)
    else:
        ctx.warnings.append("extract_medical_entities: medical transformer not loaded")


@stage("differential_diagnosis")
async def differential_diagnosis(ctx: PipelineContext, config: Dict):
    """Run Bayesian differential diagnosis."""
    if not hasattr(ctx.engine, "diagnose"):
        ctx.warnings.append("differential_diagnosis: engine.diagnose() not available")
        return

    symptoms = config.get("symptoms", [])
    history = config.get("history", {})
    diagnosis = ctx.engine.diagnose(symptoms, history)
    ctx.set("differential_diagnosis", diagnosis)


@stage("drug_interactions")
async def drug_interactions(ctx: PipelineContext, config: Dict):
    """Check drug-drug interactions."""
    if not hasattr(ctx.engine, "check_drug_safety"):
        ctx.warnings.append("drug_interactions: engine.check_drug_safety() not available")
        return

    medications = config.get("medications", [])
    interactions = ctx.engine.check_drug_safety(medications)
    ctx.set("drug_interactions", interactions)


@stage("generate_soap_note")
async def generate_soap_note(ctx: PipelineContext, config: Dict):
    """Generate SOAP clinical note."""
    if not hasattr(ctx.engine, "generate_soap_note"):
        return

    entities = ctx.get("medical_entities", {})
    diagnosis = ctx.get("differential_diagnosis", {})
    note = ctx.engine.generate_soap_note(entities, diagnosis)
    ctx.set("soap_note", note)


# --- Legal stages ---

@stage("extract_legal_entities")
async def extract_legal_entities(ctx: PipelineContext, config: Dict):
    """Extract legal references (SFS, EU, case law)."""
    if not hasattr(ctx.engine, "find_applicable_law"):
        ctx.warnings.append("extract_legal_entities: legal transformer not loaded")
        return
    keywords = ctx.get("keywords", [])
    ctx.set("legal_entities", keywords)


@stage("legal_compliance_check")
async def legal_compliance_check(ctx: PipelineContext, config: Dict):
    """Check legal compliance against jurisdiction hierarchy."""
    if not hasattr(ctx.engine, "check_legal_compliance"):
        return
    result = ctx.engine.check_legal_compliance(ctx.input_text)
    ctx.set("compliance", result)


# --- Output stages ---

@stage("format_output")
async def format_output(ctx: PipelineContext, config: Dict):
    """Format results based on system output spec."""
    fmt = ctx.system.output_format

    if fmt == "json":
        ctx.set("output", json.dumps(ctx.results, indent=2, default=str))
    elif fmt == "markdown":
        lines = [f"# {ctx.system.name} — Results\n"]
        lines.append(f"Generated: {datetime.now().isoformat()}\n")
        for key, value in ctx.results.items():
            if key == "output":
                continue
            lines.append(f"## {key}\n")
            lines.append(f"```json\n{json.dumps(value, indent=2, default=str)}\n```\n")
        ctx.set("output", "\n".join(lines))


@stage("save_output")
async def save_output(ctx: PipelineContext, config: Dict):
    """Save formatted output to disk."""
    output = ctx.get("output")
    if not output:
        return

    out_dir = Path(ctx.system.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "json" if ctx.system.output_format == "json" else "md"
    path = out_dir / f"{ctx.system.name.lower().replace(' ', '-')}_{timestamp}.{ext}"

    path.write_text(output if isinstance(output, str) else json.dumps(output, default=str),
                    encoding="utf-8")
    ctx.set("output_path", str(path))
    print(f"  Saved: {path}")


# ============================================================
# SYSTEM RUNNER — the agnostic orchestrator
# ============================================================

class SystemRunner:
    """Agnostic pipeline runner driven by SYSTEM.md files.

    Same engine, different personality.

    >>> runner = SystemRunner("systems/seo-bridges.md")
    >>> result = await runner.run({"text": "..."})
    >>>
    >>> # Swap personality
    >>> runner.load_system("systems/medical-dx.md")
    >>> result = await runner.run({"text": clinical_note})
    """

    def __init__(self, system_path: str):
        self.system: Optional[SystemDefinition] = None
        self.engine = None
        self._systems_dir = Path(__file__).parent / "systems"
        self.load_system(system_path)

    def load_system(self, system_path: str):
        """Load (or swap) a SYSTEM.md file. Reconfigures engine."""
        path = Path(system_path)

        # Allow short names: "seo-bridges" → "systems/seo-bridges.md"
        if not path.exists() and not path.suffix:
            path = self._systems_dir / f"{system_path}.md"
        if not path.exists():
            path = self._systems_dir / system_path

        if not path.exists():
            raise FileNotFoundError(f"System file not found: {system_path}")

        self.system = parse_system_file(str(path))
        self.engine = self._configure_engine()

        print(f"Loaded system: {self.system.name} v{self.system.version}")
        if self.system.transformers:
            print(f"  Transformers: {', '.join(self.system.transformers)}")
        print(f"  Stages: {' → '.join(self.system.stages)}")

    def _configure_engine(self):
        """Create and configure SIE-X engine based on system definition."""

        mode = self.system.engine_mode.lower()

        # Try production engine first, fall back to simple
        if mode in ("advanced", "ultra"):
            try:
                from sie_x.core.engine import SemanticIntelligenceEngine, ModelMode
                mode_map = {
                    "fast": ModelMode.FAST,
                    "balanced": ModelMode.BALANCED,
                    "advanced": ModelMode.ADVANCED,
                    "ultra": ModelMode.ULTRA,
                }
                engine = SemanticIntelligenceEngine(
                    mode=mode_map.get(mode, ModelMode.BALANCED),
                    enable_gpu=mode in ("advanced", "ultra"),
                )
            except ImportError:
                from sie_x.core.simple_engine import SimpleSemanticEngine
                engine = SimpleSemanticEngine()
        else:
            try:
                from sie_x.core.simple_engine import SimpleSemanticEngine
                engine = SimpleSemanticEngine()
            except ImportError:
                engine = None

        # Load transformers
        if self.system.transformers and engine:
            try:
                from sie_x.transformers.loader import TransformerLoader
                loader = TransformerLoader(engine)
                if len(self.system.transformers) == 1:
                    loader.load_transformer(self.system.transformers[0])
                else:
                    loader.create_hybrid_system(self.system.transformers)
            except Exception as e:
                print(f"  Warning: could not load transformers: {e}")

        return engine

    async def run(
        self,
        input_data: Any = None,
        input_text: str = "",
        **overrides
    ) -> PipelineContext:
        """Run the full pipeline defined by the loaded system.

        Args:
            input_data: Structured input (CSV rows, list of texts, etc.)
            input_text: Raw text input for single-document analysis.
            **overrides: Per-run overrides for stage configs.

        Returns:
            PipelineContext with all stage results accumulated.
        """
        import time

        ctx = PipelineContext(
            input_data=input_data,
            input_text=input_text or (input_data if isinstance(input_data, str) else ""),
            system=self.system,
            engine=self.engine,
            started_at=datetime.now(),
        )

        print(f"\n{'=' * 60}")
        print(f"Running: {self.system.name}")
        print(f"{'=' * 60}")

        for i, stage_name in enumerate(self.system.stages, 1):
            if stage_name not in _STAGE_REGISTRY:
                ctx.warnings.append(f"Unknown stage: {stage_name}")
                print(f"  [{i}/{len(self.system.stages)}] {stage_name} — SKIPPED (unknown)")
                continue

            stage_func = _STAGE_REGISTRY[stage_name]
            stage_config = self.system.raw_config.get("stage_config", {}).get(stage_name, {})
            stage_config.update(overrides.get(stage_name, {}))

            t0 = time.time()
            try:
                print(f"  [{i}/{len(self.system.stages)}] {stage_name}...", end=" ")
                await stage_func(ctx, stage_config)
                elapsed = time.time() - t0
                ctx.stage_timings[stage_name] = elapsed
                print(f"OK ({elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                ctx.errors.append(f"{stage_name}: {e}")
                ctx.stage_timings[stage_name] = elapsed
                print(f"ERROR: {e}")

        print(f"\n{'=' * 60}")
        print(f"Done: {len(ctx.results)} results, {len(ctx.warnings)} warnings, {len(ctx.errors)} errors")
        print(f"{'=' * 60}")

        return ctx

    def get_rules(self) -> str:
        """Return the markdown body (domain rules) for LLM agents."""
        return self.system.rules_markdown

    def list_stages(self) -> List[str]:
        """List all available stages."""
        return list(_STAGE_REGISTRY.keys())


# ============================================================
# STANDALONE TOOLS — individually callable (same pattern as BACOWR)
# ============================================================

async def tool_extract(text: str, system: str = "seo-bridges") -> str:
    """Extract keywords using a specific system. Returns JSON."""
    runner = SystemRunner(system)
    ctx = await runner.run(input_text=text)
    return json.dumps(ctx.results, indent=2, default=str)


async def tool_list_systems() -> str:
    """List available system definitions."""
    systems_dir = Path(__file__).parent / "systems"
    if not systems_dir.exists():
        return json.dumps({"systems": []})

    systems = []
    for f in sorted(systems_dir.glob("*.md")):
        try:
            sysdef = parse_system_file(str(f))
            systems.append({
                "file": f.name,
                "name": sysdef.name,
                "version": sysdef.version,
                "description": sysdef.description,
                "transformers": sysdef.transformers,
                "stages": sysdef.stages,
            })
        except Exception:
            systems.append({"file": f.name, "error": "parse failed"})

    return json.dumps({"systems": systems}, indent=2)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SIE-X SystemRunner — run any system definition"
    )
    sub = parser.add_subparsers(dest="command")

    # Run a system
    run_p = sub.add_parser("run", help="Run a system pipeline")
    run_p.add_argument("system", help="System name or path (e.g. 'seo-bridges')")
    run_p.add_argument("--input", "-i", help="Input text or file path")
    run_p.add_argument("--text", "-t", help="Direct text input")

    # List systems
    sub.add_parser("list", help="List available system definitions")

    # List stages
    sub.add_parser("stages", help="List all registered pipeline stages")

    args = parser.parse_args()

    if args.command == "list":
        print(asyncio.run(tool_list_systems()))

    elif args.command == "stages":
        for name in sorted(_STAGE_REGISTRY.keys()):
            doc = _STAGE_REGISTRY[name].__doc__ or ""
            print(f"  {name:30s} {doc.strip().split(chr(10))[0]}")

    elif args.command == "run":
        text = args.text or ""
        if args.input:
            p = Path(args.input)
            if p.exists():
                text = p.read_text(encoding="utf-8")
            else:
                text = args.input

        runner = SystemRunner(args.system)
        asyncio.run(runner.run(input_text=text))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
