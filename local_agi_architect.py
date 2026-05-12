"""Local Python inventory driven architecture synthesis for RSI-NAS.

This module connects a scanned local Python inventory to the repository's real
attention-free NAS evaluator. It does not import or execute arbitrary local
Python files. It parses them as AST evidence, extracts algorithmic motifs, and
turns those motifs into generated RSI-NAS module specs and candidate genomes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from rsi_nas import (
    ArchitectureGenome,
    CoarseNCA,
    EIEConfig,
    FractalGNNBlock,
    GatedFFN,
    GatedShiftMixer,
    LayerGene,
    ModuleRegistry,
    ModuleSpec,
    NCAStep,
    SqueezeExcite,
    build_network,
    build_rsi_nas,
    evaluate_architecture,
)


INSTALL_OR_TOOL_PATH = re.compile(
    r"\\site-packages\\|\\dist-packages\\|\\__pycache__\\|\\node_modules\\|"
    r"\\.git\\|\\(?:venv|\.venv|env|\.env|virtualenv)\\|"
    r"^C:\\Users\\[^\\]+\\AppData\\|^C:\\Users\\[^\\]+\\.gemini\\|"
    r"^C:\\Users\\[^\\]+\\.antigravity\\|^C:\\Users\\[^\\]+\\.codex\\|"
    r"^C:\\Python\d+\\|^C:\\Program Files",
    re.IGNORECASE,
)


KEYWORDS = {
    "rsi": re.compile(r"\bRSI\b|recursive[_ -]?self[_ -]?improvement|self[_ -]?improv", re.I),
    "agi": re.compile(r"\bAGI\b|cognitive|reasoning|world[_ -]?model|planner", re.I),
    "evolution": re.compile(r"evolv|mutation|fitness|genetic|population|selection|pareto", re.I),
    "code_mod": re.compile(r"ast\.|exec\(|eval\(|compile\(|subprocess|patch|modify|write_text", re.I),
    "evaluation": re.compile(r"pytest|unittest|assert |benchmark|verify_|holdout|score", re.I),
    "neural": re.compile(r"torch|nn\.|neural|tensor|gradient|optimizer|backprop|maml|ssm", re.I),
    "agent": re.compile(r"agent|tool|memory|policy|loop|orchestrator", re.I),
    "safety": re.compile(r"safety|contract|rollback|quarantine|sandbox|governance", re.I),
    "symbolic": re.compile(r"grammar|symbolic|ast|program|synthesis|interpreter", re.I),
}


PRIMITIVE_BUILDERS = {
    "nca_step": lambda d: NCAStep(d, k=5, dilations=(1, 4), exp=2, drop=0.0),
    "gated_shift_mixer": lambda d: GatedShiftMixer(d, shifts=(-8, -2, -1, 1, 2, 8)),
    "fractal_gnn": lambda d: FractalGNNBlock(d, chunk_size=8, gnn_depth=1, dropout=0.0),
    "coarse_nca": lambda d: CoarseNCA(d, stride=4, n_steps=1, k=5, dilations=(1, 4), drop=0.0),
    "squeeze_excite": lambda d: SqueezeExcite(d, r=4),
    "gated_ffn": lambda d: GatedFFN(d, exp=2, drop=0.0),
}


@dataclass
class LocalPythonSignal:
    """AST and keyword evidence extracted from one local Python file."""

    path: str
    digest: str
    bytes_read: int
    syntax_valid: bool
    score: float
    keyword_counts: Dict[str, int]
    class_names: List[str] = field(default_factory=list)
    function_names: List[str] = field(default_factory=list)
    import_roots: List[str] = field(default_factory=list)
    syntax_error: Optional[str] = None

    @property
    def motif_vector(self) -> Dict[str, int]:
        return {key: int(value) for key, value in self.keyword_counts.items() if value}


@dataclass
class EmergentArchitectureCandidate:
    """A generated architecture candidate traceable to local source evidence."""

    name: str
    generated_module_name: str
    genome: ArchitectureGenome
    motif_sequence: Tuple[str, ...]
    source_paths: List[str]
    source_digests: List[str]
    rationale: List[str]
    buildable: bool = False
    evaluation: Optional[Dict[str, object]] = None

    def to_json_dict(self) -> Dict[str, object]:
        row = asdict(self)
        row["genome"] = self.genome.to_dict()
        row["motif_sequence"] = list(self.motif_sequence)
        return row


class LocalMotifFusionBlock(nn.Module):
    """Residual fusion of primitive attention-free modules selected from motifs."""

    def __init__(
        self,
        d_model: int,
        motif_sequence: Sequence[str],
        gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        sequence = [name for name in motif_sequence if name in PRIMITIVE_BUILDERS]
        if not sequence:
            sequence = ["gated_ffn"]
        self.motif_sequence = tuple(sequence)
        self.blocks = nn.ModuleList([PRIMITIVE_BUILDERS[name](d_model) for name in sequence])
        self.gate = nn.Linear(2 * d_model, d_model, bias=True)
        nn.init.constant_(self.gate.bias, float(gate_bias))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = x
        for block in self.blocks:
            h = block(h)
        gate = torch.sigmoid(self.gate(torch.cat([residual, h], dim=-1)))
        return self.norm(residual + gate * (h - residual))


def discover_default_inventory(start: Optional[Path] = None) -> Optional[Path]:
    """Find the previous local Python inventory created by the PC scan."""

    roots: List[Path] = []
    if start is not None:
        roots.extend([start, *start.parents])
    roots.append(Path.cwd())
    roots.extend(Path.cwd().parents)
    roots.append(Path(r"C:\Users\starg\Documents\Codex\2026-05-11\pc"))

    names = ("python_file_inventory.txt", "python_candidate_analysis_clean.json")
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def inventory_paths(inventory: Path) -> List[Path]:
    """Load file paths from either the full text inventory or analysis JSON."""

    if inventory.suffix.lower() == ".json":
        data = json.loads(inventory.read_text(encoding="utf-8"))
        paths = [
            Path(row["path"])
            for row in data.get("interesting_files_sample", [])
            if row.get("path")
        ]
        sibling_full = inventory.with_name("python_file_inventory.txt")
        if sibling_full.exists():
            paths.extend(inventory_paths(sibling_full))
        return list(dict.fromkeys(paths))

    lines = inventory.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    return [Path(line.strip().lstrip("\ufeff")) for line in lines if line.strip()]


def candidate_source_paths(paths: Iterable[Path], max_files: int) -> List[Path]:
    """Filter install/tool internals and keep likely project files first."""

    scored: List[Tuple[int, str, Path]] = []
    for path in paths:
        path_text = str(path)
        if not path_text.lower().endswith(".py"):
            continue
        if INSTALL_OR_TOOL_PATH.search(path_text):
            continue
        lower = path_text.lower()
        priority = 0
        for term in ("rsi", "agi", "omega", "cognitive", "evol", "meta", "neural", "synthesis"):
            if term in lower:
                priority -= 5
        if "\\downloads\\" in lower or "\\agi_integration_lab\\" in lower:
            priority -= 2
        scored.append((priority, path_text, path))
    scored.sort()
    return [path for _, _, path in scored[:max_files]]


def extract_signal(path: Path, max_bytes: int = 300_000) -> LocalPythonSignal:
    """Extract AST-safe evidence from one Python file."""

    raw = path.read_bytes()[:max_bytes]
    text = raw.decode("utf-8-sig", errors="ignore")
    digest = hashlib.sha256(raw).hexdigest()
    keyword_counts = {
        key: len(regex.findall(text)) + len(regex.findall(str(path)))
        for key, regex in KEYWORDS.items()
    }

    class_names: List[str] = []
    function_names: List[str] = []
    import_roots: List[str] = []
    syntax_valid = True
    syntax_error = None
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        syntax_valid = False
        syntax_error = f"line {exc.lineno}: {exc.msg}"
    else:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_names.append(node.name)
            elif isinstance(node, ast.Import):
                import_roots.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                import_roots.append(node.module.split(".", 1)[0])

    ast_density = min(20, len(class_names) + len(function_names))
    keyword_score = sum(keyword_counts.values())
    neural_bonus = 8 if keyword_counts["neural"] else 0
    evidence_bonus = 6 if keyword_counts["evaluation"] else 0
    safety_bonus = 4 if keyword_counts["safety"] else 0
    syntax_penalty = 25 if not syntax_valid else 0
    score = keyword_score + ast_density + neural_bonus + evidence_bonus + safety_bonus - syntax_penalty

    return LocalPythonSignal(
        path=str(path),
        digest=digest[:16],
        bytes_read=len(raw),
        syntax_valid=syntax_valid,
        score=float(score),
        keyword_counts=keyword_counts,
        class_names=class_names[:24],
        function_names=function_names[:32],
        import_roots=sorted(set(import_roots))[:24],
        syntax_error=syntax_error,
    )


def primitive_sequence_from_signal(signal: LocalPythonSignal, offset: int = 0) -> Tuple[str, ...]:
    """Map source-code motifs to attention-free module primitives."""

    counts = signal.keyword_counts
    sequence: List[str] = []
    if counts.get("neural", 0) or counts.get("rsi", 0):
        sequence.extend(["gated_shift_mixer", "nca_step"])
    if counts.get("evolution", 0):
        sequence.extend(["fractal_gnn", "gated_ffn"])
    if counts.get("symbolic", 0) or counts.get("code_mod", 0):
        sequence.extend(["coarse_nca", "gated_shift_mixer"])
    if counts.get("evaluation", 0) or counts.get("safety", 0):
        sequence.append("squeeze_excite")
    if counts.get("agent", 0) or counts.get("agi", 0):
        sequence.extend(["gated_ffn", "coarse_nca"])
    if not sequence:
        sequence = ["gated_shift_mixer", "gated_ffn"]

    deduped: List[str] = []
    for name in sequence:
        if not deduped or deduped[-1] != name:
            deduped.append(name)
    rotation = offset % len(deduped)
    rotated = deduped[rotation:] + deduped[:rotation]
    return tuple(rotated[:4])


class LocalInventoryArchitect:
    """Mine local Python files and synthesize RSI-NAS architecture candidates."""

    def __init__(self, registry: Optional[ModuleRegistry] = None) -> None:
        self.registry = registry or ModuleRegistry()

    def scan_inventory(self, inventory: Path, max_files: int = 512) -> List[LocalPythonSignal]:
        paths = candidate_source_paths(inventory_paths(inventory), max_files=max_files)
        signals: List[LocalPythonSignal] = []
        for path in paths:
            if not path.exists() or not path.is_file():
                continue
            try:
                signals.append(extract_signal(path))
            except OSError:
                continue
        signals.sort(key=lambda row: (row.syntax_valid, row.score, row.bytes_read), reverse=True)
        return signals

    def register_motif_module(
        self,
        signal: LocalPythonSignal,
        sequence: Sequence[str],
        candidate_index: int,
    ) -> str:
        digest = hashlib.blake2b(
            json.dumps(
                {
                    "digest": signal.digest,
                    "sequence": list(sequence),
                    "index": candidate_index,
                },
                sort_keys=True,
            ).encode("utf-8"),
            digest_size=5,
        ).hexdigest()
        module_name = f"local_motif_fusion_{digest}"
        gate_bias = math.tanh(signal.score / 100.0) - 0.5

        def builder(
            d_model: int,
            motif_sequence: Sequence[str] = tuple(sequence),
            gate_bias: float = gate_bias,
            **_: object,
        ) -> nn.Module:
            return LocalMotifFusionBlock(
                d_model=d_model,
                motif_sequence=motif_sequence,
                gate_bias=gate_bias,
            )

        spec = ModuleSpec(
            name=module_name,
            builder=builder,
            default_kwargs={
                "motif_sequence": tuple(sequence),
                "gate_bias": round(gate_bias, 6),
            },
            param_cost=max(2.0, 1.5 * len(sequence) + 1.0),
            description=(
                "Generated from local Python inventory AST motifs; "
                f"source_digest={signal.digest}"
            ),
            is_generated=True,
        )
        self.registry.register(
            spec,
            birth_generation=0,
            source_action=f"local_inventory:{Path(signal.path).name}",
            parent_modules=tuple(sequence),
        )
        return module_name

    def synthesize_candidates(
        self,
        signals: Sequence[LocalPythonSignal],
        count: int = 6,
        d_model: int = 32,
    ) -> List[EmergentArchitectureCandidate]:
        usable = [signal for signal in signals if signal.syntax_valid and signal.score > 0]
        usable.sort(key=lambda row: (row.score, row.bytes_read), reverse=True)
        selected = usable[: max(1, count)]
        candidates: List[EmergentArchitectureCandidate] = []
        for index, signal in enumerate(selected):
            sequence = primitive_sequence_from_signal(signal, offset=index)
            module_name = self.register_motif_module(signal, sequence, index)
            layers = [
                LayerGene(module_name, repeat=1),
                LayerGene(sequence[-1], repeat=1),
            ]
            if "squeeze_excite" not in sequence:
                layers.append(LayerGene("squeeze_excite", repeat=1))
            genome = ArchitectureGenome(
                layers=layers,
                d_model=d_model,
                vocab_size=256,
                max_len=128,
            )
            candidate = EmergentArchitectureCandidate(
                name=f"local_architect_candidate_{index + 1}",
                generated_module_name=module_name,
                genome=genome,
                motif_sequence=sequence,
                source_paths=[signal.path],
                source_digests=[signal.digest],
                rationale=[
                    f"Local AST signal score {signal.score:.1f}.",
                    "Motifs converted into attention-free residual fusion.",
                    "Candidate remains subject to real RSI-NAS build/evaluation.",
                ],
            )
            candidate.buildable = build_network(genome, self.registry) is not None
            candidates.append(candidate)
        return candidates

    def evaluate_candidates(
        self,
        candidates: Sequence[EmergentArchitectureCandidate],
        train_steps: int = 2,
        seq_len: int = 32,
        batch_size: int = 2,
        max_params: int = 800_000,
        device: Optional[torch.device] = None,
    ) -> List[EmergentArchitectureCandidate]:
        for candidate in candidates:
            result = evaluate_architecture(
                candidate.genome,
                self.registry,
                train_steps=train_steps,
                seq_len=seq_len,
                batch_size=batch_size,
                max_params=max_params,
                device=device,
            )
            candidate.evaluation = asdict(result)
        return list(candidates)


def run_local_agi_architect(args: argparse.Namespace) -> Dict[str, object]:
    inventory = Path(args.inventory) if args.inventory else discover_default_inventory(Path.cwd())
    if inventory is None:
        raise FileNotFoundError("No local Python inventory was found. Pass --inventory.")

    device = torch.device(args.device)
    engine = build_rsi_nas(
        d_model=args.d_model,
        train_steps=args.train_steps,
        expansion_interval=3,
        pruning_interval=6,
        enable_eie=True,
        enable_meta_operator_evolution=True,
        eie_config=EIEConfig(
            grace_multiplier=2.0,
            probe_rate=1.0,
            clean_probe_first_eval=True,
        ),
        generated_min_evaluations=1,
        device=device,
    )
    architect = LocalInventoryArchitect(engine.registry)
    signals = architect.scan_inventory(inventory, max_files=args.max_files)
    candidates = architect.synthesize_candidates(
        signals,
        count=args.candidates,
        d_model=args.d_model,
    )
    if not args.no_evaluate:
        architect.evaluate_candidates(
            candidates,
            train_steps=args.train_steps,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_params=args.max_params,
            device=device,
        )

    return {
        "boundary": (
            "This is an AGI-architecture integration attempt: local source motifs "
            "are converted into generated RSI-NAS modules and evaluated. It is "
            "not a claim of achieved AGI."
        ),
        "inventory": str(inventory),
        "signals_scanned": len(signals),
        "syntax_valid_signals": sum(1 for signal in signals if signal.syntax_valid),
        "registered_generated_modules": engine.registry.generated_names(),
        "candidates": [candidate.to_json_dict() for candidate in candidates],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=None)
    parser.add_argument("--max-files", type=int, default=512)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-params", type=int, default=800_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_local_agi_architect(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report["boundary"])
        print(f"Inventory: {report['inventory']}")
        print(f"Signals scanned: {report['signals_scanned']}")
        print(f"Generated modules: {', '.join(report['registered_generated_modules'])}")
        for row in report["candidates"]:
            evaluation = row.get("evaluation") or {}
            print(
                f"- {row['name']} buildable={row['buildable']} "
                f"motifs={row['motif_sequence']} "
                f"fitness={evaluation.get('fitness')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
