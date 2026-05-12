"""Validate whether the local-inventory RSI-NAS integration proves AGI.

The validation deliberately separates three claims:

1. The integration is grounded in local source-code evidence.
2. The generated modules are real executable/evaluated RSI-NAS components, not
   only labels around an unchanged system.
3. The system demonstrates AGI-level capability.

Passing (1) and (2) is not enough for (3).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from local_agi_architect import (
    LocalInventoryArchitect,
    extract_signal,
    run_local_agi_architect,
)
from rsi_nas import (
    ArchitectureGenome,
    LayerGene,
    ModuleRegistry,
    evaluate_architecture,
)


@dataclass
class GateResult:
    name: str
    passed: bool
    evidence: Dict[str, object] = field(default_factory=dict)
    reason: str = ""


@dataclass
class AGIClaimValidationReport:
    integration_grounded: bool
    not_wrapper_only: bool
    agi_claim_verified: bool
    final_verdict: str
    gates: List[GateResult]
    local_architect_summary: Dict[str, object]
    baseline_evaluation: Optional[Dict[str, object]]
    failure_reasons: List[str]

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "integration_grounded": self.integration_grounded,
            "not_wrapper_only": self.not_wrapper_only,
            "agi_claim_verified": self.agi_claim_verified,
            "final_verdict": self.final_verdict,
            "gates": [asdict(gate) for gate in self.gates],
            "local_architect_summary": self.local_architect_summary,
            "baseline_evaluation": self.baseline_evaluation,
            "failure_reasons": self.failure_reasons,
        }


def _best_candidate(report: Dict[str, object]) -> Optional[Dict[str, object]]:
    candidates = report.get("candidates", [])
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (row.get("evaluation") or {}).get("fitness", -1.0),
    )


def _causal_conditioning_probe() -> GateResult:
    """Check whether different source motifs produce different generated modules."""

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        neural = root / "neural_rsi_evolver.py"
        symbolic = root / "symbolic_program_synthesis.py"
        neural.write_text(
            """
import torch

class NeuralRSIEvolver:
    def evolve_population(self, tensor):
        fitness = tensor.sum()
        mutation = tensor + torch.randn_like(tensor)
        return fitness, mutation
""",
            encoding="utf-8",
        )
        symbolic.write_text(
            """
import ast

class SymbolicSynthesisPlanner:
    def synthesize_program(self):
        tree = ast.parse("x = 1")
        return compile(tree, "<generated>", "exec")
""",
            encoding="utf-8",
        )

        architect = LocalInventoryArchitect()
        neural_candidate = architect.synthesize_candidates(
            [extract_signal(neural)],
            count=1,
            d_model=8,
        )[0]
        symbolic_candidate = architect.synthesize_candidates(
            [extract_signal(symbolic)],
            count=1,
            d_model=8,
        )[0]

    different = (
        neural_candidate.generated_module_name != symbolic_candidate.generated_module_name
        and neural_candidate.motif_sequence != symbolic_candidate.motif_sequence
    )
    return GateResult(
        name="source_conditioning_changes_generated_architecture",
        passed=bool(different and neural_candidate.buildable and symbolic_candidate.buildable),
        evidence={
            "neural_module": neural_candidate.generated_module_name,
            "symbolic_module": symbolic_candidate.generated_module_name,
            "neural_motifs": list(neural_candidate.motif_sequence),
            "symbolic_motifs": list(symbolic_candidate.motif_sequence),
            "neural_buildable": neural_candidate.buildable,
            "symbolic_buildable": symbolic_candidate.buildable,
        },
        reason=(
            "Different source-code motifs must change the generated architecture; "
            "otherwise the local inventory layer is only decorative."
        ),
    )


def _evaluate_baseline(args: argparse.Namespace) -> Dict[str, object]:
    registry = ModuleRegistry()
    genome = ArchitectureGenome(
        layers=[
            LayerGene("gated_shift_mixer", repeat=1),
            LayerGene("gated_ffn", repeat=1),
            LayerGene("squeeze_excite", repeat=1),
        ],
        d_model=args.d_model,
        vocab_size=256,
        max_len=128,
    )
    result = evaluate_architecture(
        genome,
        registry,
        train_steps=args.train_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_params=args.max_params,
        device=torch.device(args.device),
    )
    return asdict(result)


def validate_agi_claim(args: argparse.Namespace) -> AGIClaimValidationReport:
    local_report = run_local_agi_architect(args)
    candidates = local_report.get("candidates", [])
    best = _best_candidate(local_report)

    gates: List[GateResult] = []
    gates.append(
        GateResult(
            name="local_inventory_was_scanned",
            passed=int(local_report.get("signals_scanned", 0)) >= args.min_signals,
            evidence={
                "signals_scanned": local_report.get("signals_scanned", 0),
                "syntax_valid_signals": local_report.get("syntax_valid_signals", 0),
                "minimum_required": args.min_signals,
            },
        )
    )
    gates.append(
        GateResult(
            name="generated_modules_registered",
            passed=len(local_report.get("registered_generated_modules", [])) >= args.candidates,
            evidence={
                "registered_generated_modules": local_report.get(
                    "registered_generated_modules",
                    [],
                ),
                "minimum_required": args.candidates,
            },
        )
    )
    gates.append(
        GateResult(
            name="all_candidates_buildable",
            passed=bool(candidates) and all(row.get("buildable") for row in candidates),
            evidence={
                row.get("name", f"candidate_{i}"): row.get("buildable")
                for i, row in enumerate(candidates)
            },
        )
    )
    gates.append(
        GateResult(
            name="real_rsi_nas_evaluation_ran",
            passed=bool(candidates)
            and all(
                (row.get("evaluation") or {}).get("param_count", 0) > 0
                and 0.0 < (row.get("evaluation") or {}).get("fitness", 0.0) < 1.0
                and (row.get("evaluation") or {}).get("bpc", 99.0) < 99.0
                for row in candidates
            ),
            evidence={
                row.get("name", f"candidate_{i}"): row.get("evaluation")
                for i, row in enumerate(candidates)
            },
            reason="A wrapper-only system would not produce trainable networks with measured BPC.",
        )
    )
    gates.append(_causal_conditioning_probe())

    baseline = None
    if not args.skip_baseline:
        baseline = _evaluate_baseline(args)
        best_eval = (best or {}).get("evaluation") or {}
        gates.append(
            GateResult(
                name="generated_candidate_has_finite_baseline_comparison",
                passed=bool(best_eval) and baseline["fitness"] > 0.0,
                evidence={
                    "best_generated_fitness": best_eval.get("fitness"),
                    "best_generated_bpc": best_eval.get("bpc"),
                    "baseline_fitness": baseline.get("fitness"),
                    "baseline_bpc": baseline.get("bpc"),
                    "generated_beats_baseline": (
                        best_eval.get("fitness", -1.0) > baseline.get("fitness", 1.0)
                    ),
                },
                reason=(
                    "This gate only proves comparable real evaluation exists. "
                    "Beating a tiny baseline is not an AGI criterion."
                ),
            )
        )

    agi_gates = [
        GateResult(
            name="agi_cross_domain_generalization",
            passed=False,
            evidence={
                "implemented_domains": ["character_language_model_architecture_search"],
                "missing_domains": [
                    "tool-use planning",
                    "causal intervention",
                    "long-horizon memory tasks",
                    "embodied or desktop environment control",
                    "unseen benchmark transfer",
                ],
            },
            reason="The current system evaluates neural architectures on a narrow character LM task.",
        ),
        GateResult(
            name="agi_autonomous_goal_formation",
            passed=False,
            evidence={
                "observed": "candidate architecture generation and bounded policy mutation",
                "missing": "independent goal discovery, planning, execution, and verification across tasks",
            },
            reason="The system optimizes a predefined fitness function; it does not demonstrate autonomous general problem solving.",
        ),
        GateResult(
            name="agi_recursive_self_improvement_generalizes",
            passed=False,
            evidence={
                "observed": "bounded RSI-NAS/evaluation loop",
                "missing": "validated improvement across independent domains and holdout task families",
            },
            reason="Bounded improvement inside RSI-NAS is not evidence of open-ended AGI-level self-improvement.",
        ),
    ]
    gates.extend(agi_gates)

    integration_grounded = all(
        gate.passed
        for gate in gates
        if gate.name
        in {
            "local_inventory_was_scanned",
            "generated_modules_registered",
            "all_candidates_buildable",
            "real_rsi_nas_evaluation_ran",
            "source_conditioning_changes_generated_architecture",
        }
    )
    not_wrapper_only = integration_grounded
    agi_claim_verified = all(gate.passed for gate in agi_gates)

    failure_reasons = [gate.reason for gate in gates if not gate.passed and gate.reason]
    if agi_claim_verified:
        verdict = "AGI_CLAIM_VERIFIED"
    elif not_wrapper_only:
        verdict = "REAL_INTEGRATION_BUT_AGI_PROOF_FAILED"
    else:
        verdict = "WRAPPER_OR_BROKEN_INTEGRATION"

    summary = {
        "inventory": local_report.get("inventory"),
        "signals_scanned": local_report.get("signals_scanned"),
        "syntax_valid_signals": local_report.get("syntax_valid_signals"),
        "registered_generated_modules": local_report.get("registered_generated_modules"),
        "best_candidate": best,
    }
    return AGIClaimValidationReport(
        integration_grounded=integration_grounded,
        not_wrapper_only=not_wrapper_only,
        agi_claim_verified=agi_claim_verified,
        final_verdict=verdict,
        gates=gates,
        local_architect_summary=summary,
        baseline_evaluation=baseline,
        failure_reasons=failure_reasons,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--max-files", type=int, default=160)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--min-signals", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-params", type=int, default=600_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--json-output", default="agi_claim_validation_report.json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = validate_agi_claim(args)
    output = Path(args.json_output)
    output.write_text(
        json.dumps(
            report.to_json_dict(),
            indent=2,
            sort_keys=True,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        ),
        encoding="utf-8",
    )

    print(f"final_verdict={report.final_verdict}")
    print(f"integration_grounded={report.integration_grounded}")
    print(f"not_wrapper_only={report.not_wrapper_only}")
    print(f"agi_claim_verified={report.agi_claim_verified}")
    for gate in report.gates:
        state = "PASS" if gate.passed else "FAIL"
        print(f"{state} {gate.name}")
    print(f"report={output}")
    return 0 if report.agi_claim_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
