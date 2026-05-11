"""
Residue-conditioned AFIRSI-OMEGA outer loop for RSI-NAS.

This script replaces handcoded outer-loop policy mutation with a deterministic
OMEGA-style adapter:

real FailureResidue evidence -> structured AFIRSI residue schema -> symbolic
missing-instrument interpretation -> generated instrument patch -> executable
EIEConfig policy -> paired-seed and holdout-seed validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import torch

from omega_adapter import (
    OMEGAInstrumentGenerator,
    OMEGAPolicyCandidate,
    StructuredAFIRSIResidue,
    apply_instrument_patch,
    export_residues,
)
from omega_adapter.policy_import import DEFAULT_SCORING_COEFFICIENTS, score_summary
from rsi_nas import EIEConfig, build_rsi_nas
from validate_eie import parse_seeds


BOUNDARY_STATEMENT = (
    "This demonstrates residue-conditioned recursive improvement of the "
    "EIE/AFIRSI instrument policy across validation cycles. It is not "
    "mathematical proof of unbounded open-ended RSI, AGI, or autonomous "
    "self-improvement in the real world."
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_policy_case(
    args: argparse.Namespace,
    seed: int,
    candidate: OMEGAPolicyCandidate | None,
    enable_eie: bool,
    condition_name: str,
) -> Tuple[Dict, List[Dict]]:
    _seed_everything(seed)
    generated_min_evaluations = (
        candidate.generated_min_evaluations
        if candidate is not None
        else args.generated_min_evaluations
    )
    eie_config = candidate.config if candidate is not None else None

    engine = build_rsi_nas(
        d_model=args.d_model,
        train_steps=args.train_steps,
        expansion_interval=args.expansion_interval,
        pruning_interval=args.pruning_interval,
        generated_min_evaluations=generated_min_evaluations,
        enable_eie=enable_eie,
        eie_config=eie_config,
    )
    history = engine.run(
        generations=args.generations,
        population_size=args.population_size,
    )
    final = history[-1]
    generated_records = engine.registry.all_generated_records()
    residues = [
        residue.to_json_dict()
        for residue in export_residues(engine, seed=seed, condition=condition_name)
    ]

    row = {
        "seed": seed,
        "condition": condition_name,
        "best_bpc": final["archive_best_bpc"],
        "generated_modules": final["generated_modules"],
        "eie_residues": final["eie_residues"],
        "instrument_mutations": final["instrument_mutations"],
        "instrumented_candidates": final["instrumented_candidates"],
        "protected_pruning_attempts": final["protected_pruning_attempts"],
        "meta_operator_policy_updates": final["meta_operator_policy_updates"],
        "meta_operator_weights": final["meta_operator_weights"],
        "generated_evaluations": sum(r.evaluations for r in generated_records),
        "generated_archive_insertions": sum(
            r.archive_insertions for r in generated_records
        ),
        "generated_records": [asdict(r) for r in generated_records],
    }
    return row, residues


def evaluate_policy(
    args: argparse.Namespace,
    seeds: List[int],
    baseline_rows: List[Dict],
    candidate: OMEGAPolicyCandidate,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    rows = []
    residues = []
    for seed in seeds:
        row, exported = run_policy_case(
            args,
            seed,
            candidate=candidate,
            enable_eie=True,
            condition_name=candidate.name,
        )
        rows.append(row)
        residues.extend(exported)
    summary = summarize_policy(
        name=candidate.name,
        baseline_rows=baseline_rows,
        candidate_rows=rows,
        scoring_coefficients=candidate.scoring_coefficients,
    )
    summary["candidate"] = candidate.to_json_dict()
    return summary, rows, residues


def run_baseline(args: argparse.Namespace, seeds: List[int]) -> List[Dict]:
    rows = []
    for seed in seeds:
        row, _ = run_policy_case(
            args,
            seed,
            candidate=None,
            enable_eie=False,
            condition_name="SELF-MODIFY",
        )
        rows.append(row)
    return rows


def summarize_policy(
    name: str,
    baseline_rows: List[Dict],
    candidate_rows: List[Dict],
    scoring_coefficients: Dict[str, float] | None = None,
) -> Dict:
    baseline_bpc = [row["best_bpc"] for row in baseline_rows]
    candidate_bpc = [row["best_bpc"] for row in candidate_rows]
    mean_bpc_delta = round(mean(baseline_bpc) - mean(candidate_bpc), 4)
    wins = sum(
        1 for base, cand in zip(baseline_bpc, candidate_bpc)
        if cand < base
    )
    total_archive_insertions = sum(
        row["generated_archive_insertions"] for row in candidate_rows
    )
    total_evaluations = sum(row["generated_evaluations"] for row in candidate_rows)
    total_policy_updates = sum(
        row["meta_operator_policy_updates"] for row in candidate_rows
    )
    total_residues = sum(row["eie_residues"] for row in candidate_rows)
    total_mutations = sum(row["instrument_mutations"] for row in candidate_rows)
    total_instrumented = sum(row["instrumented_candidates"] for row in candidate_rows)

    mechanism_valid = bool(
        total_residues > 0
        and total_mutations > 0
        and total_instrumented > 0
        and total_evaluations > 0
        and total_archive_insertions > 0
        and total_policy_updates > 0
        and mean_bpc_delta > 0.0
    )
    summary = {
        "name": name,
        "mean_best_bpc": round(mean(candidate_bpc), 4),
        "baseline_mean_best_bpc": round(mean(baseline_bpc), 4),
        "mean_bpc_delta": mean_bpc_delta,
        "eie_wins": wins,
        "paired_seeds": len(candidate_rows),
        "total_generated_archive_insertions": total_archive_insertions,
        "total_generated_evaluations": total_evaluations,
        "total_meta_operator_policy_updates": total_policy_updates,
        "total_eie_residues": total_residues,
        "total_instrument_mutations": total_mutations,
        "total_instrumented_candidates": total_instrumented,
        "mechanism_valid": mechanism_valid,
    }
    coeffs = dict(DEFAULT_SCORING_COEFFICIENTS)
    if scoring_coefficients:
        coeffs.update(scoring_coefficients)
    summary["score"] = score_summary(summary, coeffs)
    summary["scoring_coefficients"] = coeffs
    return summary


def passes_acceptance_gate(
    paired_summary: Dict,
    holdout_summary: Dict,
    champion_summary: Dict,
) -> Tuple[bool, List[str]]:
    failures = []
    required_positive = {
        "generated evaluations": paired_summary["total_generated_evaluations"],
        "archive insertions": paired_summary["total_generated_archive_insertions"],
        "meta-policy updates": paired_summary["total_meta_operator_policy_updates"],
        "real residues": paired_summary["total_eie_residues"],
    }
    for label, value in required_positive.items():
        if value <= 0:
            failures.append(f"paired missing {label}")
    if not paired_summary["mechanism_valid"]:
        failures.append("paired mechanism invalid")
    if paired_summary["mean_bpc_delta"] <= 0.0:
        failures.append("paired BPC does not beat SELF-MODIFY baseline")
    if paired_summary["score"] <= champion_summary["score"]:
        failures.append("candidate score does not beat previous champion")
    if not holdout_summary["mechanism_valid"]:
        failures.append("holdout mechanism invalid")
    if holdout_summary["mean_bpc_delta"] <= 0.0:
        failures.append("holdout BPC does not beat SELF-MODIFY baseline")
    if holdout_summary["total_generated_evaluations"] <= 0:
        failures.append("holdout missing generated evaluations")
    if holdout_summary["total_generated_archive_insertions"] <= 0:
        failures.append("holdout missing archive insertions")
    if holdout_summary["total_meta_operator_policy_updates"] <= 0:
        failures.append("holdout missing meta-policy updates")
    return not failures, failures


def final_validation_valid(
    accepted_count: int,
    required_accepted_improvements: int,
    bootstrap_residue_count: int,
    champion_summary: Dict,
) -> bool:
    return bool(
        accepted_count >= required_accepted_improvements
        and bootstrap_residue_count > 0
        and champion_summary["mechanism_valid"]
        and champion_summary["mean_bpc_delta"] > 0.0
        and champion_summary["total_generated_evaluations"] > 0
        and champion_summary["total_generated_archive_insertions"] > 0
        and champion_summary["total_meta_operator_policy_updates"] > 0
    )


def build_provenance_chain(
    candidate: OMEGAPolicyCandidate,
    paired_summary: Dict,
    holdout_summary: Dict | None,
    accepted: bool,
    current_champion_name: str,
) -> Dict:
    return {
        "residue_ids": candidate.patch.source_residue_ids,
        "generated_patch_id": candidate.patch.patch_id,
        "paired_evaluation": paired_summary["name"],
        "holdout_evaluation": holdout_summary["name"] if holdout_summary else None,
        "decision": "eligible" if accepted else "rejected",
        "next_parent_policy": candidate.name if accepted else current_champion_name,
    }


def _make_seed_policy() -> OMEGAPolicyCandidate:
    from omega_adapter.schemas import InstrumentPatch

    seed_patch = InstrumentPatch(
        patch_id="seed-policy",
        candidate_name="seed_policy",
        source_residue_ids=[],
        parent_policy_name="bootstrap",
        target_updates={},
        candidate_scoring_coefficients={},
        evaluator_terms=["bootstrap_default_evaluator"],
        archive_insertion_priority="baseline",
        generated_module_scaffold_strategy="baseline",
        constraints_interpreted=["bootstrap weak EIE policy"],
        rationale=["Initial weak policy used only as a parent for residue export."],
        provenance={"source": "bootstrap"},
    )
    return OMEGAPolicyCandidate(
        name="seed_policy",
        config=EIEConfig(
            grace_multiplier=1.0,
            probe_rate=0.5,
            clean_probe_first_eval=False,
            meta_eval_gain=0.10,
            meta_archive_gain=0.50,
            meta_fitness_gain=1.00,
            meta_exploration_floor=0.25,
        ),
        generated_min_evaluations=1,
        patch=seed_patch,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--holdout-seeds", default="23,29")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--expansion-interval", type=int, default=1)
    parser.add_argument("--pruning-interval", type=int, default=1)
    parser.add_argument("--generated-min-evaluations", type=int, default=1)
    parser.add_argument("--min-accepted-improvements", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--report-path", default="omega_rsi_report.json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    paired_seeds = parse_seeds(args.seeds)
    holdout_seeds = parse_seeds(args.holdout_seeds)
    baseline_rows = run_baseline(args, paired_seeds)
    holdout_baseline_rows = run_baseline(args, holdout_seeds)

    generator = OMEGAInstrumentGenerator()
    champion = _make_seed_policy()
    champion_summary, champion_rows, champion_residues = evaluate_policy(
        args, paired_seeds, baseline_rows, champion
    )

    accepted = []
    cycles = [{
        "cycle": 0,
        "parent_policy": "bootstrap",
        "champion": champion_summary,
        "residues_used": champion_residues,
        "accepted": False,
        "decision": "bootstrap_residue_export",
    }]

    for cycle in range(1, args.cycles + 1):
        patches = generator.generate(
            residues=[
                StructuredAFIRSIResidue(**residue)
                for residue in champion_residues
            ],
            parent_config=champion.config,
            parent_generated_min_evaluations=champion.generated_min_evaluations,
            cycle=cycle,
            parent_policy_name=champion.name,
        )
        patches = patches[:max(0, args.max_candidates)]
        candidates = [
            apply_instrument_patch(
                parent_name=champion.name,
                parent_config=champion.config,
                parent_generated_min_evaluations=champion.generated_min_evaluations,
                patch=patch,
            )
            for patch in patches
        ]

        candidate_reports = []
        acceptable = []
        for candidate in candidates:
            paired_summary, paired_rows, paired_residues = evaluate_policy(
                args, paired_seeds, baseline_rows, candidate
            )
            holdout_summary = None
            holdout_rows = []
            holdout_residues = []
            if (
                paired_summary["mechanism_valid"]
                and paired_summary["mean_bpc_delta"] > 0.0
                and paired_summary["score"] > champion_summary["score"]
            ):
                holdout_summary, holdout_rows, holdout_residues = evaluate_policy(
                    args, holdout_seeds, holdout_baseline_rows, candidate
                )
                gate_ok, gate_failures = passes_acceptance_gate(
                    paired_summary, holdout_summary, champion_summary
                )
            else:
                gate_ok = False
                gate_failures = ["paired gate failed before holdout evaluation"]

            report = {
                "candidate": candidate.to_json_dict(),
                "paired_summary": paired_summary,
                "paired_rows": paired_rows,
                "paired_residues": paired_residues,
                "holdout_summary": holdout_summary,
                "holdout_rows": holdout_rows,
                "holdout_residues": holdout_residues,
                "accepted_gate": gate_ok,
                "gate_failures": gate_failures,
                "provenance_chain": build_provenance_chain(
                    candidate,
                    paired_summary,
                    holdout_summary,
                    gate_ok,
                    champion.name,
                ),
            }
            candidate_reports.append(report)
            if gate_ok:
                acceptable.append((
                    paired_summary["score"],
                    candidate,
                    paired_summary,
                    paired_rows,
                    paired_residues,
                    report,
                ))

        if acceptable:
            acceptable.sort(key=lambda item: item[0], reverse=True)
            _, champion, champion_summary, champion_rows, champion_residues, accepted_report = acceptable[0]
            accepted.append(accepted_report)
            accepted_decision = True
        else:
            accepted_decision = False

        cycles.append({
            "cycle": cycle,
            "parent_policy": cycles[-1]["champion"]["name"],
            "residues_used": champion_residues if accepted_decision else cycles[-1]["residues_used"],
            "generated_policy_candidates": [
                candidate.to_json_dict() for candidate in candidates
            ],
            "candidate_reports": candidate_reports,
            "accepted": accepted_decision,
            "champion": champion_summary,
            "next_parent_policy": champion.name,
        })

    omega_validation_valid = final_validation_valid(
        accepted_count=len(accepted),
        required_accepted_improvements=args.min_accepted_improvements,
        bootstrap_residue_count=len(cycles[0]["residues_used"]),
        champion_summary=champion_summary,
    )

    payload = {
        "omega_validation_valid": omega_validation_valid,
        "accepted_improvements": len(accepted),
        "required_accepted_improvements": args.min_accepted_improvements,
        "paired_seeds": paired_seeds,
        "holdout_seeds": holdout_seeds,
        "baseline_rows": baseline_rows,
        "holdout_baseline_rows": holdout_baseline_rows,
        "cycles": cycles,
        "accepted_reports": accepted,
        "final_champion": champion_summary,
        "final_parent_policy": champion.to_json_dict(),
        "boundary": BOUNDARY_STATEMENT,
    }

    Path(args.report_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for cycle in cycles:
            champ = cycle["champion"]
            print(
                f"cycle={cycle['cycle']} accepted={cycle['accepted']} "
                f"champion={champ['name']} score={champ['score']:.4f} "
                f"delta={champ['mean_bpc_delta']:+.4f} "
                f"archive={champ['total_generated_archive_insertions']} "
                f"evals={champ['total_generated_evaluations']}"
            )
        print()
        print(json.dumps({
            "omega_validation_valid": omega_validation_valid,
            "accepted_improvements": len(accepted),
            "final_champion": champion_summary,
            "boundary": BOUNDARY_STATEMENT,
        }, indent=2, sort_keys=True))

    return 0 if omega_validation_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
