"""
Reproducible EIE/AFIRSI validation for RSI-NAS.

This script checks the specific claim made by the EIE integration:
baseline self-modification prunes generated modules before they propagate,
while AFIRSI-EIE detects that failure residue, mutates the pruning/probing
instrument, and gives generated modules real SGD evaluation opportunities.

It does not claim that open-ended RSI is solved. Fitness improvement requires
longer GPU-scale ablations.
"""

from __future__ import annotations

import argparse
import json
import random
from statistics import mean
from typing import Dict, List

import numpy as np
import torch

from rsi_nas import build_rsi_nas


def parse_seeds(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def run_case(args: argparse.Namespace, seed: int, enable_eie: bool) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    engine = build_rsi_nas(
        d_model=args.d_model,
        train_steps=args.train_steps,
        expansion_interval=args.expansion_interval,
        pruning_interval=args.pruning_interval,
        generated_min_evaluations=args.generated_min_evaluations,
        enable_eie=enable_eie,
    )
    history = engine.run(
        generations=args.generations,
        population_size=args.population_size,
    )
    final = history[-1]
    generated_records = engine.registry.all_generated_records()

    return {
        "seed": seed,
        "condition": "AFIRSI-EIE" if enable_eie else "SELF-MODIFY",
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
    }


def summarize(results: List[Dict]) -> Dict:
    by_condition = {}
    for condition in ["SELF-MODIFY", "AFIRSI-EIE"]:
        rows = [r for r in results if r["condition"] == condition]
        by_condition[condition] = {
            "mean_best_bpc": round(mean(r["best_bpc"] for r in rows), 4),
            "mean_generated_modules": round(
                mean(r["generated_modules"] for r in rows), 4
            ),
            "total_eie_residues": sum(r["eie_residues"] for r in rows),
            "total_instrument_mutations": sum(
                r["instrument_mutations"] for r in rows
            ),
            "total_instrumented_candidates": sum(
                r["instrumented_candidates"] for r in rows
            ),
            "total_generated_evaluations": sum(
                r["generated_evaluations"] for r in rows
            ),
            "total_generated_archive_insertions": sum(
                r["generated_archive_insertions"] for r in rows
            ),
            "total_meta_operator_policy_updates": sum(
                r["meta_operator_policy_updates"] for r in rows
            ),
        }

    baseline = by_condition["SELF-MODIFY"]
    eie = by_condition["AFIRSI-EIE"]
    paired = {}
    for row in results:
        paired.setdefault(row["seed"], {})[row["condition"]] = row["best_bpc"]
    eie_wins = sum(
        1
        for row in paired.values()
        if row.get("AFIRSI-EIE", 99.0) < row.get("SELF-MODIFY", 0.0)
    )
    mean_bpc_delta = round(
        baseline["mean_best_bpc"] - eie["mean_best_bpc"],
        4,
    )
    mechanism_valid = bool(
        baseline["total_generated_evaluations"] == 0
        and eie["total_eie_residues"] > 0
        and eie["total_instrument_mutations"] > 0
        and eie["total_instrumented_candidates"] > 0
        and eie["total_generated_evaluations"] > 0
        and eie["total_generated_archive_insertions"] > 0
        and eie["total_meta_operator_policy_updates"] > 0
        and mean_bpc_delta > 0.0
    )

    return {
        "conditions": by_condition,
        "mean_bpc_delta_self_modify_minus_eie": mean_bpc_delta,
        "eie_wins": eie_wins,
        "paired_seeds": len(paired),
        "mechanism_valid": mechanism_valid,
        "bounded_claim": (
            "Validated: EIE detects pruning-propagation residue and changes "
            "the live pruning/probing instruments so generated modules receive "
            "real SGD evaluations. It also updates meta-operator weights from "
            "generated-module archive evidence. The default CPU validation "
            "requires positive mean BPC delta, but does not prove open-ended "
            "RSI or per-seed dominance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="7,11,19,23,29")
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population-size", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--expansion-interval", type=int, default=1)
    parser.add_argument("--pruning-interval", type=int, default=1)
    parser.add_argument("--generated-min-evaluations", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    results = []
    for seed in seeds:
        results.append(run_case(args, seed, enable_eie=False))
        results.append(run_case(args, seed, enable_eie=True))

    summary = summarize(results)
    payload = {"runs": results, "summary": summary}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in results:
            print(
                f"{row['condition']:11s} seed={row['seed']:3d} "
                f"bpc={row['best_bpc']:.4f} "
                f"generated={row['generated_modules']:2d} "
                f"evals={row['generated_evaluations']:2d} "
                f"archive_insertions={row['generated_archive_insertions']:2d} "
                f"residues={row['eie_residues']:2d} "
                f"mutations={row['instrument_mutations']:2d} "
                f"meta_updates={row['meta_operator_policy_updates']:2d}"
            )
        print()
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0 if summary["mechanism_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
