"""
Outer-loop recursive self-improvement search for RSI-NAS.

This script does not prove open-ended RSI. It implements the next necessary
mechanism: the system mutates and selects its own EIE/AFIRSI policy from
validation evidence, then uses the selected policy as the parent for the next
cycle.

The acceptance signal is deliberately concrete:
- generated modules must receive real evaluations,
- some generated modules must enter the MAP-Elites archive,
- the meta-operator policy must update from generated-module evidence,
- mean BPC must beat the SELF-MODIFY baseline for the same seed curriculum,
- a mutated policy must outperform the previous champion policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Dict, List

from rsi_nas import EIEConfig
from validate_eie import parse_seeds, run_case


@dataclass
class PolicyCandidate:
    name: str
    config: EIEConfig
    generated_min_evaluations: int = 1


def candidate_score(summary: Dict) -> float:
    if not summary["mechanism_valid"]:
        return -999.0
    return float(round(
        summary["mean_bpc_delta"]
        + 0.0010 * summary["total_generated_archive_insertions"]
        + 0.0002 * summary["total_generated_evaluations"]
        + 0.0002 * summary["total_meta_operator_policy_updates"]
        + 0.0050 * summary["eie_wins"],
        6,
    ))


def summarize_candidate(
    candidate: PolicyCandidate,
    baseline_rows: List[Dict],
    candidate_rows: List[Dict],
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
    total_instrumented = sum(
        row["instrumented_candidates"] for row in candidate_rows
    )

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
        "name": candidate.name,
        "config": asdict(candidate.config),
        "generated_min_evaluations": candidate.generated_min_evaluations,
        "mean_best_bpc": round(mean(candidate_bpc), 4),
        "baseline_mean_best_bpc": round(mean(baseline_bpc), 4),
        "mean_bpc_delta": mean_bpc_delta,
        "eie_wins": wins,
        "paired_seeds": len(candidate_rows),
        "total_generated_archive_insertions": total_archive_insertions,
        "total_generated_evaluations": total_evaluations,
        "total_meta_operator_policy_updates": total_policy_updates,
        "total_eie_residues": total_residues,
        "mechanism_valid": mechanism_valid,
    }
    summary["score"] = candidate_score(summary)
    return summary


def evaluate_candidate(
    args: argparse.Namespace,
    seeds: List[int],
    baseline_rows: List[Dict],
    candidate: PolicyCandidate,
) -> Dict:
    run_args = copy.copy(args)
    run_args.generated_min_evaluations = candidate.generated_min_evaluations
    rows = [
        run_case(
            run_args,
            seed,
            enable_eie=True,
            eie_config=candidate.config,
            condition_name=candidate.name,
        )
        for seed in seeds
    ]
    return summarize_candidate(candidate, baseline_rows, rows)


def mutate_candidate(parent: PolicyCandidate, cycle: int) -> List[PolicyCandidate]:
    cfg = parent.config
    return [
        PolicyCandidate(
            name=f"cycle{cycle}_clean_probe_archive",
            config=EIEConfig(
                grace_multiplier=max(3.0, cfg.grace_multiplier),
                probe_rate=1.0,
                clean_probe_first_eval=True,
                meta_eval_gain=max(0.25, cfg.meta_eval_gain),
                meta_archive_gain=max(2.0, cfg.meta_archive_gain + 0.5),
                meta_fitness_gain=max(4.0, cfg.meta_fitness_gain),
                meta_exploration_floor=max(0.15, cfg.meta_exploration_floor),
            ),
            generated_min_evaluations=max(1, parent.generated_min_evaluations),
        ),
        PolicyCandidate(
            name=f"cycle{cycle}_fitness_weighted",
            config=EIEConfig(
                grace_multiplier=max(2.0, cfg.grace_multiplier),
                probe_rate=1.0,
                clean_probe_first_eval=True,
                meta_eval_gain=max(0.15, cfg.meta_eval_gain),
                meta_archive_gain=max(1.5, cfg.meta_archive_gain),
                meta_fitness_gain=max(5.0, cfg.meta_fitness_gain + 1.0),
                meta_exploration_floor=max(0.10, cfg.meta_exploration_floor * 0.8),
            ),
            generated_min_evaluations=max(1, parent.generated_min_evaluations),
        ),
        PolicyCandidate(
            name=f"cycle{cycle}_evidence_budget",
            config=EIEConfig(
                grace_multiplier=max(4.0, cfg.grace_multiplier + 1.0),
                probe_rate=1.0,
                clean_probe_first_eval=True,
                meta_eval_gain=max(0.35, cfg.meta_eval_gain + 0.1),
                meta_archive_gain=max(2.0, cfg.meta_archive_gain),
                meta_fitness_gain=max(4.0, cfg.meta_fitness_gain),
                meta_exploration_floor=max(0.20, cfg.meta_exploration_floor),
            ),
            generated_min_evaluations=max(2, parent.generated_min_evaluations),
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--expansion-interval", type=int, default=1)
    parser.add_argument("--pruning-interval", type=int, default=1)
    parser.add_argument("--generated-min-evaluations", type=int, default=1)
    parser.add_argument("--min-accepted-improvements", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    seeds = parse_seeds(args.seeds)
    baseline_rows = [
        run_case(args, seed, enable_eie=False, condition_name="SELF-MODIFY")
        for seed in seeds
    ]

    champion = PolicyCandidate(
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
    )
    champion_summary = evaluate_candidate(args, seeds, baseline_rows, champion)
    accepted = []
    cycles = [
        {
            "cycle": 0,
            "champion": champion_summary,
            "accepted": False,
        }
    ]

    for cycle in range(1, args.cycles + 1):
        candidates = mutate_candidate(champion, cycle)
        candidate_summaries = [
            evaluate_candidate(args, seeds, baseline_rows, candidate)
            for candidate in candidates
        ]
        best_summary = max(candidate_summaries, key=lambda item: item["score"])
        improved = bool(best_summary["score"] > champion_summary["score"])
        if improved:
            accepted.append(best_summary)
            champion_index = candidate_summaries.index(best_summary)
            champion = candidates[champion_index]
            champion_summary = best_summary

        cycles.append({
            "cycle": cycle,
            "candidates": candidate_summaries,
            "champion": champion_summary,
            "accepted": improved,
        })

    open_ended_proxy_valid = bool(
        len(accepted) >= args.min_accepted_improvements
        and champion_summary["mechanism_valid"]
        and champion_summary["mean_bpc_delta"] > 0.0
    )
    payload = {
        "open_ended_proxy_valid": open_ended_proxy_valid,
        "accepted_improvements": len(accepted),
        "required_accepted_improvements": args.min_accepted_improvements,
        "final_champion": champion_summary,
        "cycles": cycles,
        "boundary": (
            "This is an outer recursive policy-improvement loop. Passing it "
            "shows self-modification of RSI policy across validation cycles, "
            "not mathematical proof of unbounded open-ended RSI."
        ),
    }

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
            "open_ended_proxy_valid": open_ended_proxy_valid,
            "accepted_improvements": len(accepted),
            "final_champion": champion_summary,
            "boundary": payload["boundary"],
        }, indent=2, sort_keys=True))

    return 0 if open_ended_proxy_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
