"""Export real RSI-NAS FailureResidue objects into AFIRSI-OMEGA schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Dict, List

from rsi_nas import EIEConfig

from .schemas import StructuredAFIRSIResidue


def _stable_id(payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=8).hexdigest()


def _policy_snapshot(engine, residue) -> Dict:
    config = getattr(engine.meta, "eie_config", EIEConfig())
    policy = engine.meta.pruning_policy
    return {
        "eie_config": asdict(config),
        "pruning_policy": {
            "generated_grace_generations": policy.generated_grace_generations,
            "generated_min_evaluations": policy.generated_min_evaluations,
            "generated_probe_rate": policy.generated_probe_rate,
            "mutation_count": policy.mutation_count,
            "last_mutation_generation": policy.last_mutation_generation,
        },
        "policy_before": dict(residue.policy_before),
        "policy_after": dict(residue.policy_after),
    }


def _failure_signature(record, residue) -> List[str]:
    signature = [residue.residue_type]
    if residue.evaluations == 0:
        signature.append("premature_prune_before_evaluation")
    elif record.archive_insertions == 0:
        signature.append("archive_starvation_after_evaluation")
    if record.best_fitness > 0.0 and record.elite_uses == 0:
        signature.append("fitness_without_elite_use")
    if record.protected_from_prune > 1:
        signature.append("repeated_prune_protection")
    if record.prune_attempts > record.protected_from_prune:
        signature.append("eventual_prune_after_protection")
    return sorted(set(signature))


def _requested_instruments(signature: List[str]) -> List[str]:
    requested = set()
    if "premature_prune_before_evaluation" in signature:
        requested.update({
            "pruning_grace",
            "evaluation_budget",
            "generated_module_scaffold_strategy",
        })
    if "archive_starvation_after_evaluation" in signature:
        requested.update({
            "archive_insertion_priority",
            "meta_archive_weighting",
        })
    if "fitness_without_elite_use" in signature:
        requested.update({
            "elite_use_weighting",
            "fitness_weighting",
        })
    if "repeated_prune_protection" in signature:
        requested.update({
            "residue_pressure_weighting",
            "evaluator_disagreement_term",
        })
    return sorted(requested)


def export_residues(engine, seed: int, condition: str) -> List[StructuredAFIRSIResidue]:
    """Collect real failure residue evidence from a completed RSI-NAS run.

    The exporter only emits residues already produced by
    ``engine.meta.pruning_residues``. If the run produced no real residues, the
    return value is empty.
    """

    records = {record.name: record for record in engine.registry.all_generated_records()}
    exported: List[StructuredAFIRSIResidue] = []

    for residue in engine.meta.pruning_residues:
        record = records.get(residue.module_name)
        if record is None:
            continue
        signature = _failure_signature(record, residue)
        id_payload = {
            "seed": seed,
            "condition": condition,
            "generation": residue.generation,
            "module": residue.module_name,
            "type": residue.residue_type,
            "age": residue.age,
            "evaluations": residue.evaluations,
            "prune_attempts": record.prune_attempts,
        }
        exported.append(StructuredAFIRSIResidue(
            residue_id=f"afirsi-residue-{_stable_id(id_payload)}",
            generation=int(residue.generation),
            seed=int(seed),
            condition=condition,
            residue_type=residue.residue_type,
            generated_module_name=residue.module_name,
            source_action=record.source_action,
            birth_generation=int(record.birth_generation),
            evaluations=int(record.evaluations),
            archive_insertions=int(record.archive_insertions),
            elite_uses=int(record.elite_uses),
            best_fitness=round(float(record.best_fitness), 8),
            prune_attempts=int(record.prune_attempts),
            protected_from_prune=int(record.protected_from_prune),
            age_at_prune_attempt=int(residue.age),
            current_policy_snapshot=_policy_snapshot(engine, residue),
            observed_failure_signature=signature,
            requested_instrument_classes=_requested_instruments(signature),
        ))

    return exported
