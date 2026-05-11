"""Bounded recursive AFIRSI/EIE instrument-policy improvement kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from afirsi_core import InstrumentMutationContract, ProblemSpaceVersionGraph
from omega_adapter import (
    OMEGAInstrumentGenerator,
    OMEGAPolicyCandidate,
    StructuredAFIRSIResidue,
    apply_instrument_patch,
    export_residues,
)
from omega_adapter.schemas import InstrumentPatch
from rsi_nas import EIEConfig, build_rsi_nas
from validate_eie import parse_seeds


BOUNDARY_STATEMENT = (
    "This validates a bounded recursive self-improvement kernel over AFIRSI/EIE "
    "instrument policy inside RSI-NAS. It is not proof of unbounded open-ended "
    "RSI, AGI, ASI, or real-world autonomous self-improvement."
)


SECOND_ORDER_RESIDUE_TYPES = {
    "ARCHIVE_STAGNATION",
    "OPERATOR_GENERATOR_MODE_COLLAPSE",
    "EVALUATOR_NOISE_OR_OVERFIT",
    "POLICY_SATURATION",
    "SCAFFOLD_BIAS",
    "META_OPERATOR_IMBALANCE",
    "PATCH_EFFECTIVENESS_FAILURE",
}


PATCH_FAMILY_BY_RESIDUE_TYPE = {
    "PRUNING_PROPAGATION_RACE": "pruning_race_evidence_budget",
    "ARCHIVE_STAGNATION": "archive_stagnation",
    "OPERATOR_GENERATOR_MODE_COLLAPSE": "operator_generator_mode_collapse",
    "EVALUATOR_NOISE_OR_OVERFIT": "evaluator_noise_holdout_stability",
    "POLICY_SATURATION": "policy_saturation_next_bottleneck",
    "SCAFFOLD_BIAS": "scaffold_bias_mixed_transfer",
    "META_OPERATOR_IMBALANCE": "meta_operator_imbalance_rebalance",
    "PATCH_EFFECTIVENESS_FAILURE": "patch_effectiveness_failure",
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_diagnostic_id(prefix: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).hexdigest()
    return f"{prefix}-{digest}"


def _positive_mechanism(summary: Optional[Dict[str, Any]]) -> bool:
    if not summary:
        return False
    return bool(
        summary["total_generated_evaluations"] > 0
        and summary["total_generated_archive_insertions"] > 0
        and summary["total_eie_residues"] > 0
        and summary["total_instrumented_candidates"] > 0
        and summary["total_meta_operator_policy_updates"] > 0
    )


def _compact_generated_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": record.get("name"),
        "birth_generation": record.get("birth_generation"),
        "source_action": record.get("source_action"),
        "evaluations": record.get("evaluations", 0),
        "archive_insertions": record.get("archive_insertions", 0),
        "elite_uses": record.get("elite_uses", 0),
        "best_fitness": record.get("best_fitness", 0.0),
        "prune_attempts": record.get("prune_attempts", 0),
        "protected_from_prune": record.get("protected_from_prune", 0),
    }


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        key: value
        for key, value in row.items()
        if key != "generated_records"
    }
    compact["generated_records"] = [
        _compact_generated_record(record)
        for record in row.get("generated_records", [])
    ]
    return compact


def _compact_residue(residue: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "residue_id": residue.get("residue_id"),
        "generation": residue.get("generation"),
        "seed": residue.get("seed"),
        "condition": residue.get("condition"),
        "recursive_policy_id": residue.get("recursive_policy_id"),
        "recursive_parent_policy_id": residue.get("recursive_parent_policy_id"),
        "residue_type": residue.get("residue_type"),
        "generated_module_name": residue.get("generated_module_name"),
        "source_action": residue.get("source_action"),
        "birth_generation": residue.get("birth_generation"),
        "evaluations": residue.get("evaluations"),
        "archive_insertions": residue.get("archive_insertions"),
        "elite_uses": residue.get("elite_uses"),
        "best_fitness": residue.get("best_fitness"),
        "prune_attempts": residue.get("prune_attempts"),
        "protected_from_prune": residue.get("protected_from_prune"),
        "age_at_prune_attempt": residue.get("age_at_prune_attempt"),
        "observed_failure_signature": residue.get("observed_failure_signature", []),
        "requested_instrument_classes": residue.get(
            "requested_instrument_classes",
            [],
        ),
        "subject_type": residue.get("subject_type", "generated_module"),
        "parent_policy_id": residue.get("parent_policy_id"),
        "candidate_policy_id": residue.get("candidate_policy_id"),
        "cycle_index": residue.get("cycle_index"),
        "patch_id": residue.get("patch_id"),
        "patch_family": residue.get("patch_family"),
        "suspected_bottleneck_category": residue.get(
            "suspected_bottleneck_category"
        ),
        "diagnostic_payload": residue.get("diagnostic_payload", {}),
        "structured_rejection_reasons": residue.get(
            "structured_rejection_reasons",
            [],
        ),
    }


@dataclass
class RecursiveDiagnosticRecord:
    diagnostic_id: str
    residue_type: str
    parent_policy_id: str
    candidate_policy_id: Optional[str]
    cycle_index: int
    paired_validation_result: Optional[Dict[str, Any]]
    holdout_validation_result: Optional[Dict[str, Any]]
    score_delta_against_parent: float
    bpc_delta_against_parent: float
    generated_evaluations: int
    archive_insertions: int
    meta_policy_updates: int
    residues_produced_under_parent: List[str]
    patch_id: Optional[str]
    patch_fields_changed: List[str]
    patch_family: Optional[str]
    rejection_reasons: List[Dict[str, Any]]
    suspected_bottleneck_category: str
    patch_equivalence_group: Optional[str] = None
    equivalent_failed_patch_ids: List[str] = field(default_factory=list)
    evidence_source: str = "bounded_rsi.validation"

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecursivePolicyVersion:
    policy_id: str
    parent_policy_id: Optional[str]
    problem_space_version_id: Optional[str]
    source_residue_ids: List[str]
    source_patch_id: Optional[str]
    eie_config: Dict[str, Any]
    generated_min_evaluations: int
    creation_cycle: int
    validation_summary: Dict[str, Any]
    accepted: bool
    rejection_reasons: List[str]
    lineage_depth: int
    policy_name: str
    candidate: Optional[OMEGAPolicyCandidate] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "parent_policy_id": self.parent_policy_id,
            "problem_space_version_id": self.problem_space_version_id,
            "source_residue_ids": list(self.source_residue_ids),
            "source_patch_id": self.source_patch_id,
            "eie_config": dict(self.eie_config),
            "generated_min_evaluations": self.generated_min_evaluations,
            "creation_cycle": self.creation_cycle,
            "validation_summary": self.validation_summary,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "lineage_depth": self.lineage_depth,
        }


@dataclass
class RecursiveValidationResult:
    candidate_policy_id: str
    candidate_policy_name: str
    parent_policy_id: str
    comparison_parent_policy_id: str
    source_residue_ids: List[str]
    patch_id: Optional[str]
    contract_validated: bool
    paired_evaluated: bool
    holdout_evaluated: bool
    paired_summary: Optional[Dict[str, Any]] = None
    holdout_summary: Optional[Dict[str, Any]] = None
    paired_rows: List[Dict[str, Any]] = field(default_factory=list)
    holdout_rows: List[Dict[str, Any]] = field(default_factory=list)
    paired_residues: List[Dict[str, Any]] = field(default_factory=list)
    holdout_residues: List[Dict[str, Any]] = field(default_factory=list)
    recursive_score: float = -999.0
    holdout_recursive_score: float = -999.0
    valid_patch_rate: float = 0.0
    residue_to_patch_conversion_success: float = 0.0
    invalid_patch_penalty: float = 0.0
    regression_penalty: float = 0.0
    novelty_penalty: float = 0.0
    patch_family: Optional[str] = None
    patch_fields_changed: List[str] = field(default_factory=list)
    patch_equivalence_group: Optional[str] = None
    patch_novel: bool = True
    equivalent_failed_patch_ids: List[str] = field(default_factory=list)
    suspected_bottleneck_category: Optional[str] = None
    structured_rejection_reasons: List[Dict[str, Any]] = field(default_factory=list)
    score_breakdown: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    problem_space_version_id: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "candidate_policy_id": self.candidate_policy_id,
            "candidate_policy_name": self.candidate_policy_name,
            "parent_policy_id": self.parent_policy_id,
            "comparison_parent_policy_id": self.comparison_parent_policy_id,
            "source_residue_ids": list(self.source_residue_ids),
            "patch_id": self.patch_id,
            "contract_validated": self.contract_validated,
            "paired_evaluated": self.paired_evaluated,
            "holdout_evaluated": self.holdout_evaluated,
            "paired_summary": self.paired_summary,
            "holdout_summary": self.holdout_summary,
            "paired_rows": [_compact_row(row) for row in self.paired_rows],
            "holdout_rows": [_compact_row(row) for row in self.holdout_rows],
            "paired_residues": [
                _compact_residue(residue) for residue in self.paired_residues
            ],
            "holdout_residues": [
                _compact_residue(residue) for residue in self.holdout_residues
            ],
            "recursive_score": self.recursive_score,
            "holdout_recursive_score": self.holdout_recursive_score,
            "valid_patch_rate": self.valid_patch_rate,
            "residue_to_patch_conversion_success": (
                self.residue_to_patch_conversion_success
            ),
            "invalid_patch_penalty": self.invalid_patch_penalty,
            "regression_penalty": self.regression_penalty,
            "novelty_penalty": self.novelty_penalty,
            "patch_family": self.patch_family,
            "patch_fields_changed": list(self.patch_fields_changed),
            "patch_equivalence_group": self.patch_equivalence_group,
            "patch_novel": self.patch_novel,
            "equivalent_failed_patch_ids": list(self.equivalent_failed_patch_ids),
            "suspected_bottleneck_category": self.suspected_bottleneck_category,
            "structured_rejection_reasons": [
                dict(reason) for reason in self.structured_rejection_reasons
            ],
            "score_breakdown": dict(self.score_breakdown),
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "problem_space_version_id": self.problem_space_version_id,
        }


@dataclass
class RecursiveImprovementCycle:
    cycle_index: int
    parent_policy_id: str
    parent_problem_space_version_id: Optional[str]
    parent_paired_summary: Dict[str, Any]
    parent_holdout_summary: Dict[str, Any]
    parent_paired_rows: List[Dict[str, Any]]
    parent_holdout_rows: List[Dict[str, Any]]
    source_residue_ids: List[str]
    source_residues: List[Dict[str, Any]]
    patch_ids: List[str]
    candidate_policies: List[RecursivePolicyVersion]
    validation_results: List[RecursiveValidationResult]
    diagnostic_records: List[RecursiveDiagnosticRecord] = field(default_factory=list)
    source_residue_distribution: Dict[str, int] = field(default_factory=dict)
    candidate_patch_families: List[str] = field(default_factory=list)
    rejected_patch_equivalence_groups: Dict[str, List[str]] = field(default_factory=dict)
    suggested_next_bottleneck: Optional[str] = None
    accepted_policy_id: Optional[str] = None
    accepted_patch_id: Optional[str] = None
    next_parent_policy_id: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "parent_policy_id": self.parent_policy_id,
            "parent_problem_space_version_id": self.parent_problem_space_version_id,
            "parent_paired_summary": self.parent_paired_summary,
            "parent_holdout_summary": self.parent_holdout_summary,
            "parent_paired_rows": [
                _compact_row(row) for row in self.parent_paired_rows
            ],
            "parent_holdout_rows": [
                _compact_row(row) for row in self.parent_holdout_rows
            ],
            "source_residue_ids": list(self.source_residue_ids),
            "source_residues": [
                _compact_residue(residue) for residue in self.source_residues
            ],
            "patch_ids": list(self.patch_ids),
            "candidate_policies": [
                policy.to_json_dict() for policy in self.candidate_policies
            ],
            "validation_results": [
                result.to_json_dict() for result in self.validation_results
            ],
            "diagnostic_records": [
                record.to_json_dict() for record in self.diagnostic_records
            ],
            "source_residue_distribution": dict(self.source_residue_distribution),
            "candidate_patch_families": list(self.candidate_patch_families),
            "rejected_patch_equivalence_groups": {
                key: list(value)
                for key, value in self.rejected_patch_equivalence_groups.items()
            },
            "suggested_next_bottleneck": self.suggested_next_bottleneck,
            "accepted_policy_id": self.accepted_policy_id,
            "accepted_patch_id": self.accepted_patch_id,
            "next_parent_policy_id": self.next_parent_policy_id,
        }


@dataclass
class RecursiveLineageReport:
    bounded_recursive_success: bool
    partial_recursive_success: bool
    accepted_policy_count: int
    cycle_count: int
    final_parent_policy_id: str
    policy_lineage: List[RecursivePolicyVersion]
    cycles: List[RecursiveImprovementCycle]
    p1_became_parent_of_cycle_2: bool
    p2_generated_from_p1_derived_residues: bool
    p2_compared_against_p1: bool
    p2_failure_diagnosis: Dict[str, Any] = field(default_factory=dict)
    p1_residue_distribution: Dict[str, int] = field(default_factory=dict)
    p2_candidate_patch_families: List[str] = field(default_factory=list)
    rejected_patch_equivalence_groups: Dict[str, List[str]] = field(default_factory=dict)
    parent_vs_child_score_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    suggested_next_bottleneck: Optional[str] = None
    boundary: str = BOUNDARY_STATEMENT

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "bounded_recursive_success": self.bounded_recursive_success,
            "partial_recursive_success": self.partial_recursive_success,
            "accepted_policy_count": self.accepted_policy_count,
            "cycle_count": self.cycle_count,
            "final_parent_policy_id": self.final_parent_policy_id,
            "full_policy_lineage": [
                policy.to_json_dict() for policy in self.policy_lineage
            ],
            "cycles": [cycle.to_json_dict() for cycle in self.cycles],
            "p1_became_parent_of_cycle_2": self.p1_became_parent_of_cycle_2,
            "p2_generated_from_p1_derived_residues": (
                self.p2_generated_from_p1_derived_residues
            ),
            "p2_compared_against_p1": self.p2_compared_against_p1,
            "p2_failure_diagnosis": dict(self.p2_failure_diagnosis),
            "p1_residue_distribution": dict(self.p1_residue_distribution),
            "p2_candidate_patch_families": list(self.p2_candidate_patch_families),
            "rejected_patch_equivalence_groups": {
                key: list(value)
                for key, value in self.rejected_patch_equivalence_groups.items()
            },
            "parent_vs_child_score_breakdown": [
                dict(row) for row in self.parent_vs_child_score_breakdown
            ],
            "suggested_next_bottleneck": self.suggested_next_bottleneck,
            "boundary": self.boundary,
        }


def make_seed_policy() -> OMEGAPolicyCandidate:
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
        rationale=["Initial weak policy used as P0 for bounded recursion."],
        provenance={"source": "bounded_rsi.bootstrap"},
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


class BoundedRSIKernel:
    """Bounded recursive kernel over AFIRSI/EIE instrument policies."""

    def __init__(
        self,
        args: argparse.Namespace,
        generator: Optional[OMEGAInstrumentGenerator] = None,
        contract: Optional[InstrumentMutationContract] = None,
        problem_space_graph: Optional[ProblemSpaceVersionGraph] = None,
    ) -> None:
        self.args = args
        self.paired_seeds = parse_seeds(args.seeds)
        self.holdout_seeds = parse_seeds(args.holdout_seeds)
        self.generator = generator or OMEGAInstrumentGenerator()
        self.contract = contract or InstrumentMutationContract()
        self.problem_space_graph = (
            problem_space_graph or ProblemSpaceVersionGraph(
                root_policy=self._instrument_policy_dict(make_seed_policy())
            )
        )
        self.accepted_lineage: List[RecursivePolicyVersion] = []
        self.cycles: List[RecursiveImprovementCycle] = []
        self._next_policy_number = 1
        self.failed_patch_memory: List[Dict[str, Any]] = []

    def run(self) -> RecursiveLineageReport:
        p0_candidate = make_seed_policy()
        parent = RecursivePolicyVersion(
            policy_id="P0",
            policy_name=p0_candidate.name,
            parent_policy_id=None,
            problem_space_version_id=self.problem_space_graph.active_version_id,
            source_residue_ids=[],
            source_patch_id=p0_candidate.patch.patch_id,
            eie_config=asdict(p0_candidate.config),
            generated_min_evaluations=p0_candidate.generated_min_evaluations,
            creation_cycle=0,
            validation_summary={},
            accepted=True,
            rejection_reasons=[],
            lineage_depth=0,
            candidate=p0_candidate,
        )
        self.accepted_lineage = [parent]

        for cycle_index in range(1, int(self.args.cycles) + 1):
            cycle = self.run_cycle(cycle_index, parent)
            self.cycles.append(cycle)
            accepted = next(
                (
                    policy for policy in cycle.candidate_policies
                    if policy.accepted and policy.policy_id == cycle.accepted_policy_id
                ),
                None,
            )
            if accepted is not None:
                parent = accepted
                self.accepted_lineage.append(accepted)

        accepted_count = max(0, len(self.accepted_lineage) - 1)
        p1_became_parent = (
            len(self.cycles) >= 2
            and self.cycles[1].parent_policy_id == "P1"
        )
        p2_from_p1 = bool(
            p1_became_parent
            and self.cycles[1].source_residue_ids
            and all(
                residue.get("recursive_policy_id") == "P1"
                for residue in self.cycles[1].source_residues
            )
        )
        p2_against_p1 = bool(
            p1_became_parent
            and self.cycles[1].validation_results
            and all(
                result.comparison_parent_policy_id == "P1"
                for result in self.cycles[1].validation_results
                if result.paired_evaluated
            )
        )
        bounded_success = bool(
            accepted_count >= 2
            and p1_became_parent
            and p2_from_p1
            and p2_against_p1
        )
        partial_success = bool(
            accepted_count == 1
            and not bounded_success
            and p1_became_parent
            and len(self.cycles) >= 2
        )
        p2_cycle = self.cycles[1] if len(self.cycles) >= 2 else None
        p2_failure_diagnosis = (
            self._cycle_failure_diagnosis(p2_cycle)
            if p2_cycle is not None
            else {}
        )
        return RecursiveLineageReport(
            bounded_recursive_success=bounded_success,
            partial_recursive_success=partial_success,
            accepted_policy_count=accepted_count,
            cycle_count=len(self.cycles),
            final_parent_policy_id=parent.policy_id,
            policy_lineage=self.accepted_lineage,
            cycles=self.cycles,
            p1_became_parent_of_cycle_2=p1_became_parent,
            p2_generated_from_p1_derived_residues=p2_from_p1,
            p2_compared_against_p1=p2_against_p1,
            p2_failure_diagnosis=p2_failure_diagnosis,
            p1_residue_distribution=(
                p2_cycle.source_residue_distribution if p2_cycle is not None else {}
            ),
            p2_candidate_patch_families=(
                p2_cycle.candidate_patch_families if p2_cycle is not None else []
            ),
            rejected_patch_equivalence_groups=(
                p2_cycle.rejected_patch_equivalence_groups
                if p2_cycle is not None else {}
            ),
            parent_vs_child_score_breakdown=(
                [
                    {
                        "candidate_policy_id": result.candidate_policy_id,
                        "parent_policy_id": result.comparison_parent_policy_id,
                        "patch_id": result.patch_id,
                        "patch_family": result.patch_family,
                        "score_breakdown": result.score_breakdown,
                        "paired_summary": result.paired_summary,
                        "holdout_summary": result.holdout_summary,
                        "rejection_reasons": list(result.rejection_reasons),
                    }
                    for result in p2_cycle.validation_results
                ]
                if p2_cycle is not None else []
            ),
            suggested_next_bottleneck=(
                p2_cycle.suggested_next_bottleneck if p2_cycle is not None else None
            ),
        )

    def run_cycle(
        self,
        cycle_index: int,
        parent: RecursivePolicyVersion,
    ) -> RecursiveImprovementCycle:
        parent_paired_summary, parent_paired_rows, source_residues = (
            self.evaluate_policy_version(
                parent,
                self.paired_seeds,
                comparison_rows=None,
            )
        )
        parent_holdout_summary, parent_holdout_rows, _ = self.evaluate_policy_version(
            parent,
            self.holdout_seeds,
            comparison_rows=None,
        )
        source_residues = self._augment_parent_residues(
            cycle_index=cycle_index,
            parent=parent,
            parent_paired_summary=parent_paired_summary,
            parent_paired_rows=parent_paired_rows,
            source_residues=source_residues,
        )
        residue_ids = [residue["residue_id"] for residue in source_residues]
        source_residue_distribution = dict(Counter(
            residue["residue_type"] for residue in source_residues
        ))
        patches = self.generator.generate(
            residues=[
                self._structured_residue_from_dict(r)
                for r in source_residues
            ],
            parent_config=parent.candidate.config,
            parent_generated_min_evaluations=parent.generated_min_evaluations,
            cycle=cycle_index,
            parent_policy_name=parent.policy_id,
        )[: max(0, int(getattr(self.args, "max_candidates", 3)))]
        patch_by_id = {patch.patch_id: patch for patch in patches}

        valid_patches: List[Tuple[
            str,
            InstrumentPatch,
            OMEGAPolicyCandidate,
            Dict[str, Any],
        ]] = []
        invalid_patch_count = 0
        validation_results: List[RecursiveValidationResult] = []
        candidate_versions: List[RecursivePolicyVersion] = []
        diagnostic_records: List[RecursiveDiagnosticRecord] = []
        for index, patch in enumerate(patches, start=1):
            candidate_policy_id = f"{parent.policy_id}->C{cycle_index}.{index}"
            patch_metadata = self._patch_metadata(patch, parent, source_residues)
            try:
                self.contract.validate_patch(patch)
            except ValueError as exc:
                invalid_patch_count += 1
                result = RecursiveValidationResult(
                    candidate_policy_id=candidate_policy_id,
                    candidate_policy_name=patch.candidate_name,
                    parent_policy_id=parent.policy_id,
                    comparison_parent_policy_id=parent.policy_id,
                    source_residue_ids=list(patch.source_residue_ids),
                    patch_id=patch.patch_id,
                    contract_validated=False,
                    paired_evaluated=False,
                    holdout_evaluated=False,
                    rejection_reasons=[f"contract validation failed: {exc}"],
                    invalid_patch_penalty=1.0,
                    patch_family=patch_metadata["patch_family"],
                    patch_fields_changed=patch_metadata["patch_fields_changed"],
                    patch_equivalence_group=patch_metadata["patch_equivalence_group"],
                    patch_novel=patch_metadata["patch_novel"],
                    equivalent_failed_patch_ids=(
                        patch_metadata["equivalent_failed_patch_ids"]
                    ),
                )
                self._finalize_rejected_result(
                    result=result,
                    parent=parent,
                    cycle_index=cycle_index,
                    source_residues=source_residues,
                    patch=patch,
                    diagnostic_records=diagnostic_records,
                )
                validation_results.append(result)
                candidate_versions.append(self._rejected_policy_version(
                    candidate_policy_id,
                    patch.candidate_name,
                    parent,
                    patch,
                    cycle_index,
                    result,
                ))
                continue
            candidate = apply_instrument_patch(
                parent_name=parent.policy_id,
                parent_config=parent.candidate.config,
                parent_generated_min_evaluations=parent.generated_min_evaluations,
                patch=patch,
                contract=self.contract,
            )
            valid_patches.append((candidate_policy_id, patch, candidate, patch_metadata))

        total_patches = len(patches)
        valid_patch_rate = len(valid_patches) / total_patches if total_patches else 0.0
        residue_conversion = len(valid_patches) / len(residue_ids) if residue_ids else 0.0
        residue_conversion = min(1.0, residue_conversion)

        eligible_results: List[
            Tuple[float, RecursiveValidationResult, OMEGAPolicyCandidate, InstrumentPatch]
        ] = []
        for candidate_policy_id, patch, candidate, patch_metadata in valid_patches:
            paired_summary, paired_rows, paired_residues = self.evaluate_policy_version(
                RecursivePolicyVersion(
                    policy_id=candidate_policy_id,
                    policy_name=candidate.name,
                    parent_policy_id=parent.policy_id,
                    problem_space_version_id=None,
                    source_residue_ids=list(patch.source_residue_ids),
                    source_patch_id=patch.patch_id,
                    eie_config=asdict(candidate.config),
                    generated_min_evaluations=candidate.generated_min_evaluations,
                    creation_cycle=cycle_index,
                    validation_summary={},
                    accepted=False,
                    rejection_reasons=[],
                    lineage_depth=parent.lineage_depth + 1,
                    candidate=candidate,
                ),
                self.paired_seeds,
                comparison_rows=parent_paired_rows,
            )
            paired_score, paired_penalty, paired_breakdown = (
                self.recursive_improvement_score(
                paired_summary,
                valid_patch_rate=valid_patch_rate,
                residue_to_patch_conversion_success=residue_conversion,
                invalid_patch_count=invalid_patch_count,
                holdout_summary=None,
                novelty_penalty=patch_metadata["novelty_penalty"],
                with_breakdown=True,
            )
            )
            result = RecursiveValidationResult(
                candidate_policy_id=candidate_policy_id,
                candidate_policy_name=candidate.name,
                parent_policy_id=parent.policy_id,
                comparison_parent_policy_id=parent.policy_id,
                source_residue_ids=list(patch.source_residue_ids),
                patch_id=patch.patch_id,
                contract_validated=True,
                paired_evaluated=True,
                holdout_evaluated=False,
                paired_summary=paired_summary,
                paired_rows=paired_rows,
                paired_residues=paired_residues,
                recursive_score=paired_score,
                valid_patch_rate=valid_patch_rate,
                residue_to_patch_conversion_success=residue_conversion,
                invalid_patch_penalty=invalid_patch_count * 0.05,
                regression_penalty=paired_penalty,
                novelty_penalty=patch_metadata["novelty_penalty"],
                patch_family=patch_metadata["patch_family"],
                patch_fields_changed=patch_metadata["patch_fields_changed"],
                patch_equivalence_group=patch_metadata["patch_equivalence_group"],
                patch_novel=patch_metadata["patch_novel"],
                equivalent_failed_patch_ids=(
                    patch_metadata["equivalent_failed_patch_ids"]
                ),
                score_breakdown=paired_breakdown,
            )

            paired_reasons = self._paired_gate_failures(result)
            if not paired_reasons:
                holdout_summary, holdout_rows, holdout_residues = (
                    self.evaluate_policy_version(
                        RecursivePolicyVersion(
                            policy_id=candidate_policy_id,
                            policy_name=candidate.name,
                            parent_policy_id=parent.policy_id,
                            problem_space_version_id=None,
                            source_residue_ids=list(patch.source_residue_ids),
                            source_patch_id=patch.patch_id,
                            eie_config=asdict(candidate.config),
                            generated_min_evaluations=(
                                candidate.generated_min_evaluations
                            ),
                            creation_cycle=cycle_index,
                            validation_summary={},
                            accepted=False,
                            rejection_reasons=[],
                            lineage_depth=parent.lineage_depth + 1,
                            candidate=candidate,
                        ),
                        self.holdout_seeds,
                        comparison_rows=parent_holdout_rows,
                    )
                )
                holdout_score, holdout_penalty, holdout_breakdown = (
                    self.recursive_improvement_score(
                    paired_summary,
                    valid_patch_rate=valid_patch_rate,
                    residue_to_patch_conversion_success=residue_conversion,
                    invalid_patch_count=invalid_patch_count,
                    holdout_summary=holdout_summary,
                    novelty_penalty=patch_metadata["novelty_penalty"],
                    with_breakdown=True,
                )
                )
                result.holdout_evaluated = True
                result.holdout_summary = holdout_summary
                result.holdout_rows = holdout_rows
                result.holdout_residues = holdout_residues
                result.holdout_recursive_score = holdout_score
                result.recursive_score = holdout_score
                result.regression_penalty = holdout_penalty
                result.score_breakdown = holdout_breakdown
            else:
                result.rejection_reasons.extend(paired_reasons)

            acceptance_reasons = self._acceptance_failures(result)
            if acceptance_reasons:
                result.rejection_reasons.extend(
                    reason
                    for reason in acceptance_reasons
                    if reason not in result.rejection_reasons
                )
                self._finalize_rejected_result(
                    result=result,
                    parent=parent,
                    cycle_index=cycle_index,
                    source_residues=source_residues,
                    patch=patch,
                    diagnostic_records=diagnostic_records,
                )
            else:
                eligible_results.append((
                    result.recursive_score,
                    result,
                    candidate,
                    patch,
                ))
            validation_results.append(result)
            candidate_versions.append(self._rejected_policy_version(
                candidate_policy_id,
                candidate.name,
                parent,
                patch,
                cycle_index,
                result,
                candidate,
            ))

        accepted_policy_id = None
        accepted_patch_id = None
        if eligible_results:
            eligible_results.sort(key=lambda item: item[0], reverse=True)
            _, accepted_result, accepted_candidate, accepted_patch = eligible_results[0]
            accepted_policy_id = f"P{self._next_policy_number}"
            self._next_policy_number += 1
            accepted_result.accepted = True
            accepted_version = self._accepted_policy_version(
                accepted_policy_id,
                parent,
                accepted_candidate,
                accepted_patch,
                cycle_index,
                accepted_result,
            )
            accepted_result.problem_space_version_id = (
                accepted_version.problem_space_version_id
            )
            accepted_version.validation_summary = accepted_result.to_json_dict()
            accepted_patch_id = accepted_patch.patch_id
            candidate_versions = [
                accepted_version
                if policy.source_patch_id == accepted_patch.patch_id
                else self._mark_not_selected(policy)
                for policy in candidate_versions
            ]
            for result in validation_results:
                if result.patch_id != accepted_patch.patch_id and not result.rejection_reasons:
                    result.rejection_reasons.append(
                        "candidate was valid but not selected as best recursive score"
                    )
                    patch = patch_by_id.get(result.patch_id)
                    if patch is not None:
                        self._finalize_rejected_result(
                            result=result,
                            parent=parent,
                            cycle_index=cycle_index,
                            source_residues=source_residues,
                            patch=patch,
                            diagnostic_records=diagnostic_records,
                        )

        rejected_groups = self._rejected_patch_equivalence_groups(validation_results)
        suggested_next_bottleneck = self._suggest_next_bottleneck(
            source_residue_distribution,
            diagnostic_records,
        )

        return RecursiveImprovementCycle(
            cycle_index=cycle_index,
            parent_policy_id=parent.policy_id,
            parent_problem_space_version_id=parent.problem_space_version_id,
            parent_paired_summary=parent_paired_summary,
            parent_holdout_summary=parent_holdout_summary,
            parent_paired_rows=parent_paired_rows,
            parent_holdout_rows=parent_holdout_rows,
            source_residue_ids=residue_ids,
            source_residues=source_residues,
            patch_ids=[patch.patch_id for patch in patches],
            candidate_policies=candidate_versions,
            validation_results=validation_results,
            diagnostic_records=diagnostic_records,
            source_residue_distribution=source_residue_distribution,
            candidate_patch_families=[
                self._patch_family(patch) for patch in patches
            ],
            rejected_patch_equivalence_groups=rejected_groups,
            suggested_next_bottleneck=suggested_next_bottleneck,
            accepted_policy_id=accepted_policy_id,
            accepted_patch_id=accepted_patch_id,
            next_parent_policy_id=accepted_policy_id or parent.policy_id,
        )

    def evaluate_policy_version(
        self,
        policy: RecursivePolicyVersion,
        seeds: Sequence[int],
        comparison_rows: Optional[List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        residues: List[Dict[str, Any]] = []
        for seed in seeds:
            row, exported = self.run_policy_case(policy, seed)
            rows.append(row)
            residues.extend(exported)
        if comparison_rows is None:
            summary = self.summarize_self(policy.policy_id, rows)
        else:
            summary = self.summarize_relative(policy.policy_id, comparison_rows, rows)
        return summary, rows, residues

    def run_policy_case(
        self,
        policy: RecursivePolicyVersion,
        seed: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        _seed_everything(seed)
        candidate = policy.candidate
        engine = build_rsi_nas(
            d_model=self.args.d_model,
            train_steps=self.args.train_steps,
            expansion_interval=self.args.expansion_interval,
            pruning_interval=self.args.pruning_interval,
            generated_min_evaluations=policy.generated_min_evaluations,
            enable_eie=True,
            eie_config=candidate.config,
        )
        history = engine.run(
            generations=self.args.generations,
            population_size=self.args.population_size,
        )
        final = history[-1]
        generated_records = engine.registry.all_generated_records()
        residues = [
            {
                **residue.to_json_dict(),
                "recursive_policy_id": policy.policy_id,
                "recursive_parent_policy_id": policy.parent_policy_id,
            }
            for residue in export_residues(
                engine,
                seed=seed,
                condition=policy.policy_id,
            )
        ]
        row = {
            "seed": seed,
            "policy_id": policy.policy_id,
            "policy_name": policy.policy_name,
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
            "problem_space_version": final.get("problem_space_version"),
            "failure_residue_ledger_entries": final.get(
                "failure_residue_ledger_entries"
            ),
        }
        return row, residues

    def summarize_self(self, policy_id: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        bpcs = [row["best_bpc"] for row in rows]
        summary = self._activity_summary(policy_id, rows)
        summary.update({
            "mean_best_bpc": round(mean(bpcs), 4),
            "parent_mean_best_bpc": round(mean(bpcs), 4),
            "mean_bpc_delta": 0.0,
            "eie_wins": 0,
            "comparison": "self",
        })
        return summary

    def summarize_relative(
        self,
        policy_id: str,
        parent_rows: List[Dict[str, Any]],
        candidate_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        parent_bpc = [row["best_bpc"] for row in parent_rows]
        candidate_bpc = [row["best_bpc"] for row in candidate_rows]
        summary = self._activity_summary(policy_id, candidate_rows)
        summary.update({
            "mean_best_bpc": round(mean(candidate_bpc), 4),
            "parent_mean_best_bpc": round(mean(parent_bpc), 4),
            "mean_bpc_delta": round(mean(parent_bpc) - mean(candidate_bpc), 4),
            "eie_wins": sum(
                1 for parent, candidate in zip(parent_bpc, candidate_bpc)
                if candidate < parent
            ),
            "comparison": "parent",
        })
        return summary

    def recursive_improvement_score(
        self,
        paired_summary: Dict[str, Any],
        valid_patch_rate: float,
        residue_to_patch_conversion_success: float,
        invalid_patch_count: int,
        holdout_summary: Optional[Dict[str, Any]],
        novelty_penalty: float = 0.0,
        with_breakdown: bool = False,
    ) -> Tuple[float, float] | Tuple[float, float, Dict[str, Any]]:
        if not paired_summary.get("counter_integrity_verified", False):
            raise ValueError("paired summary counters are not tied to generated records")
        if holdout_summary is not None and not holdout_summary.get(
            "counter_integrity_verified",
            False,
        ):
            raise ValueError("holdout summary counters are not tied to generated records")
        bpc_delta = float(paired_summary["mean_bpc_delta"])
        regression_penalty = max(0.0, -bpc_delta) * 2.0
        invalid_penalty = invalid_patch_count * 0.05
        mechanism_activity = (
            0.0005 * paired_summary["total_generated_evaluations"]
            + 0.0030 * paired_summary["total_generated_archive_insertions"]
            + 0.0004 * paired_summary["total_meta_operator_policy_updates"]
        )
        conversion_score = (
            0.05 * residue_to_patch_conversion_success
            + 0.05 * valid_patch_rate
        )
        score = (
            bpc_delta
            + mechanism_activity
            + conversion_score
            - invalid_penalty
            - regression_penalty
            - novelty_penalty
        )
        holdout_delta = None
        holdout_robustness = 0.0
        holdout_regression_penalty = 0.0
        holdout_delta_clamped = 0.0
        if holdout_summary is not None:
            holdout_delta = float(holdout_summary["mean_bpc_delta"])
            holdout_regression_penalty = max(0.0, -holdout_delta) * 2.0
            holdout_robustness = 1.0 if (
                holdout_delta >= 0.0 and _positive_mechanism(holdout_summary)
            ) else 0.0
            holdout_delta_clamped = max(-0.05, min(0.05, holdout_delta))
            score += (
                0.05 * holdout_robustness
                + holdout_delta_clamped
                - holdout_regression_penalty
            )
            regression_penalty += holdout_regression_penalty
        rounded_score = round(score, 6)
        rounded_penalty = round(regression_penalty, 6)
        breakdown = {
            "mean_bpc_delta": bpc_delta,
            "mechanism_activity_score": round(mechanism_activity, 6),
            "residue_to_patch_conversion_success": (
                residue_to_patch_conversion_success
            ),
            "valid_patch_rate": valid_patch_rate,
            "conversion_score": round(conversion_score, 6),
            "invalid_patch_penalty": round(invalid_penalty, 6),
            "regression_penalty": rounded_penalty,
            "novelty_penalty": round(novelty_penalty, 6),
            "holdout_delta": holdout_delta,
            "holdout_robustness": holdout_robustness,
            "holdout_delta_clamped": round(holdout_delta_clamped, 6),
            "total_score": rounded_score,
            "counter_integrity_verified": True,
        }
        if with_breakdown:
            return rounded_score, rounded_penalty, breakdown
        return rounded_score, rounded_penalty

    def _activity_summary(
        self,
        policy_id: str,
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        integrity_violations = self._counter_integrity_violations(rows)
        return {
            "name": policy_id,
            "paired_or_holdout_seeds": len(rows),
            "total_generated_archive_insertions": sum(
                row["generated_archive_insertions"] for row in rows
            ),
            "total_generated_evaluations": sum(
                row["generated_evaluations"] for row in rows
            ),
            "total_meta_operator_policy_updates": sum(
                row["meta_operator_policy_updates"] for row in rows
            ),
            "total_eie_residues": sum(row["eie_residues"] for row in rows),
            "total_instrument_mutations": sum(
                row["instrument_mutations"] for row in rows
            ),
            "total_instrumented_candidates": sum(
                row["instrumented_candidates"] for row in rows
            ),
            "counter_integrity_verified": not integrity_violations,
            "counter_integrity_violations": integrity_violations,
        }

    def _counter_integrity_violations(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        for row in rows:
            record_eval_sum = sum(
                int(record.get("evaluations", 0))
                for record in row.get("generated_records", [])
            )
            record_archive_sum = sum(
                int(record.get("archive_insertions", 0))
                for record in row.get("generated_records", [])
            )
            if int(row.get("generated_evaluations", 0)) != record_eval_sum:
                violations.append({
                    "seed": row.get("seed"),
                    "field": "generated_evaluations",
                    "row_value": row.get("generated_evaluations"),
                    "record_sum": record_eval_sum,
                })
            if int(row.get("generated_archive_insertions", 0)) != record_archive_sum:
                violations.append({
                    "seed": row.get("seed"),
                    "field": "generated_archive_insertions",
                    "row_value": row.get("generated_archive_insertions"),
                    "record_sum": record_archive_sum,
                })
        return violations

    def _augment_parent_residues(
        self,
        cycle_index: int,
        parent: RecursivePolicyVersion,
        parent_paired_summary: Dict[str, Any],
        parent_paired_rows: List[Dict[str, Any]],
        source_residues: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if cycle_index < 2:
            return list(source_residues)
        augmented = list(source_residues)
        existing_ids = {residue["residue_id"] for residue in augmented}

        def append_once(residue: Dict[str, Any]) -> None:
            if residue["residue_id"] not in existing_ids:
                augmented.append(residue)
                existing_ids.add(residue["residue_id"])

        archive_starvation = any(
            "archive_starvation_after_evaluation"
            in residue.get("observed_failure_signature", [])
            for residue in source_residues
        )
        total_evaluations = int(parent_paired_summary["total_generated_evaluations"])
        total_archive = int(parent_paired_summary["total_generated_archive_insertions"])
        if total_evaluations > 0 and (total_archive == 0 or archive_starvation):
            append_once(self._make_execution_diagnostic_residue(
                residue_type="ARCHIVE_STAGNATION",
                parent=parent,
                cycle_index=cycle_index,
                parent_paired_summary=parent_paired_summary,
                parent_paired_rows=parent_paired_rows,
                source_residue_ids=[r["residue_id"] for r in source_residues],
                signature="generated_evaluations_without_archive_insertions",
                requested=[
                    "archive_insertion_priority",
                    "archive_priority_terms",
                    "meta_archive_weighting",
                ],
            ))

        max_weight = 0.0
        min_weight = 1.0
        for row in parent_paired_rows:
            weights = row.get("meta_operator_weights", {}) or {}
            if weights:
                values = [float(value) for value in weights.values()]
                max_weight = max(max_weight, max(values))
                min_weight = min(min_weight, min(values))
        if total_evaluations > 0 and max_weight >= 0.80 and min_weight <= 0.05:
            append_once(self._make_execution_diagnostic_residue(
                residue_type="META_OPERATOR_IMBALANCE",
                parent=parent,
                cycle_index=cycle_index,
                parent_paired_summary=parent_paired_summary,
                parent_paired_rows=parent_paired_rows,
                source_residue_ids=[r["residue_id"] for r in source_residues],
                signature="meta_operator_weight_collapse",
                requested=[
                    "meta_operator_weighting_coefficients",
                    "meta_exploration_floor",
                ],
                extra_payload={
                    "max_operator_weight": max_weight,
                    "min_operator_weight": min_weight,
                },
            ))

        residue_types = {residue.get("residue_type") for residue in source_residues}
        if (
            cycle_index >= 2
            and total_evaluations > 0
            and total_archive > 0
            and residue_types.issubset({"PRUNING_PROPAGATION_RACE"})
        ):
            append_once(self._make_execution_diagnostic_residue(
                residue_type="POLICY_SATURATION",
                parent=parent,
                cycle_index=cycle_index,
                parent_paired_summary=parent_paired_summary,
                parent_paired_rows=parent_paired_rows,
                source_residue_ids=[r["residue_id"] for r in source_residues],
                signature="pruning_race_no_longer_sufficient_bottleneck",
                requested=[
                    "evaluator_terms",
                    "archive_priority_terms",
                    "candidate_scoring_coefficients",
                ],
            ))

        return augmented

    def _make_execution_diagnostic_residue(
        self,
        residue_type: str,
        parent: RecursivePolicyVersion,
        cycle_index: int,
        parent_paired_summary: Dict[str, Any],
        parent_paired_rows: List[Dict[str, Any]],
        source_residue_ids: List[str],
        signature: str,
        requested: List[str],
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "parent_policy_id": parent.policy_id,
            "cycle_index": cycle_index,
            "residue_type": residue_type,
            "source_residue_ids": source_residue_ids,
            "summary": parent_paired_summary,
            "signature": signature,
            "extra": extra_payload or {},
        }
        residue_id = _stable_diagnostic_id("recursive-residue", payload)
        first_row = parent_paired_rows[0] if parent_paired_rows else {}
        return {
            "residue_id": residue_id,
            "generation": int(self.args.generations),
            "seed": int(first_row.get("seed", -1)),
            "condition": parent.policy_id,
            "recursive_policy_id": parent.policy_id,
            "recursive_parent_policy_id": parent.parent_policy_id,
            "residue_type": residue_type,
            "generated_module_name": f"{parent.policy_id}:{residue_type}",
            "source_action": "recursive_diagnostic",
            "birth_generation": 0,
            "evaluations": int(parent_paired_summary["total_generated_evaluations"]),
            "archive_insertions": int(
                parent_paired_summary["total_generated_archive_insertions"]
            ),
            "elite_uses": 0,
            "best_fitness": 0.0,
            "prune_attempts": 0,
            "protected_from_prune": 0,
            "age_at_prune_attempt": 0,
            "current_policy_snapshot": {
                "recursive_policy_id": parent.policy_id,
                "eie_config": dict(parent.eie_config),
                "generated_min_evaluations": parent.generated_min_evaluations,
            },
            "observed_failure_signature": [residue_type, signature],
            "requested_instrument_classes": requested,
            "subject_type": "recursive_policy_execution",
            "parent_policy_id": parent.policy_id,
            "candidate_policy_id": None,
            "cycle_index": cycle_index,
            "patch_id": None,
            "patch_family": PATCH_FAMILY_BY_RESIDUE_TYPE.get(residue_type),
            "suspected_bottleneck_category": residue_type,
            "diagnostic_payload": payload,
            "structured_rejection_reasons": [],
        }

    def _structured_residue_from_dict(
        self,
        residue: Dict[str, Any],
    ) -> StructuredAFIRSIResidue:
        valid_fields = {field_info.name for field_info in fields(StructuredAFIRSIResidue)}
        return StructuredAFIRSIResidue(**{
            key: value
            for key, value in residue.items()
            if key in valid_fields
        })

    def _patch_metadata(
        self,
        patch: InstrumentPatch,
        parent: RecursivePolicyVersion,
        source_residues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        patch_family = self._patch_family(patch)
        equivalence_group = self._patch_equivalence_key(
            patch,
            parent,
            source_residues,
        )
        equivalent_failed = [
            row["patch_id"]
            for row in self.failed_patch_memory
            if row["equivalence_group"] == equivalence_group
        ]
        novelty_penalty = min(0.30, 0.10 * len(equivalent_failed))
        return {
            "patch_family": patch_family,
            "patch_fields_changed": sorted(patch.target_updates),
            "patch_equivalence_group": equivalence_group,
            "patch_novel": not equivalent_failed,
            "equivalent_failed_patch_ids": equivalent_failed,
            "novelty_penalty": novelty_penalty,
        }

    def _patch_family(self, patch: InstrumentPatch) -> str:
        provenance = getattr(patch, "provenance", {}) or {}
        if provenance.get("patch_family"):
            return str(provenance["patch_family"])
        omega_style = provenance.get("omega_style", {})
        if omega_style.get("constraint_interpretation"):
            return str(omega_style["constraint_interpretation"])
        name = getattr(patch, "candidate_name", "") or getattr(patch, "patch_id", "")
        return name.rsplit("_", 1)[0] if "_" in name else name

    def _patch_equivalence_key(
        self,
        patch: InstrumentPatch,
        parent: RecursivePolicyVersion,
        source_residues: List[Dict[str, Any]],
    ) -> str:
        source_residue_types = sorted({
            residue.get("residue_type", "UNKNOWN")
            for residue in source_residues
            if residue.get("residue_id") in set(patch.source_residue_ids)
        })
        directions = {}
        for field_name, value in patch.target_updates.items():
            parent_value = self._parent_policy_value(parent, field_name)
            directions[field_name] = self._update_direction(parent_value, value)
        payload = {
            "target_fields": sorted(patch.target_updates),
            "directions": directions,
            "scaffold_strategy": patch.generated_module_scaffold_strategy,
            "evaluator_terms": sorted(patch.evaluator_terms),
            "archive_insertion_priority": patch.archive_insertion_priority,
            "archive_priority_terms": patch.target_updates.get(
                "archive_priority_terms"
            ),
            "parent_residue_types": source_residue_types,
            "patch_family": self._patch_family(patch),
        }
        digest = hashlib.blake2b(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
            digest_size=8,
        ).hexdigest()
        return f"patch-equivalence-{digest}"

    def _parent_policy_value(
        self,
        parent: RecursivePolicyVersion,
        field_name: str,
    ) -> Any:
        if field_name == "generated_min_evaluations":
            return parent.generated_min_evaluations
        if field_name in parent.eie_config:
            return parent.eie_config[field_name]
        return None

    def _update_direction(self, parent_value: Any, value: Any) -> str:
        if isinstance(value, bool):
            return f"set_bool_{value}"
        if isinstance(value, (int, float)) and isinstance(parent_value, (int, float)):
            if float(value) > float(parent_value):
                return "increase"
            if float(value) < float(parent_value):
                return "decrease"
            return "same"
        if isinstance(value, list):
            return "set_terms:" + ",".join(str(item) for item in value)
        return f"set:{value}"

    def _paired_gate_failures(
        self,
        result: RecursiveValidationResult,
    ) -> List[str]:
        failures: List[str] = []
        if not result.contract_validated:
            failures.append("patch was not contract validated")
        if not result.paired_evaluated:
            failures.append("paired-seed validation was not run")
        if not result.source_residue_ids:
            failures.append("candidate has no real source residues")
        if not result.patch_id:
            failures.append("candidate has no source patch")
        if result.paired_summary and not result.paired_summary.get(
            "counter_integrity_verified",
            False,
        ):
            failures.append("paired counters are not verified against engine records")
        if not result.patch_novel and result.novelty_penalty >= 0.20:
            failures.append("patch repeats a recently failed equivalent signature")
        if not _positive_mechanism(result.paired_summary):
            failures.append("paired validation lacks positive mechanism activity")
        if result.recursive_score <= 0.0:
            failures.append("paired recursive improvement score does not beat parent")
        return failures

    def _acceptance_failures(
        self,
        result: RecursiveValidationResult,
    ) -> List[str]:
        failures = self._paired_gate_failures(result)
        if not result.holdout_evaluated:
            failures.append("holdout validation was not run")
        if result.holdout_summary and not result.holdout_summary.get(
            "counter_integrity_verified",
            False,
        ):
            failures.append("holdout counters are not verified against engine records")
        if not _positive_mechanism(result.holdout_summary):
            failures.append("holdout validation lacks positive mechanism activity")
        if result.holdout_summary and result.holdout_summary["mean_bpc_delta"] < 0.0:
            failures.append("holdout BPC regressed against parent")
        if result.holdout_recursive_score <= 0.0:
            failures.append("holdout recursive improvement score is not robust")
        return failures

    def _finalize_rejected_result(
        self,
        result: RecursiveValidationResult,
        parent: RecursivePolicyVersion,
        cycle_index: int,
        source_residues: List[Dict[str, Any]],
        patch: InstrumentPatch,
        diagnostic_records: List[RecursiveDiagnosticRecord],
    ) -> None:
        structured_reasons = [
            self._structured_reason(reason)
            for reason in result.rejection_reasons
        ]
        result.structured_rejection_reasons = structured_reasons
        categories = self._diagnostic_categories_for_rejection(result, patch)
        if not categories:
            categories = ["PATCH_EFFECTIVENESS_FAILURE"]
        result.suspected_bottleneck_category = categories[0]

        source_ids = [residue["residue_id"] for residue in source_residues]
        for category in categories:
            diagnostic = self._make_rejection_diagnostic_record(
                category=category,
                result=result,
                parent=parent,
                cycle_index=cycle_index,
                source_residue_ids=source_ids,
            )
            diagnostic_records.append(diagnostic)

        if result.patch_id:
            self.failed_patch_memory.append({
                "patch_id": result.patch_id,
                "patch_family": result.patch_family,
                "equivalence_group": result.patch_equivalence_group,
                "parent_policy_id": parent.policy_id,
                "cycle_index": cycle_index,
                "residue_types": sorted({
                    residue.get("residue_type", "UNKNOWN")
                    for residue in source_residues
                }),
                "rejection_reasons": structured_reasons,
            })

    def _structured_reason(self, reason: str) -> Dict[str, str]:
        code = (
            reason.lower()
            .replace(":", "")
            .replace("-", " ")
            .replace("/", " ")
            .replace(" ", "_")
        )
        return {"code": code[:96], "message": reason}

    def _diagnostic_categories_for_rejection(
        self,
        result: RecursiveValidationResult,
        patch: InstrumentPatch,
    ) -> List[str]:
        categories: List[str] = []
        paired = result.paired_summary or {}
        holdout = result.holdout_summary or {}
        paired_delta = float(paired.get("mean_bpc_delta", 0.0))
        holdout_delta = float(holdout.get("mean_bpc_delta", 0.0))

        if not result.contract_validated:
            return ["OPERATOR_GENERATOR_MODE_COLLAPSE"]
        if not result.patch_novel:
            categories.append("OPERATOR_GENERATOR_MODE_COLLAPSE")
        if (
            result.paired_evaluated
            and paired_delta > 0.0
            and (
                not result.holdout_evaluated
                or result.holdout_recursive_score <= 0.0
                or holdout_delta < 0.0
            )
        ):
            categories.append("EVALUATOR_NOISE_OR_OVERFIT")
        if (
            int(paired.get("total_generated_evaluations", 0)) > 0
            and int(paired.get("total_generated_archive_insertions", 0)) == 0
        ):
            categories.append("ARCHIVE_STAGNATION")
        if (
            "clean_probe" in str(patch.generated_module_scaffold_strategy)
            and int(paired.get("total_generated_evaluations", 0)) > 0
            and paired_delta <= 0.0
        ):
            categories.append("SCAFFOLD_BIAS")
        if (
            result.contract_validated
            and result.paired_evaluated
            and (
                result.recursive_score <= 0.0
                or paired_delta <= 0.0
                or not _positive_mechanism(paired)
            )
        ):
            categories.append("PATCH_EFFECTIVENESS_FAILURE")
        if any(
            "operator" in reason.get("code", "")
            for reason in result.structured_rejection_reasons
        ):
            categories.append("META_OPERATOR_IMBALANCE")
        return list(dict.fromkeys(categories))

    def _make_rejection_diagnostic_record(
        self,
        category: str,
        result: RecursiveValidationResult,
        parent: RecursivePolicyVersion,
        cycle_index: int,
        source_residue_ids: List[str],
    ) -> RecursiveDiagnosticRecord:
        paired = result.paired_summary or {}
        holdout = result.holdout_summary or {}
        payload = {
            "category": category,
            "parent_policy_id": parent.policy_id,
            "candidate_policy_id": result.candidate_policy_id,
            "cycle_index": cycle_index,
            "patch_id": result.patch_id,
            "rejection_reasons": result.structured_rejection_reasons,
            "score": result.recursive_score,
            "holdout_score": result.holdout_recursive_score,
            "patch_equivalence_group": result.patch_equivalence_group,
        }
        diagnostic_id = _stable_diagnostic_id("recursive-rejection", payload)
        return RecursiveDiagnosticRecord(
            diagnostic_id=diagnostic_id,
            residue_type=category,
            parent_policy_id=parent.policy_id,
            candidate_policy_id=result.candidate_policy_id,
            cycle_index=cycle_index,
            paired_validation_result=paired,
            holdout_validation_result=holdout if holdout else None,
            score_delta_against_parent=float(result.recursive_score),
            bpc_delta_against_parent=float(paired.get("mean_bpc_delta", 0.0)),
            generated_evaluations=int(
                paired.get("total_generated_evaluations", 0)
            ),
            archive_insertions=int(
                paired.get("total_generated_archive_insertions", 0)
            ),
            meta_policy_updates=int(
                paired.get("total_meta_operator_policy_updates", 0)
            ),
            residues_produced_under_parent=list(source_residue_ids),
            patch_id=result.patch_id,
            patch_fields_changed=list(result.patch_fields_changed),
            patch_family=result.patch_family,
            rejection_reasons=[
                dict(reason) for reason in result.structured_rejection_reasons
            ],
            suspected_bottleneck_category=category,
            patch_equivalence_group=result.patch_equivalence_group,
            equivalent_failed_patch_ids=list(result.equivalent_failed_patch_ids),
        )

    def _rejected_patch_equivalence_groups(
        self,
        validation_results: List[RecursiveValidationResult],
    ) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = defaultdict(list)
        for result in validation_results:
            if result.accepted or not result.rejection_reasons:
                continue
            if result.patch_equivalence_group:
                groups[result.patch_equivalence_group].append(result.patch_id or "")
        return {key: value for key, value in groups.items() if key}

    def _suggest_next_bottleneck(
        self,
        source_residue_distribution: Dict[str, int],
        diagnostic_records: List[RecursiveDiagnosticRecord],
    ) -> Optional[str]:
        counter: Counter[str] = Counter()
        counter.update({
            key: value
            for key, value in source_residue_distribution.items()
            if key != "PRUNING_PROPAGATION_RACE"
        })
        counter.update(record.residue_type for record in diagnostic_records)
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def _cycle_failure_diagnosis(
        self,
        cycle: RecursiveImprovementCycle,
    ) -> Dict[str, Any]:
        rejected = [
            result for result in cycle.validation_results
            if not result.accepted
        ]
        return {
            "cycle_index": cycle.cycle_index,
            "parent_policy_id": cycle.parent_policy_id,
            "accepted_policy_id": cycle.accepted_policy_id,
            "p2_failed": cycle.accepted_policy_id is None,
            "source_residue_distribution": dict(cycle.source_residue_distribution),
            "candidate_patch_families": list(cycle.candidate_patch_families),
            "rejected_candidate_count": len(rejected),
            "diagnostic_residue_types": [
                record.residue_type for record in cycle.diagnostic_records
            ],
            "structured_rejections": [
                {
                    "candidate_policy_id": result.candidate_policy_id,
                    "patch_id": result.patch_id,
                    "patch_family": result.patch_family,
                    "score_delta_against_parent": result.recursive_score,
                    "bpc_delta_against_parent": (
                        result.paired_summary or {}
                    ).get("mean_bpc_delta"),
                    "holdout_bpc_delta_against_parent": (
                        result.holdout_summary or {}
                    ).get("mean_bpc_delta"),
                    "generated_evaluations": (
                        result.paired_summary or {}
                    ).get("total_generated_evaluations"),
                    "archive_insertions": (
                        result.paired_summary or {}
                    ).get("total_generated_archive_insertions"),
                    "meta_policy_updates": (
                        result.paired_summary or {}
                    ).get("total_meta_operator_policy_updates"),
                    "patch_fields_changed": list(result.patch_fields_changed),
                    "patch_equivalence_group": result.patch_equivalence_group,
                    "patch_novel": result.patch_novel,
                    "equivalent_failed_patch_ids": list(
                        result.equivalent_failed_patch_ids
                    ),
                    "suspected_bottleneck_category": (
                        result.suspected_bottleneck_category
                    ),
                    "rejection_reasons": [
                        dict(reason)
                        for reason in result.structured_rejection_reasons
                    ],
                }
                for result in rejected
            ],
            "rejected_patch_equivalence_groups": dict(
                cycle.rejected_patch_equivalence_groups
            ),
            "suggested_next_bottleneck": cycle.suggested_next_bottleneck,
        }

    def _accepted_policy_version(
        self,
        policy_id: str,
        parent: RecursivePolicyVersion,
        candidate: OMEGAPolicyCandidate,
        patch: InstrumentPatch,
        cycle_index: int,
        result: RecursiveValidationResult,
    ) -> RecursivePolicyVersion:
        version = self.problem_space_graph.create_child(
            parent_version_id=parent.problem_space_version_id,
            patch_id=patch.patch_id,
            residue_ids=list(patch.source_residue_ids),
            observation_channel_version="bounded_rsi.observation.reuse",
            evaluator_version="bounded_rsi.recursive_score.v1",
            instrument_policy=self._instrument_policy_dict(candidate),
        )
        return RecursivePolicyVersion(
            policy_id=policy_id,
            policy_name=candidate.name,
            parent_policy_id=parent.policy_id,
            problem_space_version_id=version.version_id,
            source_residue_ids=list(patch.source_residue_ids),
            source_patch_id=patch.patch_id,
            eie_config=asdict(candidate.config),
            generated_min_evaluations=candidate.generated_min_evaluations,
            creation_cycle=cycle_index,
            validation_summary=result.to_json_dict(),
            accepted=True,
            rejection_reasons=[],
            lineage_depth=parent.lineage_depth + 1,
            candidate=candidate,
        )

    def _rejected_policy_version(
        self,
        policy_id: str,
        policy_name: str,
        parent: RecursivePolicyVersion,
        patch: InstrumentPatch,
        cycle_index: int,
        result: RecursiveValidationResult,
        candidate: Optional[OMEGAPolicyCandidate] = None,
    ) -> RecursivePolicyVersion:
        eie_config = (
            asdict(candidate.config)
            if candidate is not None
            else asdict(parent.candidate.config)
        )
        generated_min = (
            candidate.generated_min_evaluations
            if candidate is not None
            else parent.generated_min_evaluations
        )
        return RecursivePolicyVersion(
            policy_id=policy_id,
            policy_name=policy_name,
            parent_policy_id=parent.policy_id,
            problem_space_version_id=None,
            source_residue_ids=list(patch.source_residue_ids),
            source_patch_id=patch.patch_id,
            eie_config=eie_config,
            generated_min_evaluations=generated_min,
            creation_cycle=cycle_index,
            validation_summary=result.to_json_dict(),
            accepted=False,
            rejection_reasons=list(result.rejection_reasons),
            lineage_depth=parent.lineage_depth + 1,
            candidate=candidate,
        )

    def _mark_not_selected(
        self,
        policy: RecursivePolicyVersion,
    ) -> RecursivePolicyVersion:
        reasons = list(policy.rejection_reasons)
        if not reasons:
            reasons.append("candidate was valid but not selected as best recursive score")
        policy.rejection_reasons = reasons
        policy.validation_summary = {
            **policy.validation_summary,
            "accepted": False,
            "rejection_reasons": reasons,
        }
        return policy

    def _strip_recursive_keys(self, residue: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in residue.items()
            if key not in {"recursive_policy_id", "recursive_parent_policy_id"}
        }

    def _instrument_policy_dict(
        self,
        candidate: OMEGAPolicyCandidate,
    ) -> Dict[str, Any]:
        return {
            "eie_config": asdict(candidate.config),
            "generated_min_evaluations": candidate.generated_min_evaluations,
            "patch_id": candidate.patch.patch_id,
            "candidate_name": candidate.name,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--holdout-seeds", default="23,29")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population-size", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--expansion-interval", type=int, default=1)
    parser.add_argument("--pruning-interval", type=int, default=1)
    parser.add_argument("--generated-min-evaluations", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--report-path", default="bounded_rsi_report.json")
    parser.add_argument("--diagnose-p2-failure", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.verbose:
        logging.getLogger().setLevel(logging.WARNING)

    report = BoundedRSIKernel(args).run()
    payload = report.to_json_dict()
    Path(args.report_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps({
            "bounded_recursive_success": payload["bounded_recursive_success"],
            "partial_recursive_success": payload["partial_recursive_success"],
            "accepted_policy_count": payload["accepted_policy_count"],
            "final_parent_policy_id": payload["final_parent_policy_id"],
            "boundary": payload["boundary"],
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
