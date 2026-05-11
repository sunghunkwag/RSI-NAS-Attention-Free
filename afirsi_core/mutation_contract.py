"""Mutation contracts for residue-conditioned instrument patches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class InstrumentPatch:
    """A proposed instrument mutation. It does not apply itself."""

    patch_id: str
    source_residue_ids: List[str]
    target_updates: Dict[str, Any]
    parent_version_id: str
    candidate_name: str = ""
    evaluator_terms: List[str] = field(default_factory=list)
    archive_insertion_priority: Optional[str] = None
    generated_module_scaffold_strategy: Optional[str] = None
    rationale: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    validation_required: bool = True
    contract_version: str = "instrument_mutation_contract.v1"


@dataclass(frozen=True)
class InstrumentMutationContract:
    """Defines allowed and forbidden AFIRSI instrument mutations."""

    version: str = "instrument_mutation_contract.v1"
    mutable_fields: List[str] = field(default_factory=lambda: [
        "generated_grace_generations",
        "generated_min_evaluations",
        "generated_probe_rate",
        "grace_multiplier",
        "probe_rate",
        "clean_probe_first_eval",
        "evaluator_terms",
        "archive_priority_terms",
        "archive_insertion_priority",
        "generated_module_scaffold_strategy",
        "meta_eval_gain",
        "meta_archive_gain",
        "meta_fitness_gain",
        "meta_exploration_floor",
        "meta_operator_weighting_coefficients",
        "candidate_scoring_coefficients",
    ])
    immutable_requirements: List[str] = field(default_factory=lambda: [
        "actual_training_evaluation_required",
        "actual_archive_insertion_required",
        "no_fake_bpc_values",
        "no_fake_archive_insertions",
        "no_direct_success_flags",
        "no_bypass_of_build_train_evaluate",
        "no_residue_deletion_to_pass_validation",
        "no_policy_acceptance_without_validation",
        "no_seed_specific_outcomes",
    ])
    forbidden_fields: List[str] = field(default_factory=lambda: [
        "bpc",
        "best_bpc",
        "mean_bpc",
        "mean_bpc_delta",
        "fake_bpc",
        "archive_insertions",
        "generated_archive_insertions",
        "total_generated_archive_insertions",
        "evaluations",
        "generated_evaluations",
        "total_generated_evaluations",
        "mechanism_valid",
        "open_ended_proxy_valid",
        "omega_validation_valid",
        "success",
        "accepted",
        "direct_success_flag",
        "skip_training",
        "skip_build",
        "skip_evaluation",
        "bypass_build_train_evaluate",
        "delete_residue",
        "drop_residue",
        "clear_ledger",
        "residue_count",
        "hardcoded_seed",
        "seed_success",
    ])

    def validate_patch(self, patch: Any) -> None:
        updates = dict(getattr(patch, "target_updates", patch))
        allowed = set(self.mutable_fields)
        forbidden = set(self.forbidden_fields)

        illegal = sorted(set(updates) - allowed)
        if illegal:
            raise ValueError(
                f"Patch mutates immutable or unknown fields: {illegal}"
            )

        forbidden_hits = sorted(set(updates) & forbidden)
        key_fragment_hits = sorted(
            key for key in updates
            if any(fragment in key for fragment in ["fake", "bpc", "success"])
        )
        if forbidden_hits or key_fragment_hits:
            hits = sorted(set(forbidden_hits + key_fragment_hits))
            raise ValueError(f"Patch attempts forbidden instrument mutation: {hits}")

        if getattr(patch, "validation_required", True) is False:
            raise ValueError("Patch attempts policy acceptance without validation")

        source_residue_ids = getattr(patch, "source_residue_ids", None)
        if source_residue_ids == [] and updates:
            raise ValueError("Patch with target updates must cite source residues")

        for field_name, value in updates.items():
            self._validate_value(field_name, value)

    def _validate_value(self, field_name: str, value: Any) -> None:
        if field_name in {"probe_rate", "generated_probe_rate"}:
            rate = float(value)
            if rate < 0.0 or rate > 1.0:
                raise ValueError(f"{field_name} must stay within [0, 1]")
        if field_name in {
            "generated_grace_generations",
            "generated_min_evaluations",
        } and int(value) < 0:
            raise ValueError(f"{field_name} must be non-negative")
        if field_name == "clean_probe_first_eval" and not isinstance(value, bool):
            raise ValueError("clean_probe_first_eval must be boolean")
