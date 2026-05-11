"""JSON-serializable AFIRSI residue and instrument-patch schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class StructuredAFIRSIResidue:
    """Structured failure evidence exported from a real RSI-NAS run."""

    residue_id: str
    generation: int
    seed: int
    condition: str
    residue_type: str
    generated_module_name: str
    source_action: str
    birth_generation: int
    evaluations: int
    archive_insertions: int
    elite_uses: int
    best_fitness: float
    prune_attempts: int
    protected_from_prune: int
    age_at_prune_attempt: int
    current_policy_snapshot: Dict[str, Any]
    observed_failure_signature: List[str]
    requested_instrument_classes: List[str]

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InstrumentMutationContract:
    """Defines what OMEGA-style synthesis may and may not mutate."""

    mutable_fields: List[str] = field(default_factory=lambda: [
        "grace_multiplier",
        "probe_rate",
        "clean_probe_first_eval",
        "meta_eval_gain",
        "meta_archive_gain",
        "meta_fitness_gain",
        "meta_exploration_floor",
        "generated_min_evaluations",
        "candidate_scoring_coefficients",
        "evaluator_terms",
        "archive_insertion_priority",
        "generated_module_scaffold_strategy",
    ])
    immutable_requirements: List[str] = field(default_factory=lambda: [
        "sandbox_boundaries",
        "actual_training_evaluation_required",
        "paired_seed_comparison_required",
        "holdout_seed_validation_required",
        "no_fake_bpc_values",
        "no_direct_open_ended_proxy_valid_assignment",
        "no_bypass_of_build_train_evaluate",
        "no_fixed_success_flag",
    ])

    def validate_patch(self, patch: "InstrumentPatch") -> None:
        allowed = set(self.mutable_fields)
        illegal = sorted(set(patch.target_updates) - allowed)
        if illegal:
            raise ValueError(f"Patch mutates immutable or unknown fields: {illegal}")
        forbidden_success_fields = {
            "open_ended_proxy_valid",
            "omega_validation_valid",
            "mechanism_valid",
            "success",
            "accepted",
        }
        forbidden = sorted(set(patch.target_updates) & forbidden_success_fields)
        if forbidden:
            raise ValueError(f"Patch attempts to set success fields: {forbidden}")


@dataclass(frozen=True)
class InstrumentPatch:
    """A residue-conditioned instrument mutation synthesized by the adapter."""

    patch_id: str
    candidate_name: str
    source_residue_ids: List[str]
    parent_policy_name: str
    target_updates: Dict[str, Any]
    candidate_scoring_coefficients: Dict[str, float]
    evaluator_terms: List[str]
    archive_insertion_priority: str
    generated_module_scaffold_strategy: str
    constraints_interpreted: List[str]
    rationale: List[str]
    provenance: Dict[str, Any]
    mutation_contract: InstrumentMutationContract = field(
        default_factory=InstrumentMutationContract
    )

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)
