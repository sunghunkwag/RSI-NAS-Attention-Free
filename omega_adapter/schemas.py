"""JSON-serializable AFIRSI residue and instrument-patch schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from afirsi_core.mutation_contract import InstrumentMutationContract


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
    subject_type: str = "generated_module"
    parent_policy_id: Optional[str] = None
    candidate_policy_id: Optional[str] = None
    cycle_index: Optional[int] = None
    patch_id: Optional[str] = None
    patch_family: Optional[str] = None
    suspected_bottleneck_category: Optional[str] = None
    diagnostic_payload: Dict[str, Any] = field(default_factory=dict)
    structured_rejection_reasons: List[Dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
