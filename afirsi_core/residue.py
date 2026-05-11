"""Durable AFIRSI failure residue storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _stable_id(prefix: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).hexdigest()
    return f"{prefix}-{digest}"


@dataclass
class FailureResidue:
    """Structured evidence that an RSI instrument is destroying information."""

    generation: int
    residue_type: str
    subject_type: str
    subject_id: str
    triggering_event: str
    observed_evidence: Dict[str, Any]
    missing_evidence: Dict[str, Any]
    suspected_instrument_failure: str
    problem_space_version: str
    evaluator_version: str
    observation_channel_version: str
    proposed_mutation_targets: List[str]
    run_id: str = "rsi-nas"
    severity: str = "medium"
    confidence: float = 1.0
    residue_id: str = ""
    addressed: bool = False
    addressed_by_patch_id: Optional[str] = None
    policy_before: Dict[str, Any] = field(default_factory=dict)
    policy_after: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.residue_id:
            self.residue_id = _stable_id(
                "afirsi-core-residue",
                {
                    "run_id": self.run_id,
                    "generation": self.generation,
                    "residue_type": self.residue_type,
                    "subject_type": self.subject_type,
                    "subject_id": self.subject_id,
                    "triggering_event": self.triggering_event,
                    "prune_attempts": self.observed_evidence.get("prune_attempts"),
                    "evaluations": self.observed_evidence.get("evaluations"),
                    "problem_space_version": self.problem_space_version,
                },
            )

    @property
    def module_name(self) -> str:
        """Compatibility alias used by the existing omega exporter."""

        return self.subject_id

    @property
    def age(self) -> int:
        return int(self.observed_evidence.get("age", 0))

    @property
    def evaluations(self) -> int:
        return int(self.observed_evidence.get("evaluations", 0))

    @property
    def source_action(self) -> str:
        return str(self.observed_evidence.get("source_action", "unknown"))


class FailureResidueLedger:
    """Append-only queryable ledger for durable AFIRSI residue events."""

    def __init__(self) -> None:
        self._residues: Dict[str, FailureResidue] = {}
        self._order: List[str] = []

    def record(self, residue: FailureResidue) -> FailureResidue:
        residue_id = residue.residue_id
        if residue_id in self._residues:
            suffix = 2
            while f"{residue_id}-{suffix}" in self._residues:
                suffix += 1
            residue.residue_id = f"{residue_id}-{suffix}"
        self._residues[residue.residue_id] = residue
        self._order.append(residue.residue_id)
        return residue

    def get(self, residue_id: str) -> FailureResidue:
        try:
            return self._residues[residue_id]
        except KeyError as exc:
            raise KeyError(f"Unknown residue id: {residue_id}") from exc

    def all_residues(self) -> List[FailureResidue]:
        return [self._residues[residue_id] for residue_id in self._order]

    def query_by_type(self, residue_type: str) -> List[FailureResidue]:
        return [
            residue
            for residue in self.all_residues()
            if residue.residue_type == residue_type
        ]

    def unresolved_residues(
        self,
        residue_type: Optional[str] = None,
    ) -> List[FailureResidue]:
        rows = [
            residue
            for residue in self.all_residues()
            if not residue.addressed
        ]
        if residue_type is None:
            return rows
        return [residue for residue in rows if residue.residue_type == residue_type]

    def mark_addressed(
        self,
        residue_id: str,
        patch_id: str,
    ) -> FailureResidue:
        residue = self.get(residue_id)
        residue.addressed = True
        residue.addressed_by_patch_id = patch_id
        return residue

    def __len__(self) -> int:
        return len(self._order)
