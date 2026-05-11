"""Small provenance helpers for AFIRSI core artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ProvenanceRecord:
    source: str
    residue_ids: List[str]
    problem_space_version: str
    generator_version: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "residue_ids": list(self.residue_ids),
            "problem_space_version": self.problem_space_version,
            "generator_version": self.generator_version,
            "details": dict(self.details),
        }
