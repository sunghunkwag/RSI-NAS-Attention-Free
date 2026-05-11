"""Executable AFIRSI experiment unit wrapper for RSI-NAS engine steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExperimentUnit:
    """Run one real RSI-NAS generation under a named problem-space version."""

    engine: Any
    problem_space_version_id: str
    instrument_policy_version: str
    unit_id: str = "experiment-unit"
    run_id: str = "rsi-nas"
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None

    def run(self, population_size: int = 6) -> Dict[str, Any]:
        generation_before = int(getattr(self.engine, "generation", 0))
        result = self.engine.step(population_size=population_size)
        self.result = dict(result)
        self.metadata.update({
            "generation_before": generation_before,
            "generation_after": result.get("generation"),
            "problem_space_version_id": self.problem_space_version_id,
            "instrument_policy_version": self.instrument_policy_version,
        })
        return result

    def observation_metadata(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "run_id": self.run_id,
            "problem_space_version_id": self.problem_space_version_id,
            "instrument_policy_version": self.instrument_policy_version,
            "generation": self.metadata.get("generation_after"),
        }
