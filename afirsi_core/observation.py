"""Observation channels that convert RSI-NAS runtime state into evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


def _snapshot_policy(policy: Any) -> Dict[str, Any]:
    if policy is None:
        return {}
    if isinstance(policy, dict):
        return dict(policy)
    fields = [
        "generated_grace_generations",
        "generated_min_evaluations",
        "generated_probe_rate",
        "mutation_count",
        "last_mutation_generation",
    ]
    return {
        field_name: getattr(policy, field_name)
        for field_name in fields
        if hasattr(policy, field_name)
    }


@dataclass(frozen=True)
class GeneratedModuleLifecycleObservation:
    """Lifecycle evidence for one generated module at one event boundary."""

    run_id: str
    generation: int
    subject_id: str
    birth_generation: int
    source_action: str
    evaluations: int
    archive_insertions: int
    elite_uses: int
    prune_attempts: int
    protected_prune_attempts: int
    best_fitness: float
    last_evaluated_generation: int
    age: int
    triggering_event: str
    problem_space_version: str
    instrument_policy: Dict[str, Any]
    observation_channel_version: str
    subject_type: str = "generated_module"
    observation_id: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "run_id": self.run_id,
            "generation": self.generation,
            "subject_id": self.subject_id,
            "triggering_event": self.triggering_event,
            "prune_attempts": self.prune_attempts,
            "evaluations": self.evaluations,
            "problem_space_version": self.problem_space_version,
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        object.__setattr__(
            self,
            "observation_id",
            f"afirsi-observation-{hashlib.blake2b(raw, digest_size=8).hexdigest()}",
        )

    def to_evidence_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "birth_generation": self.birth_generation,
            "source_action": self.source_action,
            "evaluations": self.evaluations,
            "archive_insertions": self.archive_insertions,
            "elite_uses": self.elite_uses,
            "prune_attempts": self.prune_attempts,
            "protected_prune_attempts": self.protected_prune_attempts,
            "best_fitness": self.best_fitness,
            "last_evaluated_generation": self.last_evaluated_generation,
            "age": self.age,
            "triggering_event": self.triggering_event,
        }


class ObservationChannel:
    """Versioned channel from live RSI-NAS records to structured observations."""

    def __init__(self, version: str = "observation.generated_module_lifecycle.v1"):
        self.version = version

    def observe_generated_module(
        self,
        record: Any,
        generation: int,
        triggering_event: str,
        policy: Any,
        problem_space_version: str,
        run_id: str = "rsi-nas",
    ) -> GeneratedModuleLifecycleObservation:
        birth_generation = int(getattr(record, "birth_generation"))
        return GeneratedModuleLifecycleObservation(
            run_id=run_id,
            generation=int(generation),
            subject_id=str(getattr(record, "name")),
            birth_generation=birth_generation,
            source_action=str(getattr(record, "source_action")),
            evaluations=int(getattr(record, "evaluations")),
            archive_insertions=int(getattr(record, "archive_insertions")),
            elite_uses=int(getattr(record, "elite_uses")),
            prune_attempts=int(getattr(record, "prune_attempts")),
            protected_prune_attempts=int(getattr(record, "protected_from_prune")),
            best_fitness=float(getattr(record, "best_fitness")),
            last_evaluated_generation=int(
                getattr(record, "last_evaluated_generation")
            ),
            age=max(0, int(generation) - birth_generation),
            triggering_event=triggering_event,
            problem_space_version=problem_space_version,
            instrument_policy=_snapshot_policy(policy),
            observation_channel_version=self.version,
        )

    def observe_generated_modules(
        self,
        records: Iterable[Any],
        generation: int,
        triggering_event: str,
        policy: Any,
        problem_space_version: str,
        run_id: str = "rsi-nas",
    ) -> List[GeneratedModuleLifecycleObservation]:
        return [
            self.observe_generated_module(
                record=record,
                generation=generation,
                triggering_event=triggering_event,
                policy=policy,
                problem_space_version=problem_space_version,
                run_id=run_id,
            )
            for record in records
        ]
