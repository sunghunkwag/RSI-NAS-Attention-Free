"""AFIRSI evaluators that turn observations into structured residue."""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional

from .observation import GeneratedModuleLifecycleObservation
from .residue import FailureResidue


class Evaluator:
    """Versioned AFIRSI evaluator for generated-module evidence windows."""

    def __init__(
        self,
        expansion_interval: int,
        min_evaluations: int = 2,
        config: Optional[Any] = None,
        version: str = "evaluator.pruning_propagation_race.v1",
    ) -> None:
        self.expansion_interval = max(1, int(expansion_interval))
        self.min_evaluations = max(0, int(min_evaluations))
        self.config = config
        self.version = version

    @property
    def grace_multiplier(self) -> float:
        return float(getattr(self.config, "grace_multiplier", 3.0))

    @property
    def probe_rate(self) -> float:
        return float(getattr(self.config, "probe_rate", 1.0))

    def evaluate(
        self,
        observations: Iterable[GeneratedModuleLifecycleObservation],
    ) -> List[FailureResidue]:
        residues: List[FailureResidue] = []
        for observation in observations:
            residue = self.evaluate_observation(observation)
            if residue is not None:
                residues.append(residue)
        return residues

    def evaluate_observation(
        self,
        observation: GeneratedModuleLifecycleObservation,
    ) -> Optional[FailureResidue]:
        if observation.triggering_event != "prune_attempt":
            return None

        policy = observation.instrument_policy
        target_grace = max(
            int(policy.get("generated_grace_generations", 0)),
            int(math.ceil(self.grace_multiplier * self.expansion_interval)),
        )
        target_evals = max(
            int(policy.get("generated_min_evaluations", 0)),
            self.min_evaluations,
        )

        under_age = observation.age < target_grace
        under_evaluated = observation.evaluations < target_evals
        if not (under_age or under_evaluated):
            return None

        missing = {
            "required_age_generations": target_grace,
            "observed_age_generations": observation.age,
            "required_evaluations": target_evals,
            "observed_evaluations": observation.evaluations,
            "under_age": under_age,
            "under_evaluated": under_evaluated,
        }
        targets = []
        if int(policy.get("generated_grace_generations", 0)) < target_grace:
            targets.append("generated_grace_generations")
        if int(policy.get("generated_min_evaluations", 0)) < target_evals:
            targets.append("generated_min_evaluations")
        if float(policy.get("generated_probe_rate", 0.0)) < self.probe_rate:
            targets.append("generated_probe_rate")
        if observation.evaluations == 0:
            targets.append("clean_probe_first_eval")
        targets.extend(["evaluator_terms", "archive_insertion_priority"])

        severity = "high" if observation.evaluations == 0 else "medium"
        return FailureResidue(
            run_id=observation.run_id,
            generation=observation.generation,
            residue_type="PRUNING_PROPAGATION_RACE",
            subject_type=observation.subject_type,
            subject_id=observation.subject_id,
            triggering_event=observation.triggering_event,
            observed_evidence=observation.to_evidence_dict(),
            missing_evidence=missing,
            suspected_instrument_failure=(
                "Generated module is being pruned before the active "
                "instrument has collected the required age/evaluation evidence."
            ),
            problem_space_version=observation.problem_space_version,
            evaluator_version=self.version,
            observation_channel_version=observation.observation_channel_version,
            proposed_mutation_targets=sorted(set(targets)),
            severity=severity,
            confidence=0.95,
            policy_before=dict(policy),
        )
