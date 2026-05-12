"""EIE real-world instrument evolution benchmark.

This extends the repository's EIE idea beyond generated-module pruning. The
instrument being evolved here is the evaluator itself: which validation views,
stress probes, and robustness terms are allowed to select a learner.

The loop is intentionally bounded and auditable:

1. Fit candidate learners on train data only.
2. Evaluate them on validation data through the current epistemic instrument.
3. Diagnose evaluation blind spots as residue.
4. Synthesize an instrument patch from those residues.
5. Repeat, then perform final held-out test evaluation.

This ports the probationary evaluator-evolution idea from
DeepNeural-AutoExploration/evaluator_evolution.py into the real-data benchmark:
mutated evaluators are accepted only after validation-only adversarial checks.

This is stronger than fixed model selection, but it is still not AGI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from real_world_generalization_benchmark import (
    CandidatePipeline,
    RealDatasetTask,
    build_candidate_library,
    build_dataset_tasks,
)


ArrayPair = Tuple[np.ndarray, np.ndarray]


def _stable_id(prefix: str, payload: Dict[str, Any], digest_size: int = 7) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=digest_size).hexdigest()
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class EpistemicInstrument:
    instrument_id: str
    generation: int
    parent_instrument_id: Optional[str]
    balanced_accuracy_weight: float = 1.0
    f1_macro_weight: float = 0.0
    worst_class_recall_weight: float = 0.0
    noise_stress_weight: float = 0.0
    feature_dropout_stress_weight: float = 0.0
    ranking_stability_weight: float = 0.0
    complexity_penalty: float = 0.0
    diagnostic_noise_std: float = 0.05
    diagnostic_feature_dropout_rate: float = 0.08
    source_residue_ids: Tuple[str, ...] = ()
    status: str = "accepted"

    def score(self, metrics: Dict[str, float], candidate: CandidatePipeline) -> float:
        weights = {
            "validation_balanced_accuracy": self.balanced_accuracy_weight,
            "validation_f1_macro": self.f1_macro_weight,
            "worst_class_recall": self.worst_class_recall_weight,
            "noise_balanced_accuracy": self.noise_stress_weight,
            "feature_dropout_balanced_accuracy": self.feature_dropout_stress_weight,
            "ranking_stability": self.ranking_stability_weight,
        }
        total_weight = sum(max(0.0, weight) for weight in weights.values())
        if total_weight <= 0.0:
            total_weight = 1.0
        raw = sum(metrics[key] * weight for key, weight in weights.items()) / total_weight
        return float(raw - self.complexity_penalty * candidate.complexity)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InstrumentResidue:
    residue_id: str
    generation: int
    task_name: str
    residue_type: str
    selected_candidate: str
    observed_evidence: Dict[str, Any]
    missing_evidence: Dict[str, Any]
    proposed_mutation_targets: Tuple[str, ...]
    severity: str = "medium"
    confidence: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpistemicInstrumentPatch:
    patch_id: str
    parent_instrument_id: str
    source_residue_ids: Tuple[str, ...]
    target_updates: Dict[str, Any]
    rationale: Tuple[str, ...]
    evaluator_terms: Tuple[str, ...]
    validation_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InstrumentProbationDecision:
    parent_instrument_id: str
    candidate_instrument_id: str
    accepted: bool
    status: str
    generation: int
    ranking_changed_tasks: Tuple[str, ...]
    parent_validation_summary: Dict[str, float]
    candidate_validation_summary: Dict[str, float]
    score_delta: float
    adversarial_checks: Dict[str, Any]
    rejection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InstrumentCandidateEvaluation:
    task_name: str
    candidate: Dict[str, Any]
    fit_ok: bool
    metrics: Dict[str, float]
    instrument_score: float
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def selection_tuple(self) -> Tuple[float, float, float, int, str]:
        return (
            float(self.instrument_score),
            float(self.metrics.get("validation_balanced_accuracy", 0.0)),
            float(self.metrics.get("validation_f1_macro", 0.0)),
            -int(self.candidate.get("complexity", 999)),
            str(self.candidate.get("name", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvolutionCycle:
    generation: int
    instrument_before: Dict[str, Any]
    selected_by_task: Dict[str, str]
    residues: List[InstrumentResidue]
    patch: Optional[EpistemicInstrumentPatch]
    probation_decision: Optional[InstrumentProbationDecision]
    instrument_after: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "instrument_before": self.instrument_before,
            "selected_by_task": dict(self.selected_by_task),
            "residues": [residue.to_dict() for residue in self.residues],
            "patch": None if self.patch is None else self.patch.to_dict(),
            "probation_decision": (
                None if self.probation_decision is None else self.probation_decision.to_dict()
            ),
            "instrument_after": self.instrument_after,
        }


@dataclass
class EIEDatasetResult:
    task_name: str
    domain: str
    passed: bool
    selected_candidate: Dict[str, Any]
    baseline: Dict[str, float]
    validation_metrics: Dict[str, float]
    clean_test_metrics: Dict[str, float]
    stress_test_metrics: Dict[str, float]
    thresholds: Dict[str, float]
    candidate_count: int
    split: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EIEGate:
    name: str
    passed: bool
    score: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EIERealWorldReport:
    final_verdict: str
    eie_real_benchmark_success: bool
    instrument_evolved: bool
    agi_claim_verified: bool
    final_instrument: Dict[str, Any]
    evolution_cycles: List[EvolutionCycle]
    dataset_results: List[EIEDatasetResult]
    gates: List[EIEGate]
    failure_reasons: List[str]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "final_verdict": self.final_verdict,
            "eie_real_benchmark_success": self.eie_real_benchmark_success,
            "instrument_evolved": self.instrument_evolved,
            "agi_claim_verified": self.agi_claim_verified,
            "final_instrument": dict(self.final_instrument),
            "evolution_cycles": [cycle.to_dict() for cycle in self.evolution_cycles],
            "dataset_results": [result.to_dict() for result in self.dataset_results],
            "gates": [asdict(gate) for gate in self.gates],
            "failure_reasons": list(self.failure_reasons),
        }


class EIERealWorldInstrumentEvolver:
    """Residue-conditioned evaluator evolution over real datasets."""

    def __init__(self, seed: int = 17, generations: int = 2) -> None:
        self.seed = seed
        self.generations = max(1, int(generations))
        self.candidates = build_candidate_library()
        self._candidate_by_name = {candidate.name: candidate for candidate in self.candidates}

    def evaluate(self, tasks: Sequence[RealDatasetTask]) -> EIERealWorldReport:
        splits = {task.name: self._split(task) for task in tasks}
        seed_instrument = self._seed_instrument()
        instrument = seed_instrument
        cycles: List[EvolutionCycle] = []

        for generation in range(self.generations):
            selected_by_task: Dict[str, str] = {}
            residues: List[InstrumentResidue] = []
            for task in tasks:
                evaluations = self._evaluate_validation_candidates(
                    task,
                    splits[task.name],
                    instrument,
                )
                selected = self._select(evaluations)
                selected_by_task[task.name] = str(selected.candidate["name"])
                residues.extend(self._diagnose_residue(task, generation, instrument, evaluations, selected))

            patch = self._make_patch(instrument, residues)
            probation_decision = None
            next_instrument = instrument
            if patch is not None:
                candidate_instrument = self._apply_patch(instrument, patch)
                probation_decision = self._validate_probationary_instrument(
                    tasks=tasks,
                    splits=splits,
                    parent=instrument,
                    candidate=candidate_instrument,
                    generation=generation,
                )
                if probation_decision.accepted:
                    next_instrument = candidate_instrument
            cycles.append(EvolutionCycle(
                generation=generation,
                instrument_before=instrument.to_dict(),
                selected_by_task=selected_by_task,
                residues=residues,
                patch=patch,
                probation_decision=probation_decision,
                instrument_after=next_instrument.to_dict(),
            ))
            if patch is None or probation_decision is None or not probation_decision.accepted:
                break
            instrument = next_instrument

        dataset_results = [
            self._final_evaluate_task(task, splits[task.name], instrument)
            for task in tasks
        ]
        instrument_evolved = instrument.instrument_id != seed_instrument.instrument_id
        success = all(result.passed for result in dataset_results) and instrument_evolved
        gates = self._build_gates(dataset_results, cycles, seed_instrument, instrument)
        agi_verified = all(gate.passed for gate in gates)
        if agi_verified:
            verdict = "AGI_CLAIM_VERIFIED"
        elif success:
            verdict = "EIE_REAL_WORLD_INSTRUMENT_EVOLVED_BUT_AGI_FAILED"
        else:
            verdict = "EIE_REAL_WORLD_INSTRUMENT_ATTEMPT_FAILED"
        return EIERealWorldReport(
            final_verdict=verdict,
            eie_real_benchmark_success=success,
            instrument_evolved=instrument_evolved,
            agi_claim_verified=agi_verified,
            final_instrument=instrument.to_dict(),
            evolution_cycles=cycles,
            dataset_results=dataset_results,
            gates=gates,
            failure_reasons=[
                str(gate.evidence.get("missing"))
                for gate in gates
                if not gate.passed and gate.evidence.get("missing")
            ],
        )

    def _seed_instrument(self) -> EpistemicInstrument:
        payload = {
            "balanced_accuracy_weight": 1.0,
            "diagnostic_noise_std": 0.05,
            "diagnostic_feature_dropout_rate": 0.08,
        }
        return EpistemicInstrument(
            instrument_id=_stable_id("eie-instrument-seed", payload),
            generation=0,
            parent_instrument_id=None,
        )

    def _split(self, task: RealDatasetTask) -> Dict[str, ArrayPair]:
        from sklearn.model_selection import train_test_split

        x = task.x.reshape(task.x.shape[0], -1)
        y = task.y
        x_train, x_temp, y_train, y_temp = train_test_split(
            x,
            y,
            test_size=0.4,
            random_state=self.seed,
            stratify=y,
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.5,
            random_state=self.seed + 1,
            stratify=y_temp,
        )
        return {
            "train": (x_train, y_train),
            "validation": (x_val, y_val),
            "test": (x_test, y_test),
        }

    def _evaluate_validation_candidates(
        self,
        task: RealDatasetTask,
        split: Dict[str, ArrayPair],
        instrument: EpistemicInstrument,
    ) -> List[InstrumentCandidateEvaluation]:
        x_train, y_train = split["train"]
        x_val, y_val = split["validation"]
        return [
            self._evaluate_candidate(
                task=task,
                candidate=candidate,
                x_train=x_train,
                y_train=y_train,
                x_eval=x_val,
                y_eval=y_val,
                instrument=instrument,
                split_name="validation",
            )
            for candidate in self.candidates
        ]

    def _evaluate_candidate(
        self,
        task: RealDatasetTask,
        candidate: CandidatePipeline,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        instrument: EpistemicInstrument,
        split_name: str,
    ) -> InstrumentCandidateEvaluation:
        start = time.monotonic()
        try:
            model = candidate.build(self.seed)
            model.fit(x_train, y_train)
            metrics = self._instrument_metrics(
                task=task,
                model=model,
                x_train=x_train,
                x_eval=x_eval,
                y_eval=y_eval,
                instrument=instrument,
                candidate_name=candidate.name,
                split_name=split_name,
            )
            score = instrument.score(metrics, candidate)
            return InstrumentCandidateEvaluation(
                task_name=task.name,
                candidate=candidate.to_dict(),
                fit_ok=True,
                metrics=metrics,
                instrument_score=score,
                elapsed_seconds=time.monotonic() - start,
            )
        except Exception as exc:
            return InstrumentCandidateEvaluation(
                task_name=task.name,
                candidate=candidate.to_dict(),
                fit_ok=False,
                metrics={},
                instrument_score=-1.0,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - start,
            )

    def _instrument_metrics(
        self,
        task: RealDatasetTask,
        model: BaseEstimator,
        x_train: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        instrument: EpistemicInstrument,
        candidate_name: str,
        split_name: str,
    ) -> Dict[str, float]:
        clean = self._score_predictions(model, x_eval, y_eval, prefix=split_name)
        noise_x = self._noise_stress(
            task.name,
            candidate_name,
            split_name,
            x_train,
            x_eval,
            instrument.diagnostic_noise_std,
        )
        dropout_x = self._feature_dropout_stress(
            task.name,
            candidate_name,
            split_name,
            x_train,
            x_eval,
            instrument.diagnostic_feature_dropout_rate,
        )
        noise = self._score_predictions(model, noise_x, y_eval, prefix="noise")
        dropout = self._score_predictions(model, dropout_x, y_eval, prefix="feature_dropout")
        stress_floor = min(
            noise["noise_balanced_accuracy"],
            dropout["feature_dropout_balanced_accuracy"],
        )
        clean_balanced = clean[f"{split_name}_balanced_accuracy"]
        metrics = {
            "validation_balanced_accuracy": clean_balanced,
            "validation_accuracy": clean[f"{split_name}_accuracy"],
            "validation_f1_macro": clean[f"{split_name}_f1_macro"],
            "worst_class_recall": clean[f"{split_name}_worst_class_recall"],
            "noise_balanced_accuracy": noise["noise_balanced_accuracy"],
            "feature_dropout_balanced_accuracy": dropout["feature_dropout_balanced_accuracy"],
            "ranking_stability": max(0.0, 1.0 - max(0.0, clean_balanced - stress_floor)),
            "stress_floor_balanced_accuracy": stress_floor,
            "stress_gap": max(0.0, clean_balanced - stress_floor),
        }
        return metrics

    def _score_predictions(
        self,
        model: BaseEstimator,
        x: np.ndarray,
        y: np.ndarray,
        prefix: str,
    ) -> Dict[str, float]:
        prediction = model.predict(x)
        per_class = recall_score(y, prediction, average=None, labels=np.unique(y), zero_division=0)
        return {
            f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            f"{prefix}_accuracy": float(accuracy_score(y, prediction)),
            f"{prefix}_f1_macro": float(f1_score(y, prediction, average="macro")),
            f"{prefix}_worst_class_recall": float(np.min(per_class)),
        }

    def _noise_stress(
        self,
        task_name: str,
        candidate_name: str,
        split_name: str,
        x_train: np.ndarray,
        x_eval: np.ndarray,
        noise_std: float,
    ) -> np.ndarray:
        rng = self._rng(task_name, candidate_name, split_name, "noise")
        feature_std = np.std(x_train, axis=0)
        feature_std = np.where(feature_std > 0.0, feature_std, 1.0)
        noise = rng.normal(0.0, noise_std, size=x_eval.shape) * feature_std
        return x_eval + noise

    def _feature_dropout_stress(
        self,
        task_name: str,
        candidate_name: str,
        split_name: str,
        x_train: np.ndarray,
        x_eval: np.ndarray,
        rate: float,
    ) -> np.ndarray:
        rng = self._rng(task_name, candidate_name, split_name, "feature_dropout")
        means = np.mean(x_train, axis=0)
        mask = rng.random(x_eval.shape) < rate
        stressed = np.array(x_eval, copy=True)
        stressed[mask] = np.take(means, np.where(mask)[1])
        return stressed

    def _rng(self, *parts: str) -> np.random.Generator:
        payload = "|".join([str(self.seed), *parts])
        seed = int(hashlib.blake2b(payload.encode("utf-8"), digest_size=4).hexdigest(), 16)
        return np.random.default_rng(seed)

    def _select(
        self,
        evaluations: Sequence[InstrumentCandidateEvaluation],
    ) -> InstrumentCandidateEvaluation:
        valid = [evaluation for evaluation in evaluations if evaluation.fit_ok]
        if not valid:
            raise ValueError("no valid candidate evaluations")
        return sorted(valid, key=lambda evaluation: evaluation.selection_tuple(), reverse=True)[0]

    def _diagnose_residue(
        self,
        task: RealDatasetTask,
        generation: int,
        instrument: EpistemicInstrument,
        evaluations: Sequence[InstrumentCandidateEvaluation],
        selected: InstrumentCandidateEvaluation,
    ) -> List[InstrumentResidue]:
        residues: List[InstrumentResidue] = []
        metrics = selected.metrics
        selected_name = str(selected.candidate["name"])
        clean = metrics["validation_balanced_accuracy"]
        worst = metrics["worst_class_recall"]
        stress_gap = metrics["stress_gap"]
        stress_best = self._best_stress_candidate(evaluations)

        if instrument.worst_class_recall_weight == 0.0 and (
            worst < clean - 0.035 or worst < 0.90
        ):
            residues.append(self._make_residue(
                generation=generation,
                task_name=task.name,
                residue_type="CLASS_RECALL_BLIND_SPOT",
                selected_candidate=selected_name,
                observed={
                    "validation_balanced_accuracy": clean,
                    "worst_class_recall": worst,
                    "candidate": selected_name,
                },
                missing={
                    "missing_instrument_term": "worst_class_recall",
                    "reason": "clean balanced accuracy hides a weak class slice",
                },
                targets=("worst_class_recall_weight", "f1_macro_weight"),
            ))

        if instrument.noise_stress_weight == 0.0 and stress_gap > 0.015:
            residues.append(self._make_residue(
                generation=generation,
                task_name=task.name,
                residue_type="ROBUSTNESS_STRESS_BLIND_SPOT",
                selected_candidate=selected_name,
                observed={
                    "validation_balanced_accuracy": clean,
                    "stress_floor_balanced_accuracy": metrics["stress_floor_balanced_accuracy"],
                    "stress_gap": stress_gap,
                    "candidate": selected_name,
                },
                missing={
                    "missing_instrument_terms": [
                        "noise_stress_weight",
                        "feature_dropout_stress_weight",
                    ],
                    "reason": "candidate selection ignores validation stress degradation",
                },
                targets=(
                    "noise_stress_weight",
                    "feature_dropout_stress_weight",
                    "ranking_stability_weight",
                ),
            ))

        if (
            instrument.ranking_stability_weight == 0.0
            and stress_best is not None
            and str(stress_best.candidate["name"]) != selected_name
        ):
            residues.append(self._make_residue(
                generation=generation,
                task_name=task.name,
                residue_type="STRESS_RANKING_DISAGREEMENT",
                selected_candidate=selected_name,
                observed={
                    "clean_selected_candidate": selected_name,
                    "stress_selected_candidate": stress_best.candidate["name"],
                    "clean_score": selected.instrument_score,
                    "stress_floor_of_clean_selection": metrics["stress_floor_balanced_accuracy"],
                    "stress_floor_of_stress_selection": stress_best.metrics[
                        "stress_floor_balanced_accuracy"
                    ],
                },
                missing={
                    "missing_instrument_term": "ranking_stability",
                    "reason": "the evaluator would choose a different learner under diagnostic stress probes",
                },
                targets=("ranking_stability_weight",),
            ))

        if not residues and self._has_unweighted_diagnostic_signal(instrument, evaluations):
            residues.append(self._make_residue(
                generation=generation,
                task_name=task.name,
                residue_type="UNWEIGHTED_DIAGNOSTIC_SIGNAL",
                selected_candidate=selected_name,
                observed={
                    "candidate_count": len(evaluations),
                    "max_stress_gap": max(e.metrics.get("stress_gap", 0.0) for e in evaluations),
                },
                missing={
                    "missing_instrument_term": "diagnostic_probe_weighting",
                    "reason": "diagnostic probes exist but cannot influence selection",
                },
                targets=("ranking_stability_weight", "noise_stress_weight"),
                severity="low",
                confidence=0.75,
            ))
        return residues

    def _make_residue(
        self,
        generation: int,
        task_name: str,
        residue_type: str,
        selected_candidate: str,
        observed: Dict[str, Any],
        missing: Dict[str, Any],
        targets: Tuple[str, ...],
        severity: str = "medium",
        confidence: float = 0.9,
    ) -> InstrumentResidue:
        payload = {
            "generation": generation,
            "task_name": task_name,
            "residue_type": residue_type,
            "selected_candidate": selected_candidate,
            "observed": observed,
            "targets": targets,
        }
        return InstrumentResidue(
            residue_id=_stable_id("eie-real-residue", payload),
            generation=generation,
            task_name=task_name,
            residue_type=residue_type,
            selected_candidate=selected_candidate,
            observed_evidence=observed,
            missing_evidence=missing,
            proposed_mutation_targets=targets,
            severity=severity,
            confidence=confidence,
        )

    def _best_stress_candidate(
        self,
        evaluations: Sequence[InstrumentCandidateEvaluation],
    ) -> Optional[InstrumentCandidateEvaluation]:
        valid = [evaluation for evaluation in evaluations if evaluation.fit_ok]
        if not valid:
            return None
        return sorted(
            valid,
            key=lambda evaluation: (
                evaluation.metrics["stress_floor_balanced_accuracy"],
                evaluation.metrics["worst_class_recall"],
                -int(evaluation.candidate.get("complexity", 999)),
                str(evaluation.candidate["name"]),
            ),
            reverse=True,
        )[0]

    def _has_unweighted_diagnostic_signal(
        self,
        instrument: EpistemicInstrument,
        evaluations: Sequence[InstrumentCandidateEvaluation],
    ) -> bool:
        if instrument.noise_stress_weight > 0.0 or instrument.ranking_stability_weight > 0.0:
            return False
        gaps = [evaluation.metrics.get("stress_gap", 0.0) for evaluation in evaluations]
        return bool(gaps and max(gaps) > 0.01)

    def _make_patch(
        self,
        instrument: EpistemicInstrument,
        residues: Sequence[InstrumentResidue],
    ) -> Optional[EpistemicInstrumentPatch]:
        if not residues:
            return None
        residue_types = {residue.residue_type for residue in residues}
        updates: Dict[str, Any] = {}
        evaluator_terms: List[str] = []
        rationale: List[str] = []

        if "CLASS_RECALL_BLIND_SPOT" in residue_types:
            updates["worst_class_recall_weight"] = min(
                0.45,
                instrument.worst_class_recall_weight + 0.25,
            )
            updates["f1_macro_weight"] = min(0.35, instrument.f1_macro_weight + 0.15)
            evaluator_terms.extend(["class_slice_recall", "macro_f1"])
            rationale.append("Residue showed that aggregate validation skill hid a weak class slice.")

        if "ROBUSTNESS_STRESS_BLIND_SPOT" in residue_types:
            updates["noise_stress_weight"] = min(0.35, instrument.noise_stress_weight + 0.22)
            updates["feature_dropout_stress_weight"] = min(
                0.30,
                instrument.feature_dropout_stress_weight + 0.18,
            )
            updates["ranking_stability_weight"] = min(
                0.25,
                instrument.ranking_stability_weight + 0.12,
            )
            evaluator_terms.extend(["noise_stress", "feature_dropout_stress"])
            rationale.append("Residue showed validation stress degradation ignored by the active evaluator.")

        if "STRESS_RANKING_DISAGREEMENT" in residue_types:
            updates["ranking_stability_weight"] = max(
                updates.get("ranking_stability_weight", instrument.ranking_stability_weight),
                min(0.30, instrument.ranking_stability_weight + 0.20),
            )
            updates["complexity_penalty"] = min(0.01, instrument.complexity_penalty + 0.002)
            evaluator_terms.append("stress_ranking_stability")
            rationale.append("Diagnostic stress probes selected a different learner than the clean evaluator.")

        if "UNWEIGHTED_DIAGNOSTIC_SIGNAL" in residue_types:
            updates["ranking_stability_weight"] = max(
                updates.get("ranking_stability_weight", instrument.ranking_stability_weight),
                min(0.20, instrument.ranking_stability_weight + 0.10),
            )
            updates["noise_stress_weight"] = max(
                updates.get("noise_stress_weight", instrument.noise_stress_weight),
                min(0.20, instrument.noise_stress_weight + 0.10),
            )
            evaluator_terms.append("diagnostic_probe_weighting")
            rationale.append("Diagnostic probes existed but had no causal influence on selection.")

        if not updates:
            return None
        source_ids = tuple(residue.residue_id for residue in residues)
        patch_payload = {
            "parent": instrument.instrument_id,
            "source_residue_ids": source_ids,
            "target_updates": updates,
            "evaluator_terms": sorted(set(evaluator_terms)),
        }
        return EpistemicInstrumentPatch(
            patch_id=_stable_id("eie-real-patch", patch_payload),
            parent_instrument_id=instrument.instrument_id,
            source_residue_ids=source_ids,
            target_updates=updates,
            rationale=tuple(rationale),
            evaluator_terms=tuple(sorted(set(evaluator_terms))),
        )

    def _apply_patch(
        self,
        instrument: EpistemicInstrument,
        patch: EpistemicInstrumentPatch,
    ) -> EpistemicInstrument:
        if not patch.validation_required:
            raise ValueError("instrument patch attempted to skip validation")
        if not patch.source_residue_ids:
            raise ValueError("instrument patch must cite source residues")
        allowed = {
            "balanced_accuracy_weight",
            "f1_macro_weight",
            "worst_class_recall_weight",
            "noise_stress_weight",
            "feature_dropout_stress_weight",
            "ranking_stability_weight",
            "complexity_penalty",
            "diagnostic_noise_std",
            "diagnostic_feature_dropout_rate",
        }
        illegal = sorted(set(patch.target_updates) - allowed)
        if illegal:
            raise ValueError(f"illegal instrument updates: {illegal}")
        payload = {
            "parent": instrument.instrument_id,
            "generation": instrument.generation + 1,
            "updates": patch.target_updates,
            "source_residue_ids": patch.source_residue_ids,
        }
        return replace(
            instrument,
            instrument_id=_stable_id("eie-instrument", payload),
            generation=instrument.generation + 1,
            parent_instrument_id=instrument.instrument_id,
            source_residue_ids=tuple(sorted(set(instrument.source_residue_ids + patch.source_residue_ids))),
            status="probation",
            **patch.target_updates,
        )

    def _validate_probationary_instrument(
        self,
        tasks: Sequence[RealDatasetTask],
        splits: Dict[str, Dict[str, ArrayPair]],
        parent: EpistemicInstrument,
        candidate: EpistemicInstrument,
        generation: int,
    ) -> InstrumentProbationDecision:
        parent_rows = []
        candidate_rows = []
        ranking_changed: List[str] = []
        for task in tasks:
            split = splits[task.name]
            parent_selected = self._select(self._evaluate_validation_candidates(task, split, parent))
            candidate_selected = self._select(self._evaluate_validation_candidates(task, split, candidate))
            parent_rows.append(parent_selected)
            candidate_rows.append(candidate_selected)
            if parent_selected.candidate["name"] != candidate_selected.candidate["name"]:
                ranking_changed.append(task.name)

        parent_summary = self._selection_summary(parent_rows)
        candidate_summary = self._selection_summary(candidate_rows)
        clean_delta = (
            candidate_summary["validation_balanced_accuracy"]
            - parent_summary["validation_balanced_accuracy"]
        )
        stress_delta = (
            candidate_summary["stress_floor_balanced_accuracy"]
            - parent_summary["stress_floor_balanced_accuracy"]
        )
        worst_delta = (
            candidate_summary["worst_class_recall"]
            - parent_summary["worst_class_recall"]
        )
        score_delta = stress_delta + 0.5 * worst_delta + 0.25 * clean_delta
        weights_changed = (
            candidate.noise_stress_weight > parent.noise_stress_weight
            or candidate.feature_dropout_stress_weight > parent.feature_dropout_stress_weight
            or candidate.ranking_stability_weight > parent.ranking_stability_weight
            or candidate.worst_class_recall_weight > parent.worst_class_recall_weight
        )
        adversarial = {
            "source_residue_required": bool(candidate.source_residue_ids),
            "test_split_not_used": True,
            "weights_changed": weights_changed,
            "no_clean_validation_collapse": clean_delta >= -0.02,
            "no_diagnostic_regression": score_delta >= -0.01,
            "ranking_changed_or_new_terms": bool(ranking_changed) or weights_changed,
        }
        accepted = all(bool(value) for value in adversarial.values())
        return InstrumentProbationDecision(
            parent_instrument_id=parent.instrument_id,
            candidate_instrument_id=candidate.instrument_id,
            accepted=accepted,
            status="probation" if accepted else "rejected",
            generation=generation,
            ranking_changed_tasks=tuple(ranking_changed),
            parent_validation_summary=parent_summary,
            candidate_validation_summary=candidate_summary,
            score_delta=float(score_delta),
            adversarial_checks=adversarial,
            rejection_reason=(
                "accepted_for_probation"
                if accepted
                else "failed_validation_only_adversarial_checks"
            ),
        )

    def _selection_summary(
        self,
        selections: Sequence[InstrumentCandidateEvaluation],
    ) -> Dict[str, float]:
        if not selections:
            return {
                "validation_balanced_accuracy": 0.0,
                "validation_f1_macro": 0.0,
                "worst_class_recall": 0.0,
                "stress_floor_balanced_accuracy": 0.0,
                "ranking_stability": 0.0,
            }
        keys = [
            "validation_balanced_accuracy",
            "validation_f1_macro",
            "worst_class_recall",
            "stress_floor_balanced_accuracy",
            "ranking_stability",
        ]
        return {
            key: float(sum(row.metrics[key] for row in selections) / len(selections))
            for key in keys
        }

    def _final_evaluate_task(
        self,
        task: RealDatasetTask,
        split: Dict[str, ArrayPair],
        instrument: EpistemicInstrument,
    ) -> EIEDatasetResult:
        x_train, y_train = split["train"]
        x_val, y_val = split["validation"]
        x_test, y_test = split["test"]
        validation_evaluations = self._evaluate_validation_candidates(task, split, instrument)
        selected_eval = self._select(validation_evaluations)
        selected_candidate = self._candidate_by_name[str(selected_eval.candidate["name"])]

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(x_train, y_train)
        baseline_clean = self._plain_score(baseline, x_test, y_test)

        final_model = clone(selected_candidate.build(self.seed))
        x_fit = np.concatenate([x_train, x_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        final_model.fit(x_fit, y_fit)

        clean_test = self._plain_score(final_model, x_test, y_test)
        stress_test = self._final_stress_score(
            task,
            final_model,
            x_fit,
            x_test,
            y_test,
            instrument,
            selected_candidate.name,
        )
        clean_margin = clean_test["balanced_accuracy"] - baseline_clean["balanced_accuracy"]
        stress_threshold = max(0.72, task.min_test_balanced_accuracy - 0.14)
        passed = (
            clean_test["balanced_accuracy"] >= task.min_test_balanced_accuracy
            and clean_margin >= task.min_baseline_margin
            and stress_test["stress_floor_balanced_accuracy"] >= stress_threshold
        )
        return EIEDatasetResult(
            task_name=task.name,
            domain=task.domain,
            passed=passed,
            selected_candidate=selected_candidate.to_dict(),
            baseline=baseline_clean,
            validation_metrics=selected_eval.metrics,
            clean_test_metrics=clean_test,
            stress_test_metrics=stress_test,
            thresholds={
                "min_clean_test_balanced_accuracy": task.min_test_balanced_accuracy,
                "min_baseline_margin": task.min_baseline_margin,
                "observed_baseline_margin": clean_margin,
                "min_stress_floor_balanced_accuracy": stress_threshold,
            },
            candidate_count=len(self.candidates),
            split={name: int(values[0].shape[0]) for name, values in split.items()},
        )

    def _plain_score(self, model: BaseEstimator, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        prediction = model.predict(x)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "accuracy": float(accuracy_score(y, prediction)),
            "f1_macro": float(f1_score(y, prediction, average="macro")),
        }

    def _final_stress_score(
        self,
        task: RealDatasetTask,
        model: BaseEstimator,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        instrument: EpistemicInstrument,
        candidate_name: str,
    ) -> Dict[str, float]:
        noise_x = self._noise_stress(
            task.name,
            candidate_name,
            "heldout_test",
            x_train,
            x_test,
            instrument.diagnostic_noise_std,
        )
        dropout_x = self._feature_dropout_stress(
            task.name,
            candidate_name,
            "heldout_test",
            x_train,
            x_test,
            instrument.diagnostic_feature_dropout_rate,
        )
        noise = self._plain_score(model, noise_x, y_test)
        dropout = self._plain_score(model, dropout_x, y_test)
        return {
            "noise_balanced_accuracy": noise["balanced_accuracy"],
            "feature_dropout_balanced_accuracy": dropout["balanced_accuracy"],
            "stress_floor_balanced_accuracy": min(
                noise["balanced_accuracy"],
                dropout["balanced_accuracy"],
            ),
        }

    def _build_gates(
        self,
        results: Sequence[EIEDatasetResult],
        cycles: Sequence[EvolutionCycle],
        seed_instrument: EpistemicInstrument,
        final_instrument: EpistemicInstrument,
    ) -> List[EIEGate]:
        all_residues = [residue for cycle in cycles for residue in cycle.residues]
        patches = [cycle.patch for cycle in cycles if cycle.patch is not None]
        probation_decisions = [
            cycle.probation_decision
            for cycle in cycles
            if cycle.probation_decision is not None
        ]
        average_clean = sum(
            result.clean_test_metrics["balanced_accuracy"] for result in results
        ) / max(1, len(results))
        average_stress = sum(
            result.stress_test_metrics["stress_floor_balanced_accuracy"]
            for result in results
        ) / max(1, len(results))
        gates = [
            EIEGate(
                name="residue_conditioned_instrument_mutation",
                passed=bool(patches and all(patch.source_residue_ids for patch in patches)),
                score=float(len(patches)),
                evidence={
                    "patch_count": len(patches),
                    "residue_count": len(all_residues),
                    "residue_types": sorted({residue.residue_type for residue in all_residues}),
                },
            ),
            EIEGate(
                name="instrument_lineage_changed_evaluator_terms",
                passed=final_instrument.instrument_id != seed_instrument.instrument_id
                and final_instrument.source_residue_ids,
                score=float(final_instrument.generation),
                evidence={
                    "seed_instrument_id": seed_instrument.instrument_id,
                    "final_instrument_id": final_instrument.instrument_id,
                    "final_weights": {
                        "balanced_accuracy_weight": final_instrument.balanced_accuracy_weight,
                        "f1_macro_weight": final_instrument.f1_macro_weight,
                        "worst_class_recall_weight": final_instrument.worst_class_recall_weight,
                        "noise_stress_weight": final_instrument.noise_stress_weight,
                        "feature_dropout_stress_weight": final_instrument.feature_dropout_stress_weight,
                        "ranking_stability_weight": final_instrument.ranking_stability_weight,
                    },
                },
            ),
            EIEGate(
                name="probationary_evaluator_acceptance",
                passed=bool(probation_decisions)
                and all(decision.accepted for decision in probation_decisions),
                score=float(sum(1 for decision in probation_decisions if decision.accepted)),
                evidence={
                    "decision_count": len(probation_decisions),
                    "accepted": sum(1 for decision in probation_decisions if decision.accepted),
                    "statuses": [decision.status for decision in probation_decisions],
                    "adversarial_checks": [
                        decision.adversarial_checks for decision in probation_decisions
                    ],
                },
            ),
            EIEGate(
                name="heldout_real_data_after_eie_selection",
                passed=all(result.passed for result in results),
                score=average_clean,
                evidence={
                    "passed": sum(1 for result in results if result.passed),
                    "total": len(results),
                    "average_stress_floor_balanced_accuracy": average_stress,
                },
            ),
            EIEGate(
                name="test_split_not_used_for_instrument_evolution",
                passed=True,
                score=1.0,
                evidence={
                    "evolution_signal": "train and validation only",
                    "test_signal": "held-out once after final instrument selection",
                },
            ),
            EIEGate(
                name="learner_primitive_invention",
                passed=False,
                score=0.0,
                evidence={
                    "observed": "the evaluator instrument evolved, but candidate learners were predefined",
                    "missing": "new learner primitives synthesized and accepted from real-data residue",
                },
            ),
            EIEGate(
                name="autonomous_goal_formation",
                passed=False,
                score=0.0,
                evidence={
                    "observed": "EIE mutates evaluator terms for supplied benchmark goals",
                    "missing": "self-proposed real-world goals and success criteria",
                },
            ),
            EIEGate(
                name="open_ended_recursive_self_improvement",
                passed=False,
                score=0.0,
                evidence={
                    "observed": "bounded evaluator evolution over a fixed candidate library",
                    "missing": "recursive improvement that creates new task families or learning mechanisms without manual insertion",
                },
            ),
        ]
        return gates


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="breast_cancer_diagnosis,wine_chemical_origin,handwritten_digit_recognition",
        help="Comma-separated dataset task names to run.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--json-output", default="eie_real_world_report.json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tasks = build_dataset_tasks(args.datasets.split(","))
    if not tasks:
        raise SystemExit("no selected datasets")
    report = EIERealWorldInstrumentEvolver(
        seed=args.seed,
        generations=args.generations,
    ).evaluate(tasks)
    output = Path(args.json_output)
    output.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"final_verdict={report.final_verdict}")
    print(f"eie_real_benchmark_success={report.eie_real_benchmark_success}")
    print(f"instrument_evolved={report.instrument_evolved}")
    print(f"agi_claim_verified={report.agi_claim_verified}")
    print(
        "final_instrument_weights="
        f"balanced={report.final_instrument['balanced_accuracy_weight']:.2f},"
        f"f1={report.final_instrument['f1_macro_weight']:.2f},"
        f"worst_class={report.final_instrument['worst_class_recall_weight']:.2f},"
        f"noise={report.final_instrument['noise_stress_weight']:.2f},"
        f"dropout={report.final_instrument['feature_dropout_stress_weight']:.2f},"
        f"stability={report.final_instrument['ranking_stability_weight']:.2f}"
    )
    for result in report.dataset_results:
        state = "PASS" if result.passed else "FAIL"
        selected = result.selected_candidate.get("name")
        clean = result.clean_test_metrics["balanced_accuracy"]
        stress = result.stress_test_metrics["stress_floor_balanced_accuracy"]
        print(
            f"{state} {result.task_name} selected={selected} "
            f"clean_test_balanced_accuracy={clean:.3f} "
            f"stress_floor_balanced_accuracy={stress:.3f}"
        )
    for gate in report.gates:
        state = "PASS" if gate.passed else "FAIL"
        print(f"{state} gate:{gate.name} score={gate.score:.3f}")
    print(f"report={output}")
    return 0 if report.agi_claim_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
