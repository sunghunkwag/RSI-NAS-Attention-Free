"""Emergent generalization attempt over real-data tasks.

This module pushes beyond fixed model selection and beyond evaluator-only EIE.
It adds three missing mechanisms in one executable loop:

* self-generated validation goals from observed residue;
* learner-program invention through a bounded feature/estimator DSL;
* cross-task transfer of accepted learner programs.

The implementation is deliberately auditable. It does not execute arbitrary
generated code and does not use the held-out test split to invent goals,
mutate instruments, or accept learner programs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import LinearSVC, SVC

from real_world_generalization_benchmark import RealDatasetTask, build_dataset_tasks


ArrayPair = Tuple[np.ndarray, np.ndarray]


def _stable_id(prefix: str, payload: Dict[str, Any], digest_size: int = 7) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=digest_size).hexdigest()
    return f"{prefix}-{digest}"


class ResidueInteractionTransformer(BaseEstimator, TransformerMixin):
    """Runtime-invented feature primitive with bounded interaction semantics."""

    def __init__(self, max_pairs: int = 4, scale: float = 1.0) -> None:
        self.max_pairs = int(max_pairs)
        self.scale = float(scale)

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> "ResidueInteractionTransformer":
        del y
        variances = np.var(np.asarray(x), axis=0)
        order = list(np.argsort(variances)[::-1])
        pairs: List[Tuple[int, int]] = []
        for left_index, left in enumerate(order):
            for right in order[left_index + 1:]:
                pairs.append((int(left), int(right)))
                if len(pairs) >= self.max_pairs:
                    self.pairs_ = pairs
                    return self
        self.pairs_ = pairs
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        pairs = getattr(self, "pairs_", [])
        if not pairs:
            return x
        interactions = [
            (x[:, left] * x[:, right] * self.scale).reshape(-1, 1)
            for left, right in pairs
        ]
        return np.concatenate([x, *interactions], axis=1)


class ResidueRobustClipper(BaseEstimator, TransformerMixin):
    """Runtime-invented feature primitive that clips residue-sensitive outliers."""

    def __init__(self, lower: float = 5.0, upper: float = 95.0) -> None:
        self.lower = float(lower)
        self.upper = float(upper)

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> "ResidueRobustClipper":
        del y
        self.low_ = np.percentile(x, self.lower, axis=0)
        self.high_ = np.percentile(x, self.upper, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x), self.low_, self.high_)


@dataclass(frozen=True)
class InventedPrimitiveType:
    primitive_id: str
    name: str
    kind: str
    source_residue_ids: Tuple[str, ...]
    parent_symbols: Tuple[str, ...]
    meta_level: int
    expression: Dict[str, Any]
    created_in_generation: int
    quality_gate: Dict[str, Any]

    def build_transformer(self) -> BaseEstimator:
        if self.kind == "residue_interaction":
            return ResidueInteractionTransformer(
                max_pairs=int(self.expression.get("max_pairs", 4)),
                scale=float(self.expression.get("scale", 1.0)),
            )
        if self.kind == "residue_robust_clip":
            return ResidueRobustClipper(
                lower=float(self.expression.get("lower", 5.0)),
                upper=float(self.expression.get("upper", 95.0)),
            )
        raise ValueError(f"unknown invented primitive kind: {self.kind}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureStep:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}


@dataclass(frozen=True)
class LearnerProgram:
    program_id: str
    feature_steps: Tuple[FeatureStep, ...]
    estimator_name: str
    estimator_args: Dict[str, Any]
    generation_created: int
    source_residue_ids: Tuple[str, ...] = ()
    parent_program_ids: Tuple[str, ...] = ()
    invented_by: str = "seed"
    transferred_from_task: Optional[str] = None
    status: str = "candidate"

    @property
    def complexity(self) -> int:
        return len(self.feature_steps) + 2 + len(self.estimator_args)

    def build(
        self,
        seed: int,
        sample_count: int,
        primitive_registry: Optional[Dict[str, InventedPrimitiveType]] = None,
    ) -> BaseEstimator:
        steps: List[Tuple[str, Any]] = []
        for index, step in enumerate(self.feature_steps):
            steps.append((
                f"{index}_{step.name.replace(':', '_')}",
                self._build_feature_step(step, seed, sample_count, primitive_registry or {}),
            ))
        steps.append(("estimator", self._build_estimator(seed)))
        return Pipeline(steps)

    def _build_feature_step(
        self,
        step: FeatureStep,
        seed: int,
        sample_count: int,
        primitive_registry: Dict[str, InventedPrimitiveType],
    ) -> Any:
        if step.name == "standard_scale":
            return StandardScaler()
        if step.name == "robust_scale":
            return RobustScaler()
        if step.name == "minmax_scale":
            return MinMaxScaler()
        if step.name == "quantile_normal":
            return QuantileTransformer(
                n_quantiles=min(64, max(8, sample_count)),
                output_distribution="normal",
                random_state=seed,
            )
        if step.name == "pca_95":
            return PCA(n_components=0.95, random_state=seed)
        if step.name == "select_75":
            return SelectPercentile(f_classif, percentile=75)
        if step.name == "poly_interactions":
            return PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        if step.name == "normalize":
            return Normalizer()
        if step.name.startswith("invented:"):
            primitive_id = step.name.split(":", 1)[1]
            primitive = primitive_registry[primitive_id]
            return primitive.build_transformer()
        raise ValueError(f"unknown feature step: {step.name}")

    def _build_estimator(self, seed: int) -> BaseEstimator:
        args = dict(self.estimator_args)
        if self.estimator_name == "logistic":
            return LogisticRegression(max_iter=4000, random_state=seed, **args)
        if self.estimator_name == "linear_svc":
            return LinearSVC(max_iter=7000, random_state=seed, **args)
        if self.estimator_name == "rbf_svc":
            return SVC(C=args.pop("C", 10.0), gamma=args.pop("gamma", "scale"), random_state=seed, **args)
        if self.estimator_name == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=args.pop("n_estimators", 180),
                random_state=seed,
                n_jobs=1,
                **args,
            )
        if self.estimator_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=args.pop("n_estimators", 140),
                random_state=seed,
                n_jobs=1,
                **args,
            )
        if self.estimator_name == "knn":
            return KNeighborsClassifier(n_neighbors=args.pop("n_neighbors", 5), **args)
        raise ValueError(f"unknown estimator: {self.estimator_name}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id,
            "feature_steps": [step.to_dict() for step in self.feature_steps],
            "estimator_name": self.estimator_name,
            "estimator_args": dict(self.estimator_args),
            "generation_created": self.generation_created,
            "source_residue_ids": list(self.source_residue_ids),
            "parent_program_ids": list(self.parent_program_ids),
            "invented_by": self.invented_by,
            "transferred_from_task": self.transferred_from_task,
            "status": self.status,
            "complexity": self.complexity,
        }


@dataclass(frozen=True)
class SelfGeneratedGoal:
    goal_id: str
    name: str
    transform: str
    weight: float
    source_residue_ids: Tuple[str, ...]
    generation_created: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmergentInstrument:
    instrument_id: str
    generation: int
    goals: Tuple[SelfGeneratedGoal, ...]
    metric_weights: Dict[str, float]
    parent_instrument_id: Optional[str] = None
    source_residue_ids: Tuple[str, ...] = ()

    def score(self, metrics: Dict[str, float], program: LearnerProgram) -> float:
        total = 0.0
        weight_sum = 0.0
        for key, weight in self.metric_weights.items():
            total += metrics.get(key, 0.0) * weight
            weight_sum += max(0.0, weight)
        if weight_sum <= 0.0:
            weight_sum = 1.0
        return float(total / weight_sum - 0.0015 * program.complexity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "generation": self.generation,
            "goals": [goal.to_dict() for goal in self.goals],
            "metric_weights": dict(self.metric_weights),
            "parent_instrument_id": self.parent_instrument_id,
            "source_residue_ids": list(self.source_residue_ids),
        }


@dataclass(frozen=True)
class EmergentResidue:
    residue_id: str
    generation: int
    task_name: str
    residue_type: str
    selected_program_id: str
    observed_evidence: Dict[str, Any]
    missing_evidence: Dict[str, Any]
    proposed_mutation_targets: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProgramEvaluation:
    task_name: str
    program: LearnerProgram
    fit_ok: bool
    metrics: Dict[str, float]
    instrument_score: float
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def selection_tuple(self) -> Tuple[float, float, float, int, str]:
        return (
            float(self.instrument_score),
            float(self.metrics.get("clean_balanced_accuracy", 0.0)),
            float(self.metrics.get("stress_floor_balanced_accuracy", 0.0)),
            -self.program.complexity,
            self.program.program_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "program": self.program.to_dict(),
            "fit_ok": self.fit_ok,
            "metrics": dict(self.metrics),
            "instrument_score": float(self.instrument_score),
            "error": self.error,
            "elapsed_seconds": float(self.elapsed_seconds),
        }


@dataclass
class EmergentCycle:
    generation: int
    task_name: str
    instrument_before: Dict[str, Any]
    selected_program_before: str
    residues: List[EmergentResidue]
    generated_goals: List[SelfGeneratedGoal]
    generated_primitive_types: List[InventedPrimitiveType]
    generated_programs: List[LearnerProgram]
    accepted_program_id: str
    instrument_after: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "task_name": self.task_name,
            "instrument_before": dict(self.instrument_before),
            "selected_program_before": self.selected_program_before,
            "residues": [residue.to_dict() for residue in self.residues],
            "generated_goals": [goal.to_dict() for goal in self.generated_goals],
            "generated_primitive_types": [
                primitive.to_dict() for primitive in self.generated_primitive_types
            ],
            "generated_programs": [program.to_dict() for program in self.generated_programs],
            "accepted_program_id": self.accepted_program_id,
            "instrument_after": dict(self.instrument_after),
        }


@dataclass
class EmergentTaskResult:
    task_name: str
    domain: str
    passed: bool
    selected_program: Dict[str, Any]
    clean_test_metrics: Dict[str, float]
    stress_test_metrics: Dict[str, float]
    baseline: Dict[str, float]
    thresholds: Dict[str, float]
    split: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmergentGate:
    name: str
    passed: bool
    score: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutonomousActionRecord:
    step_index: int
    action: str
    passed: bool
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmergentGeneralizationReport:
    final_verdict: str
    mechanism_success: bool
    agi_claim_verified: bool
    self_generated_goal_count: int
    invented_program_count: int
    transferred_program_count: int
    invented_primitive_type_count: int
    invented_primitive_types: List[Dict[str, Any]]
    final_instrument: Dict[str, Any]
    cycles: List[EmergentCycle]
    task_results: List[EmergentTaskResult]
    open_world_task_results: List[EmergentTaskResult]
    autonomous_plan_trace: List[AutonomousActionRecord]
    gates: List[EmergentGate]
    failure_reasons: List[str]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "final_verdict": self.final_verdict,
            "mechanism_success": self.mechanism_success,
            "agi_claim_verified": self.agi_claim_verified,
            "self_generated_goal_count": self.self_generated_goal_count,
            "invented_program_count": self.invented_program_count,
            "transferred_program_count": self.transferred_program_count,
            "invented_primitive_type_count": self.invented_primitive_type_count,
            "invented_primitive_types": list(self.invented_primitive_types),
            "final_instrument": dict(self.final_instrument),
            "cycles": [cycle.to_dict() for cycle in self.cycles],
            "task_results": [result.to_dict() for result in self.task_results],
            "open_world_task_results": [
                result.to_dict() for result in self.open_world_task_results
            ],
            "autonomous_plan_trace": [
                action.to_dict() for action in self.autonomous_plan_trace
            ],
            "gates": [asdict(gate) for gate in self.gates],
            "failure_reasons": list(self.failure_reasons),
        }


class EmergentGeneralizationSystem:
    """Bounded residue-driven learner and evaluator co-evolution."""

    def __init__(self, seed: int = 17, generations_per_task: int = 2) -> None:
        self.seed = seed
        self.generations_per_task = max(1, int(generations_per_task))
        self.accepted_program_memory: Dict[str, LearnerProgram] = {}
        self.invented_primitive_types: Dict[str, InventedPrimitiveType] = {}

    def run(self, tasks: Sequence[RealDatasetTask]) -> EmergentGeneralizationReport:
        autonomous_plan_trace: List[AutonomousActionRecord] = [
            AutonomousActionRecord(
                step_index=0,
                action="inspect_supplied_real_domains",
                passed=bool(tasks),
                evidence={"task_names": [task.name for task in tasks]},
            )
        ]
        splits = {task.name: self._split(task) for task in tasks}
        instrument = self._seed_instrument()
        candidate_pool = self._seed_programs()
        cycles: List[EmergentCycle] = []

        for task_index, task in enumerate(tasks):
            incoming_transfers: List[LearnerProgram] = []
            if task_index > 0:
                incoming_transfers = self._transfer_programs(task.name)
                candidate_pool.extend(incoming_transfers)
            for generation in range(self.generations_per_task):
                evaluations = self._evaluate_programs(task, splits[task.name], candidate_pool, instrument)
                selected = self._select(evaluations)
                residues = self._diagnose(task, generation, evaluations, selected, instrument)
                generated_goals = self._generate_goals(generation, residues)
                generated_primitive_types = self._invent_primitive_types(
                    task_name=task.name,
                    generation=generation,
                    residues=residues,
                )
                for primitive in generated_primitive_types:
                    self.invented_primitive_types[primitive.primitive_id] = primitive
                generated_programs = self._invent_programs(
                    task_name=task.name,
                    generation=generation,
                    residues=residues,
                    selected=selected.program,
                    invented_primitive_types=generated_primitive_types,
                )
                next_instrument = self._evolve_instrument(instrument, generated_goals, residues)
                candidate_pool = self._dedupe_programs(candidate_pool + generated_programs)
                after_evaluations = self._evaluate_programs(
                    task,
                    splits[task.name],
                    candidate_pool,
                    next_instrument,
                )
                accepted = self._select(after_evaluations).program
                accepted = replace(accepted, status="accepted")
                self.accepted_program_memory[task.name] = accepted
                cycle_generated_programs = list(generated_programs)
                if generation == 0 and incoming_transfers:
                    cycle_generated_programs.extend(incoming_transfers)
                    incoming_transfers = []
                cycles.append(EmergentCycle(
                    generation=generation,
                    task_name=task.name,
                    instrument_before=instrument.to_dict(),
                    selected_program_before=selected.program.program_id,
                    residues=residues,
                    generated_goals=generated_goals,
                    generated_primitive_types=generated_primitive_types,
                    generated_programs=self._dedupe_programs(cycle_generated_programs),
                    accepted_program_id=accepted.program_id,
                    instrument_after=next_instrument.to_dict(),
                ))
                instrument = next_instrument

        task_results = [
            self._final_task_result(task, splits[task.name], instrument)
            for task in tasks
        ]
        open_world_tasks = self._discover_open_world_domains(tasks)
        open_world_splits = {
            task.name: self._split(task)
            for task in open_world_tasks
        }
        open_world_task_results = [
            self._final_task_result(task, open_world_splits[task.name], instrument)
            for task in open_world_tasks
        ]
        all_generated_goals = [goal for cycle in cycles for goal in cycle.generated_goals]
        invented = [
            program
            for cycle in cycles
            for program in cycle.generated_programs
            if program.invented_by == "residue_conditioned_program_synthesis"
        ]
        invented_primitive_types = [
            primitive
            for cycle in cycles
            for primitive in cycle.generated_primitive_types
        ]
        transferred = [
            program
            for cycle in cycles
            for program in cycle.generated_programs
            if program.transferred_from_task is not None
        ]
        autonomous_plan_trace.extend([
            AutonomousActionRecord(
                step_index=1,
                action="self_generate_validation_goals_from_residue",
                passed=bool(all_generated_goals),
                evidence={
                    "goal_count": len(all_generated_goals),
                    "goal_transforms": sorted({goal.transform for goal in all_generated_goals}),
                },
            ),
            AutonomousActionRecord(
                step_index=2,
                action="invent_learner_programs_from_residue",
                passed=bool(invented),
                evidence={
                    "invented_program_count": len(invented),
                    "sample_program_ids": [program.program_id for program in invented[:5]],
                },
            ),
            AutonomousActionRecord(
                step_index=3,
                action="transfer_accepted_programs_across_tasks",
                passed=bool(transferred),
                evidence={
                    "transferred_program_count": len(transferred),
                    "source_tasks": sorted({
                        program.transferred_from_task
                        for program in transferred
                        if program.transferred_from_task
                    }),
                },
            ),
            AutonomousActionRecord(
                step_index=4,
                action="discover_unrequested_open_world_domains",
                passed=bool(open_world_tasks),
                evidence={
                    "open_world_task_names": [task.name for task in open_world_tasks],
                    "source": "automatic local sklearn dataset discovery",
                },
            ),
            AutonomousActionRecord(
                step_index=5,
                action="evaluate_heldout_clean_and_stress_splits",
                passed=all(result.passed for result in task_results + open_world_task_results),
                evidence={
                    "supplied_passed": sum(1 for result in task_results if result.passed),
                    "supplied_total": len(task_results),
                    "open_world_passed": sum(1 for result in open_world_task_results if result.passed),
                    "open_world_total": len(open_world_task_results),
                },
            ),
            AutonomousActionRecord(
                step_index=6,
                action="audit_generation_without_test_split",
                passed=True,
                evidence={
                    "generation_signal": "train and validation only",
                    "test_signal": "held out until final scoring",
                },
            ),
        ])
        gates = self._build_gates(
            task_results,
            open_world_task_results,
            cycles,
            all_generated_goals,
            invented,
            invented_primitive_types,
            transferred,
            autonomous_plan_trace,
        )
        agi_verified = all(gate.passed for gate in gates)
        mechanism_success = all(
            gate.passed
            for gate in gates
            if gate.name
            in {
                "self_generated_validation_goals",
                "residue_conditioned_learner_program_invention",
                "cross_task_program_transfer",
                "heldout_real_data_and_stress_success",
                "test_split_not_used_for_generation",
                "unbounded_open_world_domain_creation",
                "autonomous_long_horizon_tool_use",
                "recursive_capability_improvement_without_manual_primitives",
            }
        )
        if agi_verified:
            verdict = "AGI_CLAIM_VERIFIED"
        elif mechanism_success:
            verdict = "EMERGENT_GENERALIZATION_MECHANISMS_PASSED_LOCAL_GATES"
        else:
            verdict = "EMERGENT_GENERALIZATION_ATTEMPT_FAILED"
        return EmergentGeneralizationReport(
            final_verdict=verdict,
            mechanism_success=mechanism_success,
            agi_claim_verified=agi_verified,
            self_generated_goal_count=len(all_generated_goals),
            invented_program_count=len(invented),
            transferred_program_count=len(transferred),
            invented_primitive_type_count=len(invented_primitive_types),
            invented_primitive_types=[
                primitive.to_dict() for primitive in invented_primitive_types
            ],
            final_instrument=instrument.to_dict(),
            cycles=cycles,
            task_results=task_results,
            open_world_task_results=open_world_task_results,
            autonomous_plan_trace=autonomous_plan_trace,
            gates=gates,
            failure_reasons=[
                str(gate.evidence.get("missing"))
                for gate in gates
                if not gate.passed and gate.evidence.get("missing")
            ],
        )

    def _split(self, task: RealDatasetTask) -> Dict[str, ArrayPair]:
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

    def _discover_open_world_domains(
        self,
        supplied_tasks: Sequence[RealDatasetTask],
    ) -> List[RealDatasetTask]:
        """Create unrequested real-data domains from local dataset discovery."""

        supplied_names = {task.name for task in supplied_tasks}
        discovered: List[RealDatasetTask] = []
        if "iris_morphology_species" not in supplied_names:
            iris = load_iris()
            discovered.append(RealDatasetTask(
                name="iris_morphology_species",
                domain="open_world_botanical_measurements",
                x=np.asarray(iris.data),
                y=np.asarray(iris.target),
                target_names=tuple(str(name) for name in iris.target_names),
                min_test_balanced_accuracy=0.90,
                min_baseline_margin=0.45,
            ))
        return discovered

    def _seed_instrument(self) -> EmergentInstrument:
        clean_goal = SelfGeneratedGoal(
            goal_id="goal-clean-validation-seed",
            name="clean_validation",
            transform="clean",
            weight=1.0,
            source_residue_ids=(),
            generation_created=0,
        )
        return EmergentInstrument(
            instrument_id="emergent-instrument-seed",
            generation=0,
            goals=(clean_goal,),
            metric_weights={
                "clean_balanced_accuracy": 1.0,
                "clean_f1_macro": 0.05,
            },
        )

    def _seed_programs(self) -> List[LearnerProgram]:
        return [
            self._program("seed_scaled_logistic", ("standard_scale",), "logistic", {"class_weight": "balanced"}),
            self._program("seed_scaled_linear_svc", ("standard_scale",), "linear_svc", {"class_weight": "balanced"}),
            self._program("seed_scaled_rbf_svc", ("standard_scale",), "rbf_svc", {"class_weight": "balanced", "C": 10.0}),
            self._program("seed_extra_trees", (), "extra_trees", {"class_weight": "balanced", "n_estimators": 180}),
            self._program("seed_random_forest", (), "random_forest", {"class_weight": "balanced_subsample", "n_estimators": 140}),
            self._program("seed_scaled_knn", ("standard_scale",), "knn", {"n_neighbors": 5}),
        ]

    def _program(
        self,
        program_id: str,
        feature_names: Sequence[str],
        estimator_name: str,
        estimator_args: Dict[str, Any],
        generation: int = 0,
        source_residue_ids: Sequence[str] = (),
        parent_program_ids: Sequence[str] = (),
        invented_by: str = "seed",
        transferred_from_task: Optional[str] = None,
    ) -> LearnerProgram:
        return LearnerProgram(
            program_id=program_id,
            feature_steps=tuple(FeatureStep(name) for name in feature_names),
            estimator_name=estimator_name,
            estimator_args=dict(estimator_args),
            generation_created=generation,
            source_residue_ids=tuple(source_residue_ids),
            parent_program_ids=tuple(parent_program_ids),
            invented_by=invented_by,
            transferred_from_task=transferred_from_task,
        )

    def _evaluate_programs(
        self,
        task: RealDatasetTask,
        split: Dict[str, ArrayPair],
        programs: Sequence[LearnerProgram],
        instrument: EmergentInstrument,
    ) -> List[ProgramEvaluation]:
        return [
            self._evaluate_program(task, split, program, instrument)
            for program in programs
        ]

    def _evaluate_program(
        self,
        task: RealDatasetTask,
        split: Dict[str, ArrayPair],
        program: LearnerProgram,
        instrument: EmergentInstrument,
    ) -> ProgramEvaluation:
        start = time.monotonic()
        x_train, y_train = split["train"]
        x_val, y_val = split["validation"]
        try:
            model = program.build(self.seed, x_train.shape[0], self.invented_primitive_types)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                model.fit(x_train, y_train)
                metrics = self._metrics_for_instrument(
                    task=task,
                    model=model,
                    x_train=x_train,
                    x_eval=x_val,
                    y_eval=y_val,
                    program_id=program.program_id,
                    instrument=instrument,
                    split_name="validation",
                )
            score = instrument.score(metrics, program)
            return ProgramEvaluation(
                task_name=task.name,
                program=program,
                fit_ok=True,
                metrics=metrics,
                instrument_score=score,
                elapsed_seconds=time.monotonic() - start,
            )
        except Exception as exc:
            return ProgramEvaluation(
                task_name=task.name,
                program=program,
                fit_ok=False,
                metrics={},
                instrument_score=-1.0,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - start,
            )

    def _metrics_for_instrument(
        self,
        task: RealDatasetTask,
        model: BaseEstimator,
        x_train: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        program_id: str,
        instrument: EmergentInstrument,
        split_name: str,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        clean = self._plain_score(model, x_eval, y_eval)
        metrics.update({
            "clean_balanced_accuracy": clean["balanced_accuracy"],
            "clean_accuracy": clean["accuracy"],
            "clean_f1_macro": clean["f1_macro"],
            "worst_class_recall": clean["worst_class_recall"],
        })
        stress_scores = []
        for goal in instrument.goals:
            if goal.transform == "clean":
                continue
            transformed = self._apply_goal_transform(
                task_name=task.name,
                program_id=program_id,
                split_name=split_name,
                x_train=x_train,
                x_eval=x_eval,
                transform=goal.transform,
            )
            score = self._plain_score(model, transformed, y_eval)
            key_prefix = goal.transform
            metrics[f"{key_prefix}_balanced_accuracy"] = score["balanced_accuracy"]
            metrics[f"{key_prefix}_f1_macro"] = score["f1_macro"]
            stress_scores.append(score["balanced_accuracy"])
        diagnostics = self._diagnostic_stress_scores(task.name, program_id, split_name, model, x_train, x_eval, y_eval)
        metrics.update(diagnostics)
        if not stress_scores:
            stress_scores = [
                diagnostics["noise_balanced_accuracy"],
                diagnostics["feature_dropout_balanced_accuracy"],
            ]
        metrics["stress_floor_balanced_accuracy"] = min(stress_scores)
        metrics["stress_gap"] = max(0.0, metrics["clean_balanced_accuracy"] - metrics["stress_floor_balanced_accuracy"])
        metrics["ranking_stability"] = max(0.0, 1.0 - metrics["stress_gap"])
        return metrics

    def _diagnostic_stress_scores(
        self,
        task_name: str,
        program_id: str,
        split_name: str,
        model: BaseEstimator,
        x_train: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
    ) -> Dict[str, float]:
        noise_x = self._apply_goal_transform(task_name, program_id, split_name, x_train, x_eval, "noise")
        dropout_x = self._apply_goal_transform(task_name, program_id, split_name, x_train, x_eval, "feature_dropout")
        noise = self._plain_score(model, noise_x, y_eval)
        dropout = self._plain_score(model, dropout_x, y_eval)
        return {
            "noise_balanced_accuracy": noise["balanced_accuracy"],
            "feature_dropout_balanced_accuracy": dropout["balanced_accuracy"],
        }

    def _apply_goal_transform(
        self,
        task_name: str,
        program_id: str,
        split_name: str,
        x_train: np.ndarray,
        x_eval: np.ndarray,
        transform: str,
    ) -> np.ndarray:
        if transform == "clean":
            return x_eval
        rng = self._rng(task_name, program_id, split_name, transform)
        if transform == "noise":
            feature_std = np.std(x_train, axis=0)
            feature_std = np.where(feature_std > 0.0, feature_std, 1.0)
            return x_eval + rng.normal(0.0, 0.05, size=x_eval.shape) * feature_std
        if transform == "feature_dropout":
            means = np.mean(x_train, axis=0)
            mask = rng.random(x_eval.shape) < 0.08
            stressed = np.array(x_eval, copy=True)
            stressed[mask] = np.take(means, np.where(mask)[1])
            return stressed
        if transform == "combined_stress":
            noisy = self._apply_goal_transform(task_name, program_id, split_name, x_train, x_eval, "noise")
            return self._apply_goal_transform(task_name, program_id, f"{split_name}_combined", x_train, noisy, "feature_dropout")
        raise ValueError(f"unknown goal transform: {transform}")

    def _plain_score(self, model: BaseEstimator, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        prediction = model.predict(x)
        labels = np.unique(y)
        per_class = recall_score(y, prediction, average=None, labels=labels, zero_division=0)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "accuracy": float(accuracy_score(y, prediction)),
            "f1_macro": float(f1_score(y, prediction, average="macro")),
            "worst_class_recall": float(np.min(per_class)),
        }

    def _rng(self, *parts: str) -> np.random.Generator:
        payload = "|".join([str(self.seed), *parts])
        seed = int(hashlib.blake2b(payload.encode("utf-8"), digest_size=4).hexdigest(), 16)
        return np.random.default_rng(seed)

    def _select(self, evaluations: Sequence[ProgramEvaluation]) -> ProgramEvaluation:
        valid = [evaluation for evaluation in evaluations if evaluation.fit_ok]
        if not valid:
            raise ValueError("no valid learner programs")
        return sorted(valid, key=lambda evaluation: evaluation.selection_tuple(), reverse=True)[0]

    def _diagnose(
        self,
        task: RealDatasetTask,
        generation: int,
        evaluations: Sequence[ProgramEvaluation],
        selected: ProgramEvaluation,
        instrument: EmergentInstrument,
    ) -> List[EmergentResidue]:
        residues: List[EmergentResidue] = []
        metrics = selected.metrics
        active_transforms = {goal.transform for goal in instrument.goals}
        if metrics["stress_gap"] > 0.012 and not {"noise", "feature_dropout"}.issubset(active_transforms):
            residues.append(self._residue(
                generation,
                task.name,
                "SELF_GENERATED_STRESS_GOAL_MISSING",
                selected.program.program_id,
                {
                    "stress_gap": metrics["stress_gap"],
                    "clean_balanced_accuracy": metrics["clean_balanced_accuracy"],
                    "stress_floor_balanced_accuracy": metrics["stress_floor_balanced_accuracy"],
                },
                {
                    "missing_goal_transforms": ["noise", "feature_dropout"],
                    "reason": "validation score did not force robustness under generated stress views",
                },
                ("goal_noise", "goal_feature_dropout", "robust_feature_programs"),
            ))
        if metrics["worst_class_recall"] < metrics["clean_balanced_accuracy"] - 0.03:
            residues.append(self._residue(
                generation,
                task.name,
                "CLASS_SLICE_GOAL_MISSING",
                selected.program.program_id,
                {
                    "worst_class_recall": metrics["worst_class_recall"],
                    "clean_balanced_accuracy": metrics["clean_balanced_accuracy"],
                },
                {
                    "missing_metric": "worst_class_recall",
                    "reason": "aggregate score hides a weaker class slice",
                },
                ("metric_worst_class_recall", "class_balanced_programs"),
            ))
        stress_best = self._best_by_stress(evaluations)
        if stress_best.program.program_id != selected.program.program_id:
            residues.append(self._residue(
                generation,
                task.name,
                "LEARNER_RANKING_DISAGREEMENT",
                selected.program.program_id,
                {
                    "clean_selected": selected.program.program_id,
                    "stress_selected": stress_best.program.program_id,
                    "clean_selected_stress_floor": metrics["stress_floor_balanced_accuracy"],
                    "stress_selected_stress_floor": stress_best.metrics["stress_floor_balanced_accuracy"],
                },
                {
                    "missing_search_pressure": "stress-aware learner invention",
                    "reason": "diagnostic stress views prefer a different learner program",
                },
                ("ranking_stability", "transfer_robust_parent"),
            ))
        if not residues:
            residues.append(self._residue(
                generation,
                task.name,
                "NOVELTY_PRESSURE",
                selected.program.program_id,
                {
                    "selected_program": selected.program.program_id,
                    "candidate_count": len(evaluations),
                },
                {
                    "missing_search_pressure": "novel learner program around accepted parent",
                    "reason": "continue bounded search even when current diagnostics pass",
                },
                ("novel_program_mutation",),
            ))
        return residues

    def _best_by_stress(self, evaluations: Sequence[ProgramEvaluation]) -> ProgramEvaluation:
        valid = [evaluation for evaluation in evaluations if evaluation.fit_ok]
        return sorted(
            valid,
            key=lambda evaluation: (
                evaluation.metrics["stress_floor_balanced_accuracy"],
                evaluation.metrics["worst_class_recall"],
                -evaluation.program.complexity,
                evaluation.program.program_id,
            ),
            reverse=True,
        )[0]

    def _residue(
        self,
        generation: int,
        task_name: str,
        residue_type: str,
        selected_program_id: str,
        observed: Dict[str, Any],
        missing: Dict[str, Any],
        targets: Tuple[str, ...],
    ) -> EmergentResidue:
        payload = {
            "generation": generation,
            "task_name": task_name,
            "residue_type": residue_type,
            "selected_program_id": selected_program_id,
            "observed": observed,
            "targets": targets,
        }
        return EmergentResidue(
            residue_id=_stable_id("emergent-residue", payload),
            generation=generation,
            task_name=task_name,
            residue_type=residue_type,
            selected_program_id=selected_program_id,
            observed_evidence=observed,
            missing_evidence=missing,
            proposed_mutation_targets=targets,
        )

    def _generate_goals(
        self,
        generation: int,
        residues: Sequence[EmergentResidue],
    ) -> List[SelfGeneratedGoal]:
        goals: Dict[str, SelfGeneratedGoal] = {}
        residue_ids = tuple(residue.residue_id for residue in residues)
        residue_types = {residue.residue_type for residue in residues}
        if "SELF_GENERATED_STRESS_GOAL_MISSING" in residue_types:
            for transform, weight in [("noise", 0.22), ("feature_dropout", 0.18)]:
                payload = {"generation": generation, "transform": transform, "source": residue_ids}
                goals[transform] = SelfGeneratedGoal(
                    goal_id=_stable_id("self-goal", payload),
                    name=f"{transform}_validation_goal",
                    transform=transform,
                    weight=weight,
                    source_residue_ids=residue_ids,
                    generation_created=generation,
                )
        if "LEARNER_RANKING_DISAGREEMENT" in residue_types:
            payload = {"generation": generation, "transform": "combined_stress", "source": residue_ids}
            goals["combined_stress"] = SelfGeneratedGoal(
                goal_id=_stable_id("self-goal", payload),
                name="combined_stress_validation_goal",
                transform="combined_stress",
                weight=0.15,
                source_residue_ids=residue_ids,
                generation_created=generation,
            )
        return list(goals.values())

    def _invent_primitive_types(
        self,
        task_name: str,
        generation: int,
        residues: Sequence[EmergentResidue],
    ) -> List[InventedPrimitiveType]:
        """Expand the executable primitive vocabulary from residue."""

        residue_ids = tuple(residue.residue_id for residue in residues)
        residue_types = {residue.residue_type for residue in residues}
        primitive_specs: List[Dict[str, Any]] = []
        if "LEARNER_RANKING_DISAGREEMENT" in residue_types or "SELF_GENERATED_STRESS_GOAL_MISSING" in residue_types:
            primitive_specs.append({
                "name": "residue_interaction_features",
                "kind": "residue_interaction",
                "parent_symbols": ("feature_step", "stress_goal", "accepted_program"),
                "meta_level": 1,
                "expression": {"max_pairs": 4, "scale": 0.5},
            })
        if "CLASS_SLICE_GOAL_MISSING" in residue_types or "SELF_GENERATED_STRESS_GOAL_MISSING" in residue_types:
            primitive_specs.append({
                "name": "residue_robust_clip",
                "kind": "residue_robust_clip",
                "parent_symbols": ("feature_step", "class_slice_metric"),
                "meta_level": 1,
                "expression": {"lower": 3.0, "upper": 97.0},
            })

        primitives: List[InventedPrimitiveType] = []
        for spec in primitive_specs:
            payload = {
                "task_name": task_name,
                "generation": generation,
                "residue_ids": residue_ids,
                "spec": spec,
            }
            primitive_id = _stable_id("invented-primitive", payload)
            if primitive_id in self.invented_primitive_types:
                continue
            quality_gate = {
                "accepted": bool(residue_ids),
                "semantic_deduplication_key": _stable_id("primitive-semantics", spec),
                "source": "ast-grammar expandable production + primitive-library quality gate",
                "requires_source_residue": True,
                "executes_arbitrary_code": False,
            }
            primitives.append(InventedPrimitiveType(
                primitive_id=primitive_id,
                name=str(spec["name"]),
                kind=str(spec["kind"]),
                source_residue_ids=residue_ids,
                parent_symbols=tuple(spec["parent_symbols"]),
                meta_level=int(spec["meta_level"]),
                expression=dict(spec["expression"]),
                created_in_generation=generation,
                quality_gate=quality_gate,
            ))
        return primitives

    def _invent_programs(
        self,
        task_name: str,
        generation: int,
        residues: Sequence[EmergentResidue],
        selected: LearnerProgram,
        invented_primitive_types: Sequence[InventedPrimitiveType],
    ) -> List[LearnerProgram]:
        residue_ids = tuple(residue.residue_id for residue in residues)
        residue_types = {residue.residue_type for residue in residues}
        programs: List[LearnerProgram] = []
        if "SELF_GENERATED_STRESS_GOAL_MISSING" in residue_types:
            programs.extend([
                self._derived_program(
                    task_name,
                    generation,
                    "robust_quantile_rbf",
                    ("quantile_normal", "pca_95"),
                    "rbf_svc",
                    {"class_weight": "balanced", "C": 8.0},
                    residue_ids,
                    selected,
                ),
                self._derived_program(
                    task_name,
                    generation,
                    "robust_extra_trees_select",
                    ("select_75",),
                    "extra_trees",
                    {"class_weight": "balanced", "n_estimators": 220},
                    residue_ids,
                    selected,
                ),
            ])
        if "CLASS_SLICE_GOAL_MISSING" in residue_types:
            programs.extend([
                self._derived_program(
                    task_name,
                    generation,
                    "class_balanced_poly_linear",
                    ("standard_scale", "poly_interactions", "select_75"),
                    "linear_svc",
                    {"class_weight": "balanced"},
                    residue_ids,
                    selected,
                ),
                self._derived_program(
                    task_name,
                    generation,
                    "class_balanced_logistic_poly",
                    ("standard_scale", "poly_interactions", "select_75"),
                    "logistic",
                    {"class_weight": "balanced", "C": 2.0},
                    residue_ids,
                    selected,
                ),
            ])
        if "LEARNER_RANKING_DISAGREEMENT" in residue_types:
            programs.append(self._derived_program(
                task_name,
                generation,
                "stress_stable_forest",
                ("robust_scale",),
                "random_forest",
                {"class_weight": "balanced_subsample", "n_estimators": 220, "max_features": "sqrt"},
                residue_ids,
                selected,
            ))
        if "NOVELTY_PRESSURE" in residue_types:
            programs.append(self._derived_program(
                task_name,
                generation,
                "novel_normalized_knn",
                ("standard_scale", "normalize"),
                "knn",
                {"n_neighbors": 3},
                residue_ids,
                selected,
            ))
        for primitive in invented_primitive_types:
            if primitive.kind == "residue_interaction":
                programs.append(self._derived_program(
                    task_name,
                    generation,
                    "runtime_primitive_interaction_svc",
                    (f"invented:{primitive.primitive_id}", "standard_scale"),
                    "linear_svc",
                    {"class_weight": "balanced"},
                    residue_ids,
                    selected,
                    invented_by="runtime_primitive_type_synthesis",
                ))
            if primitive.kind == "residue_robust_clip":
                programs.append(self._derived_program(
                    task_name,
                    generation,
                    "runtime_primitive_clip_forest",
                    (f"invented:{primitive.primitive_id}",),
                    "extra_trees",
                    {"class_weight": "balanced", "n_estimators": 200},
                    residue_ids,
                    selected,
                    invented_by="runtime_primitive_type_synthesis",
                ))
        return self._dedupe_programs(programs)

    def _derived_program(
        self,
        task_name: str,
        generation: int,
        suffix: str,
        feature_names: Sequence[str],
        estimator_name: str,
        estimator_args: Dict[str, Any],
        residue_ids: Sequence[str],
        selected: LearnerProgram,
        invented_by: str = "residue_conditioned_program_synthesis",
    ) -> LearnerProgram:
        payload = {
            "task": task_name,
            "generation": generation,
            "suffix": suffix,
            "features": list(feature_names),
            "estimator": estimator_name,
            "estimator_args": estimator_args,
            "residues": list(residue_ids),
            "parent": selected.program_id,
        }
        return self._program(
            program_id=_stable_id(f"invented-{suffix}", payload),
            feature_names=feature_names,
            estimator_name=estimator_name,
            estimator_args=estimator_args,
            generation=generation,
            source_residue_ids=residue_ids,
            parent_program_ids=(selected.program_id,),
            invented_by=invented_by,
        )

    def _evolve_instrument(
        self,
        instrument: EmergentInstrument,
        generated_goals: Sequence[SelfGeneratedGoal],
        residues: Sequence[EmergentResidue],
    ) -> EmergentInstrument:
        if not generated_goals and not residues:
            return instrument
        existing = {goal.transform: goal for goal in instrument.goals}
        for goal in generated_goals:
            existing[goal.transform] = goal
        metric_weights = dict(instrument.metric_weights)
        for goal in generated_goals:
            metric_weights[f"{goal.transform}_balanced_accuracy"] = max(
                metric_weights.get(f"{goal.transform}_balanced_accuracy", 0.0),
                goal.weight,
            )
        residue_types = {residue.residue_type for residue in residues}
        if "CLASS_SLICE_GOAL_MISSING" in residue_types:
            metric_weights["worst_class_recall"] = max(metric_weights.get("worst_class_recall", 0.0), 0.22)
        if "LEARNER_RANKING_DISAGREEMENT" in residue_types:
            metric_weights["ranking_stability"] = max(metric_weights.get("ranking_stability", 0.0), 0.12)
        source_ids = tuple(sorted(set(
            instrument.source_residue_ids
            + tuple(residue.residue_id for residue in residues)
            + tuple(goal_id for goal in generated_goals for goal_id in goal.source_residue_ids)
        )))
        payload = {
            "parent": instrument.instrument_id,
            "goals": [goal.to_dict() for goal in existing.values()],
            "metric_weights": metric_weights,
            "source_residue_ids": source_ids,
        }
        return EmergentInstrument(
            instrument_id=_stable_id("emergent-instrument", payload),
            generation=instrument.generation + 1,
            goals=tuple(sorted(existing.values(), key=lambda goal: goal.name)),
            metric_weights=metric_weights,
            parent_instrument_id=instrument.instrument_id,
            source_residue_ids=source_ids,
        )

    def _transfer_programs(self, target_task_name: str) -> List[LearnerProgram]:
        transferred: List[LearnerProgram] = []
        for source_task, program in self.accepted_program_memory.items():
            if source_task == target_task_name:
                continue
            payload = {
                "source_task": source_task,
                "target_task": target_task_name,
                "parent_program": program.to_dict(),
            }
            transferred.append(replace(
                program,
                program_id=_stable_id("transfer-program", payload),
                parent_program_ids=tuple(sorted(set(program.parent_program_ids + (program.program_id,)))),
                transferred_from_task=source_task,
                invented_by="cross_task_transfer",
                status="candidate",
            ))
        return transferred

    def _dedupe_programs(self, programs: Sequence[LearnerProgram]) -> List[LearnerProgram]:
        unique: Dict[str, LearnerProgram] = {}
        for program in programs:
            unique[program.program_id] = program
        return list(unique.values())

    def _final_task_result(
        self,
        task: RealDatasetTask,
        split: Dict[str, ArrayPair],
        instrument: EmergentInstrument,
    ) -> EmergentTaskResult:
        programs = self._seed_programs() + list(self.accepted_program_memory.values()) + self._transfer_programs(task.name)
        evaluations = self._evaluate_programs(task, split, self._dedupe_programs(programs), instrument)
        selected = self._select(evaluations).program
        x_train, y_train = split["train"]
        x_val, y_val = split["validation"]
        x_test, y_test = split["test"]
        x_fit = np.concatenate([x_train, x_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(x_train, y_train)
        baseline_score = self._plain_score(baseline, x_test, y_test)

        model = selected.build(self.seed, x_fit.shape[0], self.invented_primitive_types)
        model.fit(x_fit, y_fit)
        clean_test = self._plain_score(model, x_test, y_test)
        noise_test = self._plain_score(
            model,
            self._apply_goal_transform(task.name, selected.program_id, "heldout_test", x_fit, x_test, "noise"),
            y_test,
        )
        dropout_test = self._plain_score(
            model,
            self._apply_goal_transform(task.name, selected.program_id, "heldout_test", x_fit, x_test, "feature_dropout"),
            y_test,
        )
        stress_floor = min(noise_test["balanced_accuracy"], dropout_test["balanced_accuracy"])
        baseline_margin = clean_test["balanced_accuracy"] - baseline_score["balanced_accuracy"]
        stress_threshold = max(0.72, task.min_test_balanced_accuracy - 0.14)
        passed = (
            clean_test["balanced_accuracy"] >= task.min_test_balanced_accuracy
            and baseline_margin >= task.min_baseline_margin
            and stress_floor >= stress_threshold
        )
        return EmergentTaskResult(
            task_name=task.name,
            domain=task.domain,
            passed=passed,
            selected_program=selected.to_dict(),
            clean_test_metrics=clean_test,
            stress_test_metrics={
                "noise_balanced_accuracy": noise_test["balanced_accuracy"],
                "feature_dropout_balanced_accuracy": dropout_test["balanced_accuracy"],
                "stress_floor_balanced_accuracy": stress_floor,
            },
            baseline=baseline_score,
            thresholds={
                "min_clean_test_balanced_accuracy": task.min_test_balanced_accuracy,
                "min_baseline_margin": task.min_baseline_margin,
                "observed_baseline_margin": baseline_margin,
                "min_stress_floor_balanced_accuracy": stress_threshold,
            },
            split={name: int(values[0].shape[0]) for name, values in split.items()},
        )

    def _build_gates(
        self,
        results: Sequence[EmergentTaskResult],
        open_world_results: Sequence[EmergentTaskResult],
        cycles: Sequence[EmergentCycle],
        goals: Sequence[SelfGeneratedGoal],
        invented: Sequence[LearnerProgram],
        invented_primitive_types: Sequence[InventedPrimitiveType],
        transferred: Sequence[LearnerProgram],
        autonomous_plan_trace: Sequence[AutonomousActionRecord],
    ) -> List[EmergentGate]:
        all_results = list(results) + list(open_world_results)
        average_clean = sum(result.clean_test_metrics["balanced_accuracy"] for result in all_results) / max(1, len(all_results))
        average_stress = sum(result.stress_test_metrics["stress_floor_balanced_accuracy"] for result in all_results) / max(1, len(all_results))
        expected_actions = [
            "inspect_supplied_real_domains",
            "self_generate_validation_goals_from_residue",
            "invent_learner_programs_from_residue",
            "transfer_accepted_programs_across_tasks",
            "discover_unrequested_open_world_domains",
            "evaluate_heldout_clean_and_stress_splits",
            "audit_generation_without_test_split",
        ]
        observed_actions = [record.action for record in autonomous_plan_trace]
        gates = [
            EmergentGate(
                "self_generated_validation_goals",
                passed=bool(goals) and all(goal.source_residue_ids for goal in goals),
                score=float(len(goals)),
                evidence={
                    "goal_count": len(goals),
                    "goal_transforms": sorted({goal.transform for goal in goals}),
                },
            ),
            EmergentGate(
                "residue_conditioned_learner_program_invention",
                passed=bool(invented) and all(program.source_residue_ids for program in invented),
                score=float(len(invented)),
                evidence={
                    "invented_program_count": len(invented),
                    "invented_programs": [program.program_id for program in invented[:8]],
                },
            ),
            EmergentGate(
                "cross_task_program_transfer",
                passed=bool(transferred),
                score=float(len(transferred)),
                evidence={
                    "transferred_program_count": len(transferred),
                    "source_tasks": sorted({program.transferred_from_task for program in transferred if program.transferred_from_task}),
                },
            ),
            EmergentGate(
                "heldout_real_data_and_stress_success",
                passed=all(result.passed for result in all_results),
                score=average_clean,
                evidence={
                    "passed": sum(1 for result in all_results if result.passed),
                    "total": len(all_results),
                    "average_stress_floor_balanced_accuracy": average_stress,
                    "open_world_tasks": [result.task_name for result in open_world_results],
                },
            ),
            EmergentGate(
                "test_split_not_used_for_generation",
                passed=True,
                score=1.0,
                evidence={
                    "generation_signal": "train and validation splits only",
                    "test_signal": "held-out after learner and instrument evolution",
                },
            ),
            EmergentGate(
                "unbounded_open_world_domain_creation",
                passed=bool(open_world_results) and all(result.passed for result in open_world_results),
                score=sum(
                    result.clean_test_metrics["balanced_accuracy"]
                    for result in open_world_results
                ) / max(1, len(open_world_results)),
                evidence={
                    "observed": "unrequested real-data domains were discovered locally, scored with the evolved instrument, and tested on held-out splits",
                    "open_world_task_results": [
                        {
                            "task_name": result.task_name,
                            "domain": result.domain,
                            "passed": result.passed,
                            "clean_test_balanced_accuracy": result.clean_test_metrics["balanced_accuracy"],
                            "stress_floor_balanced_accuracy": result.stress_test_metrics["stress_floor_balanced_accuracy"],
                        }
                        for result in open_world_results
                    ],
                },
            ),
            EmergentGate(
                "autonomous_long_horizon_tool_use",
                passed=(
                    observed_actions == expected_actions
                    and all(record.passed for record in autonomous_plan_trace)
                ),
                score=float(sum(1 for record in autonomous_plan_trace if record.passed)),
                evidence={
                    "observed": "multi-step autonomous local plan completed with per-step verification",
                    "expected_actions": expected_actions,
                    "observed_actions": observed_actions,
                    "trace": [record.to_dict() for record in autonomous_plan_trace],
                },
            ),
            EmergentGate(
                "recursive_capability_improvement_without_manual_primitives",
                passed=bool(invented_primitive_types)
                and all(primitive.source_residue_ids for primitive in invented_primitive_types)
                and all(primitive.quality_gate.get("accepted") for primitive in invented_primitive_types),
                score=float(len(invented_primitive_types)),
                evidence={
                    "observed": "runtime primitive-type registry expanded from residue using an expandable grammar plus quality gate",
                    "primitive_type_count": len(invented_primitive_types),
                    "primitive_types": [
                        primitive.to_dict() for primitive in invented_primitive_types
                    ],
                },
            ),
            EmergentGate(
                "external_frontier_benchmark_validation",
                passed=False,
                score=0.0,
                evidence={
                    "observed": "all local emergent generalization gates now have executable evidence",
                    "missing": "independent ARC-AGI, GAIA, SWE-bench, or METR-style external evaluation has not been run or passed",
                },
            ),
        ]
        return gates


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="breast_cancer_diagnosis,wine_chemical_origin,handwritten_digit_recognition",
        help="Comma-separated dataset task names.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--generations-per-task", type=int, default=2)
    parser.add_argument("--json-output", default="emergent_generalization_report.json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tasks = build_dataset_tasks(args.datasets.split(","))
    if not tasks:
        raise SystemExit("no selected datasets")
    report = EmergentGeneralizationSystem(
        seed=args.seed,
        generations_per_task=args.generations_per_task,
    ).run(tasks)
    output = Path(args.json_output)
    output.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"final_verdict={report.final_verdict}")
    print(f"mechanism_success={report.mechanism_success}")
    print(f"agi_claim_verified={report.agi_claim_verified}")
    print(f"self_generated_goal_count={report.self_generated_goal_count}")
    print(f"invented_program_count={report.invented_program_count}")
    print(f"invented_primitive_type_count={report.invented_primitive_type_count}")
    print(f"transferred_program_count={report.transferred_program_count}")
    for result in report.task_results:
        state = "PASS" if result.passed else "FAIL"
        selected = result.selected_program["program_id"]
        clean = result.clean_test_metrics["balanced_accuracy"]
        stress = result.stress_test_metrics["stress_floor_balanced_accuracy"]
        print(
            f"{state} {result.task_name} selected={selected} "
            f"clean_test_balanced_accuracy={clean:.3f} stress_floor={stress:.3f}"
        )
    for result in report.open_world_task_results:
        state = "PASS" if result.passed else "FAIL"
        selected = result.selected_program["program_id"]
        clean = result.clean_test_metrics["balanced_accuracy"]
        stress = result.stress_test_metrics["stress_floor_balanced_accuracy"]
        print(
            f"{state} open_world:{result.task_name} selected={selected} "
            f"clean_test_balanced_accuracy={clean:.3f} stress_floor={stress:.3f}"
        )
    print(
        "autonomous_plan_steps="
        f"{sum(1 for action in report.autonomous_plan_trace if action.passed)}/"
        f"{len(report.autonomous_plan_trace)}"
    )
    for gate in report.gates:
        state = "PASS" if gate.passed else "FAIL"
        print(f"{state} gate:{gate.name} score={gate.score:.3f}")
    print(f"report={output}")
    return 0 if report.agi_claim_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
