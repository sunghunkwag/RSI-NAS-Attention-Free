"""Real-data generalization benchmark for the RSI-NAS AGI boundary.

This module deliberately raises the bar above the toy cross-domain probe. It
uses real, built-in scikit-learn datasets, separates train/validation/test
splits, selects only from validation evidence, then reports final holdout
performance against a dummy baseline.

Passing this benchmark is still not AGI. It is evidence that the system can run
bounded model selection over real datasets without test leakage or fake success
flags.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression


ArrayPair = Tuple[np.ndarray, np.ndarray]
EstimatorBuilder = Callable[[int], BaseEstimator]


@dataclass(frozen=True)
class RealDatasetTask:
    name: str
    domain: str
    x: np.ndarray
    y: np.ndarray
    target_names: Tuple[str, ...]
    min_test_balanced_accuracy: float
    min_baseline_margin: float

    @property
    def sample_count(self) -> int:
        return int(self.x.shape[0])

    @property
    def feature_count(self) -> int:
        return int(np.prod(self.x.shape[1:]))

    @property
    def class_count(self) -> int:
        return int(len(np.unique(self.y)))


@dataclass(frozen=True)
class CandidatePipeline:
    name: str
    family: str
    complexity: int
    builder: EstimatorBuilder

    def build(self, seed: int) -> BaseEstimator:
        return self.builder(seed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "complexity": self.complexity,
        }


@dataclass
class CandidateEvaluation:
    candidate: Dict[str, Any]
    fit_ok: bool
    validation_balanced_accuracy: float
    validation_accuracy: float
    validation_f1_macro: float
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def selection_tuple(self) -> Tuple[float, float, float, int, str]:
        complexity = int(self.candidate.get("complexity", 999))
        return (
            float(self.validation_balanced_accuracy),
            float(self.validation_f1_macro),
            float(self.validation_accuracy),
            -complexity,
            str(self.candidate.get("name", "")),
        )


@dataclass
class RealDatasetResult:
    task_name: str
    domain: str
    passed: bool
    selected_candidate: Dict[str, Any]
    baseline: Dict[str, float]
    validation: Dict[str, float]
    test: Dict[str, float]
    thresholds: Dict[str, float]
    candidate_count: int
    candidate_evaluations: List[CandidateEvaluation] = field(default_factory=list)
    split: Dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["candidate_evaluations"] = [
            asdict(evaluation) for evaluation in self.candidate_evaluations
        ]
        return payload


@dataclass
class BenchmarkGate:
    name: str
    passed: bool
    score: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealWorldGeneralizationReport:
    final_verdict: str
    real_benchmark_success: bool
    agi_claim_verified: bool
    dataset_results: List[RealDatasetResult]
    gates: List[BenchmarkGate]
    failure_reasons: List[str]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "final_verdict": self.final_verdict,
            "real_benchmark_success": self.real_benchmark_success,
            "agi_claim_verified": self.agi_claim_verified,
            "dataset_results": [result.to_dict() for result in self.dataset_results],
            "gates": [asdict(gate) for gate in self.gates],
            "failure_reasons": list(self.failure_reasons),
        }


class RealWorldGeneralizationEvaluator:
    """Select bounded model pipelines on validation data and test holdout skill."""

    def __init__(self, seed: int = 17) -> None:
        self.seed = seed
        self.candidates = build_candidate_library()

    def evaluate(self, tasks: Sequence[RealDatasetTask]) -> RealWorldGeneralizationReport:
        dataset_results = [self.evaluate_task(task) for task in tasks]
        real_success = all(result.passed for result in dataset_results)
        gates = self._build_gates(dataset_results)
        agi_verified = all(gate.passed for gate in gates)
        if agi_verified:
            verdict = "AGI_CLAIM_VERIFIED"
        elif real_success:
            verdict = "REAL_BENCHMARK_GENERALIZATION_PASSED_BUT_AGI_FAILED"
        else:
            verdict = "REAL_BENCHMARK_ATTEMPT_FAILED"
        return RealWorldGeneralizationReport(
            final_verdict=verdict,
            real_benchmark_success=real_success,
            agi_claim_verified=agi_verified,
            dataset_results=dataset_results,
            gates=gates,
            failure_reasons=[
                str(gate.evidence.get("missing"))
                for gate in gates
                if not gate.passed and gate.evidence.get("missing")
            ],
        )

    def evaluate_task(self, task: RealDatasetTask) -> RealDatasetResult:
        start = time.monotonic()
        split = self._split(task)
        x_train, y_train = split["train"]
        x_val, y_val = split["validation"]
        x_test, y_test = split["test"]

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(x_train, y_train)
        baseline_test = self._score(baseline, x_test, y_test)

        evaluations = [
            self._evaluate_candidate(candidate, x_train, y_train, x_val, y_val)
            for candidate in self.candidates
        ]
        valid = [evaluation for evaluation in evaluations if evaluation.fit_ok]
        if not valid:
            return RealDatasetResult(
                task_name=task.name,
                domain=task.domain,
                passed=False,
                selected_candidate={"name": "none"},
                baseline=baseline_test,
                validation={"balanced_accuracy": 0.0, "accuracy": 0.0, "f1_macro": 0.0},
                test={"balanced_accuracy": 0.0, "accuracy": 0.0, "f1_macro": 0.0},
                thresholds={
                    "min_test_balanced_accuracy": task.min_test_balanced_accuracy,
                    "min_baseline_margin": task.min_baseline_margin,
                },
                candidate_count=len(self.candidates),
                candidate_evaluations=evaluations,
                split=self._split_counts(split),
                elapsed_seconds=time.monotonic() - start,
            )

        selected_eval = sorted(valid, key=lambda evaluation: evaluation.selection_tuple(), reverse=True)[0]
        selected_candidate = self._candidate_by_name(str(selected_eval.candidate["name"]))
        final_model = clone(selected_candidate.build(self.seed))
        x_fit = np.concatenate([x_train, x_val], axis=0)
        y_fit = np.concatenate([y_train, y_val], axis=0)
        final_model.fit(x_fit, y_fit)
        final_test = self._score(final_model, x_test, y_test)

        baseline_margin = (
            final_test["balanced_accuracy"] - baseline_test["balanced_accuracy"]
        )
        passed = (
            final_test["balanced_accuracy"] >= task.min_test_balanced_accuracy
            and baseline_margin >= task.min_baseline_margin
        )
        return RealDatasetResult(
            task_name=task.name,
            domain=task.domain,
            passed=passed,
            selected_candidate=selected_candidate.to_dict(),
            baseline=baseline_test,
            validation={
                "balanced_accuracy": selected_eval.validation_balanced_accuracy,
                "accuracy": selected_eval.validation_accuracy,
                "f1_macro": selected_eval.validation_f1_macro,
            },
            test=final_test,
            thresholds={
                "min_test_balanced_accuracy": task.min_test_balanced_accuracy,
                "min_baseline_margin": task.min_baseline_margin,
                "observed_baseline_margin": baseline_margin,
            },
            candidate_count=len(self.candidates),
            candidate_evaluations=evaluations,
            split=self._split_counts(split),
            elapsed_seconds=time.monotonic() - start,
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

    def _evaluate_candidate(
        self,
        candidate: CandidatePipeline,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> CandidateEvaluation:
        start = time.monotonic()
        try:
            model = candidate.build(self.seed)
            model.fit(x_train, y_train)
            scores = self._score(model, x_val, y_val)
            return CandidateEvaluation(
                candidate=candidate.to_dict(),
                fit_ok=True,
                validation_balanced_accuracy=scores["balanced_accuracy"],
                validation_accuracy=scores["accuracy"],
                validation_f1_macro=scores["f1_macro"],
                elapsed_seconds=time.monotonic() - start,
            )
        except Exception as exc:
            return CandidateEvaluation(
                candidate=candidate.to_dict(),
                fit_ok=False,
                validation_balanced_accuracy=0.0,
                validation_accuracy=0.0,
                validation_f1_macro=0.0,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.monotonic() - start,
            )

    def _score(self, model: BaseEstimator, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        prediction = model.predict(x)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "accuracy": float(accuracy_score(y, prediction)),
            "f1_macro": float(f1_score(y, prediction, average="macro")),
        }

    def _candidate_by_name(self, name: str) -> CandidatePipeline:
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def _split_counts(self, split: Dict[str, ArrayPair]) -> Dict[str, int]:
        return {name: int(values[0].shape[0]) for name, values in split.items()}

    def _build_gates(self, results: Sequence[RealDatasetResult]) -> List[BenchmarkGate]:
        average_score = sum(result.test["balanced_accuracy"] for result in results) / max(1, len(results))
        gates = [
            BenchmarkGate(
                name="held_out_real_dataset_generalization",
                passed=all(result.passed for result in results),
                score=average_score,
                evidence={
                    "passed": sum(1 for result in results if result.passed),
                    "total": len(results),
                    "datasets": [result.task_name for result in results],
                },
            ),
            BenchmarkGate(
                name="selection_uses_validation_not_test",
                passed=True,
                score=1.0,
                evidence={
                    "selection_signal": "validation split only",
                    "final_signal": "held-out test split after refitting train+validation",
                },
            ),
            BenchmarkGate(
                name="beats_dummy_baseline_by_required_margin",
                passed=all(
                    result.thresholds["observed_baseline_margin"]
                    >= result.thresholds["min_baseline_margin"]
                    for result in results
                ),
                score=sum(
                    result.thresholds["observed_baseline_margin"] for result in results
                ) / max(1, len(results)),
                evidence={
                    result.task_name: result.thresholds["observed_baseline_margin"]
                    for result in results
                },
            ),
            BenchmarkGate(
                name="novel_domain_without_predefined_candidate_library",
                passed=False,
                score=0.0,
                evidence={
                    "observed": "bounded candidate library over real datasets",
                    "missing": "the system does not invent new learners or primitives for unseen real domains",
                },
            ),
            BenchmarkGate(
                name="autonomous_goal_formation",
                passed=False,
                score=0.0,
                evidence={
                    "observed": "externally supplied benchmark goals",
                    "missing": "self-generated goals with independent success criteria",
                },
            ),
            BenchmarkGate(
                name="recursive_self_improvement_on_real_tasks",
                passed=False,
                score=0.0,
                evidence={
                    "observed": "model selection, not recursive improvement of the learner",
                    "missing": "accepted self-modifications that improve future real-data learning without manual edits",
                },
            ),
        ]
        return gates


def build_candidate_library() -> List[CandidatePipeline]:
    return [
        CandidatePipeline(
            "scaled_logistic_regression",
            "linear_model",
            2,
            lambda seed: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=4000, random_state=seed)),
            ]),
        ),
        CandidatePipeline(
            "scaled_linear_svc",
            "margin_classifier",
            3,
            lambda seed: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LinearSVC(max_iter=6000, random_state=seed)),
            ]),
        ),
        CandidatePipeline(
            "scaled_rbf_svc",
            "kernel_classifier",
            4,
            lambda seed: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(C=10.0, gamma="scale", random_state=seed)),
            ]),
        ),
        CandidatePipeline(
            "scaled_knn_5",
            "instance_classifier",
            3,
            lambda seed: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]),
        ),
        CandidatePipeline(
            "random_forest",
            "tree_ensemble",
            4,
            lambda seed: RandomForestClassifier(
                n_estimators=120,
                random_state=seed,
                n_jobs=1,
                class_weight="balanced_subsample",
            ),
        ),
        CandidatePipeline(
            "extra_trees",
            "tree_ensemble",
            4,
            lambda seed: ExtraTreesClassifier(
                n_estimators=160,
                random_state=seed,
                n_jobs=1,
                class_weight="balanced",
            ),
        ),
    ]


def build_dataset_tasks(selected: Optional[Iterable[str]] = None) -> List[RealDatasetTask]:
    wanted = None if selected is None else {name.strip() for name in selected if name.strip()}
    datasets: List[RealDatasetTask] = []

    cancer = load_breast_cancer()
    datasets.append(RealDatasetTask(
        name="breast_cancer_diagnosis",
        domain="tabular_medical_measurements",
        x=np.asarray(cancer.data),
        y=np.asarray(cancer.target),
        target_names=tuple(str(name) for name in cancer.target_names),
        min_test_balanced_accuracy=0.90,
        min_baseline_margin=0.25,
    ))

    wine = load_wine()
    datasets.append(RealDatasetTask(
        name="wine_chemical_origin",
        domain="tabular_chemical_measurements",
        x=np.asarray(wine.data),
        y=np.asarray(wine.target),
        target_names=tuple(str(name) for name in wine.target_names),
        min_test_balanced_accuracy=0.90,
        min_baseline_margin=0.35,
    ))

    digits = load_digits()
    datasets.append(RealDatasetTask(
        name="handwritten_digit_recognition",
        domain="image_feature_classification",
        x=np.asarray(digits.data),
        y=np.asarray(digits.target),
        target_names=tuple(str(name) for name in digits.target_names),
        min_test_balanced_accuracy=0.92,
        min_baseline_margin=0.75,
    ))

    if wanted is None:
        return datasets
    return [task for task in datasets if task.name in wanted]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="breast_cancer_diagnosis,wine_chemical_origin,handwritten_digit_recognition",
        help="Comma-separated dataset task names to run.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-output", default="real_world_generalization_report.json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tasks = build_dataset_tasks(args.datasets.split(","))
    if not tasks:
        raise SystemExit("no selected datasets")
    evaluator = RealWorldGeneralizationEvaluator(seed=args.seed)
    report = evaluator.evaluate(tasks)
    output = Path(args.json_output)
    output.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"final_verdict={report.final_verdict}")
    print(f"real_benchmark_success={report.real_benchmark_success}")
    print(f"agi_claim_verified={report.agi_claim_verified}")
    for result in report.dataset_results:
        state = "PASS" if result.passed else "FAIL"
        selected = result.selected_candidate.get("name")
        test_score = result.test["balanced_accuracy"]
        margin = result.thresholds["observed_baseline_margin"]
        print(
            f"{state} {result.task_name} selected={selected} "
            f"test_balanced_accuracy={test_score:.3f} baseline_margin={margin:.3f}"
        )
    for gate in report.gates:
        state = "PASS" if gate.passed else "FAIL"
        print(f"{state} gate:{gate.name} score={gate.score:.3f}")
    print(f"report={output}")
    return 0 if report.agi_claim_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
