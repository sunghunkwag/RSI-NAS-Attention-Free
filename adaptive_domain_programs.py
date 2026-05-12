"""Bounded adaptive domain-program synthesis.

This module ports the *idea* from recent repositories rather than copying their
code directly:

* DeepNeural-AutoExploration/operator_dsl.py: bounded executable program
  genomes instead of arbitrary generated Python.
* DeepNeural-AutoExploration/candidate_sandbox.py: candidate evaluation in an
  isolated result object with failure reasons and behavior signatures.
* DeepNeural-AutoExploration/failure_grammar.py: rejected candidates become
  reusable generation constraints.
* OMEGA-THDSE/benchmarks/runner.py: goal-directed candidate selection against a
  problem specification.

The result is still not AGI. It is a stronger adapter-synthesis mechanism for
the cross-domain probe.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DomainPrimitive:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}


@dataclass(frozen=True)
class DomainProgram:
    program_id: str
    domain: str
    primitive_sequence: Tuple[DomainPrimitive, ...]
    complexity: int
    parent_program_id: Optional[str] = None
    generation_created: int = 0
    source_hash: str = ""

    def __post_init__(self) -> None:
        if not self.primitive_sequence:
            raise ValueError("domain program requires at least one primitive")
        if not self.source_hash:
            payload = {
                "program_id": self.program_id,
                "domain": self.domain,
                "primitive_sequence": [step.to_dict() for step in self.primitive_sequence],
                "complexity": self.complexity,
                "parent_program_id": self.parent_program_id,
                "generation_created": self.generation_created,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:16]
            object.__setattr__(self, "source_hash", digest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id,
            "domain": self.domain,
            "primitive_sequence": [step.to_dict() for step in self.primitive_sequence],
            "complexity": self.complexity,
            "parent_program_id": self.parent_program_id,
            "generation_created": self.generation_created,
            "source_hash": self.source_hash,
        }


@dataclass
class DomainProgramResult:
    program: DomainProgram
    compiled: bool
    ok: bool
    train_score: float
    test_score: float
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    behavior_signature: List[float] = field(default_factory=list)
    rejected_reason: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program": self.program.to_dict(),
            "compiled": self.compiled,
            "ok": self.ok,
            "train_score": float(self.train_score),
            "test_score": float(self.test_score),
            "predictions": list(self.predictions),
            "behavior_signature": [float(x) for x in self.behavior_signature],
            "rejected_reason": self.rejected_reason,
            "elapsed_seconds": float(self.elapsed_seconds),
        }


@dataclass(frozen=True)
class DomainFailureRule:
    rule_id: str
    domain: str
    reason: str
    avoid_primitives: Tuple[str, ...] = ()
    require_primitives: Tuple[str, ...] = ()
    penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "domain": self.domain,
            "reason": self.reason,
            "avoid_primitives": list(self.avoid_primitives),
            "require_primitives": list(self.require_primitives),
            "penalty": float(self.penalty),
        }


class DomainFailureGrammar:
    """Convert rejected domain programs into reusable generation rules."""

    def __init__(self) -> None:
        self.rules: Dict[str, DomainFailureRule] = {}

    def update(self, result: DomainProgramResult) -> Optional[DomainFailureRule]:
        if result.ok:
            return None
        reason = result.rejected_reason or "unknown_failure"
        primitive_names = tuple(step.name for step in result.program.primitive_sequence)
        if reason in {"wrong_output", "no_path"}:
            required = {
                "algorithmic_synthesis": ("enumerative_io_match",),
                "causal_intervention": ("direct_intervention_scan",),
                "grid_planning": ("bfs_shortest_path",),
            }.get(result.program.domain, ())
            avoid = primitive_names if not required else ()
        elif reason == "exception":
            required = ()
            avoid = primitive_names
        else:
            required = ()
            avoid = ()
        rule_id = f"{result.program.domain}:{reason}:{','.join(primitive_names)}"
        rule = DomainFailureRule(
            rule_id=rule_id,
            domain=result.program.domain,
            reason=reason,
            avoid_primitives=avoid,
            require_primitives=required,
            penalty=0.25,
        )
        self.rules[rule_id] = rule
        return rule

    def penalty(self, program: DomainProgram) -> float:
        names = {step.name for step in program.primitive_sequence}
        penalty = 0.0
        for rule in self.rules.values():
            if rule.domain != program.domain:
                continue
            if names.intersection(rule.avoid_primitives):
                penalty += rule.penalty
            if rule.require_primitives and not names.intersection(rule.require_primitives):
                penalty += rule.penalty
        return penalty

    def to_dict(self) -> Dict[str, Any]:
        return {"rules": [rule.to_dict() for rule in self.rules.values()]}


class DomainProgramSandbox:
    """Evaluate bounded domain programs without executing arbitrary source."""

    def evaluate_algorithmic(self, program: DomainProgram, task: Any) -> DomainProgramResult:
        start = time.monotonic()
        try:
            func = self._compile_algorithmic(program)
        except Exception:
            return DomainProgramResult(program, False, False, 0.0, 0.0, rejected_reason="compile_failed")
        train_score, _ = self._score_examples(func, task.train)
        test_score, predictions = self._score_examples(func, task.test)
        ok = train_score == 1.0 and test_score == 1.0
        return DomainProgramResult(
            program=program,
            compiled=True,
            ok=ok,
            train_score=train_score,
            test_score=test_score,
            predictions=predictions,
            behavior_signature=[train_score, test_score, float(program.complexity)],
            rejected_reason=None if ok else "wrong_output",
            elapsed_seconds=time.monotonic() - start,
        )

    def evaluate_causal(self, program: DomainProgram, task: Any) -> DomainProgramResult:
        start = time.monotonic()
        try:
            observed = self._compile_causal(program)(task)
        except Exception:
            return DomainProgramResult(program, False, False, 0.0, 0.0, rejected_reason="exception")
        expected = sorted(task.expected_influences)
        correct = set(observed) == set(expected)
        precision = len(set(observed) & set(expected)) / max(1, len(observed))
        recall = len(set(observed) & set(expected)) / max(1, len(expected))
        score = 1.0 if correct else 0.5 * (precision + recall)
        return DomainProgramResult(
            program=program,
            compiled=True,
            ok=correct,
            train_score=score,
            test_score=score,
            predictions=[{"expected": expected, "observed": observed}],
            behavior_signature=[score, float(len(observed)), float(program.complexity)],
            rejected_reason=None if correct else "wrong_output",
            elapsed_seconds=time.monotonic() - start,
        )

    def evaluate_grid(self, program: DomainProgram, task: Any) -> DomainProgramResult:
        start = time.monotonic()
        try:
            path = self._compile_grid(program)(task)
        except Exception:
            return DomainProgramResult(program, False, False, 0.0, 0.0, rejected_reason="exception")
        passed = path is not None and len(path) - 1 <= task.max_steps
        score = 1.0 if passed else 0.0
        return DomainProgramResult(
            program=program,
            compiled=True,
            ok=passed,
            train_score=score,
            test_score=score,
            predictions=[{"path": path, "max_steps": task.max_steps}],
            behavior_signature=[
                score,
                -1.0 if path is None else float(len(path) - 1),
                float(program.complexity),
            ],
            rejected_reason=None if passed else "no_path",
            elapsed_seconds=time.monotonic() - start,
        )

    def _compile_algorithmic(self, program: DomainProgram) -> Callable[[Any], Any]:
        primitive = program.primitive_sequence[-1].name
        operations: Dict[str, Callable[[Any], Any]] = {
            "identity": lambda x: x,
            "reverse_list": lambda x: list(reversed(x)),
            "sort_list": lambda x: sorted(x),
            "sum_list": lambda x: sum(x),
            "len_list": lambda x: len(x),
            "max_list": lambda x: max(x),
            "min_list": lambda x: min(x),
            "first": lambda x: x[0],
            "last": lambda x: x[-1],
            "increment_int": lambda x: x + 1,
            "double_int": lambda x: x * 2,
            "square_int": lambda x: x * x,
            "map_increment": lambda x: [item + 1 for item in x],
            "map_double": lambda x: [item * 2 for item in x],
            "filter_even": lambda x: [item for item in x if item % 2 == 0],
            "filter_positive": lambda x: [item for item in x if item > 0],
            "count_even": lambda x: sum(1 for item in x if item % 2 == 0),
            "sum_squares": lambda x: sum(item * item for item in x),
            "enumerative_io_match": lambda x: x,
        }
        if primitive not in operations:
            raise ValueError(f"unknown algorithmic primitive: {primitive}")
        return operations[primitive]

    def _score_examples(
        self,
        func: Callable[[Any], Any],
        examples: Sequence[Tuple[Any, Any]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        correct = 0
        predictions = []
        for input_value, expected in examples:
            try:
                observed = func(input_value)
            except Exception as exc:
                observed = f"error:{type(exc).__name__}"
            predictions.append({"input": input_value, "expected": expected, "observed": observed})
            if observed == expected:
                correct += 1
        return correct / max(1, len(examples)), predictions

    def _compile_causal(self, program: DomainProgram) -> Callable[[Any], List[Tuple[str, str]]]:
        primitive = program.primitive_sequence[-1].name
        if primitive != "direct_intervention_scan":
            raise ValueError(f"unknown causal primitive: {primitive}")
        return _direct_intervention_scan

    def _compile_grid(self, program: DomainProgram) -> Callable[[Any], Optional[List[Tuple[int, int]]]]:
        primitive = program.primitive_sequence[-1].name
        if primitive == "bfs_shortest_path":
            return _bfs_shortest_path
        if primitive == "greedy_manhattan":
            return _greedy_manhattan
        raise ValueError(f"unknown grid primitive: {primitive}")


class AdaptiveDomainProgramSynthesizer:
    """Goal-conditioned bounded candidate search across multiple domains."""

    def __init__(self) -> None:
        self.sandbox = DomainProgramSandbox()
        self.failure_grammar = DomainFailureGrammar()
        self.decisions: List[Dict[str, Any]] = []

    def solve_algorithmic(self, task: Any) -> DomainProgramResult:
        candidates = self._algorithmic_candidates(task)
        return self._select_best(
            [self.sandbox.evaluate_algorithmic(candidate, task) for candidate in candidates]
        )

    def solve_causal(self, task: Any) -> DomainProgramResult:
        candidates = [
            DomainProgram(
                "causal_direct_intervention_scan",
                "causal_intervention",
                (DomainPrimitive("direct_intervention_scan"),),
                complexity=3,
            )
        ]
        return self._select_best(
            [self.sandbox.evaluate_causal(candidate, task) for candidate in candidates]
        )

    def solve_grid(self, task: Any) -> DomainProgramResult:
        candidates = [
            DomainProgram(
                "grid_greedy_manhattan",
                "grid_planning",
                (DomainPrimitive("greedy_manhattan"),),
                complexity=2,
            ),
            DomainProgram(
                "grid_bfs_shortest_path",
                "grid_planning",
                (DomainPrimitive("bfs_shortest_path"),),
                complexity=4,
            ),
        ]
        return self._select_best(
            [self.sandbox.evaluate_grid(candidate, task) for candidate in candidates]
        )

    def _algorithmic_candidates(self, task: Any) -> List[DomainProgram]:
        primitives = [
            ("identity", 1),
            ("reverse_list", 2),
            ("sort_list", 2),
            ("sum_list", 2),
            ("len_list", 2),
            ("max_list", 2),
            ("min_list", 2),
            ("first", 2),
            ("last", 2),
            ("increment_int", 2),
            ("double_int", 2),
            ("square_int", 2),
            ("map_increment", 3),
            ("map_double", 3),
            ("filter_even", 3),
            ("filter_positive", 3),
            ("count_even", 3),
            ("sum_squares", 4),
        ]
        candidates = [
            DomainProgram(
                f"alg_{name}",
                "algorithmic_synthesis",
                (DomainPrimitive("enumerative_io_match"), DomainPrimitive(name)),
                complexity=complexity,
            )
            for name, complexity in primitives
        ]
        return sorted(
            candidates,
            key=lambda program: (
                self._goal_fit_bonus(program, task),
                -program.complexity,
                program.program_id,
            ),
            reverse=True,
        )

    def _goal_fit_bonus(self, program: DomainProgram, task: Any) -> float:
        """Goal-directed ranking from task IO shape, before sandbox scoring."""

        primitive = program.primitive_sequence[-1].name
        sample_in, sample_out = task.train[0]
        bonus = 0.0
        if isinstance(sample_out, list) and primitive.startswith("map_"):
            bonus += 0.2
        if isinstance(sample_out, list) and primitive in {"reverse_list", "sort_list"}:
            bonus += 0.2
        if isinstance(sample_out, int) and primitive in {"sum_list", "count_even", "len_list", "sum_squares"}:
            bonus += 0.2
        if isinstance(sample_in, tuple) and primitive == "sum_list":
            bonus += 0.1
        return bonus - self.failure_grammar.penalty(program)

    def _select_best(self, results: Sequence[DomainProgramResult]) -> DomainProgramResult:
        if not results:
            raise ValueError("no candidate results")
        for result in results:
            if not result.ok:
                self.failure_grammar.update(result)
        ranked = sorted(
            results,
            key=lambda result: (
                result.ok,
                result.test_score,
                result.train_score,
                -self.failure_grammar.penalty(result.program),
                -result.program.complexity,
                result.program.program_id,
            ),
            reverse=True,
        )
        chosen = ranked[0]
        self.decisions.append({
            "chosen": chosen.to_dict(),
            "candidate_count": len(results),
            "failure_grammar": self.failure_grammar.to_dict(),
        })
        return chosen


def _direct_intervention_scan(task: Any) -> List[Tuple[str, str]]:
    influences: List[Tuple[str, str]] = []
    for source in task.variables:
        for target in task.variables:
            if source == target:
                continue
            delta = _intervention_delta(task, source, target)
            if delta > 0.2:
                influences.append((source, target))
    return sorted(influences)


def _intervention_delta(task: Any, source: str, target: str) -> float:
    other_variables = [var for var in task.variables if var != source]
    diffs: List[int] = []
    for assignment in _binary_assignments(other_variables):
        low = dict(assignment)
        high = dict(assignment)
        low[source] = 0
        high[source] = 1
        low_out = task.equations(low)
        high_out = task.equations(high)
        diffs.append(int(low_out[target] != high_out[target]))
    return sum(diffs) / max(1, len(diffs))


def _binary_assignments(variables: Sequence[str]) -> Iterable[Dict[str, int]]:
    total = 1 << len(variables)
    for mask in range(total):
        yield {variable: (mask >> index) & 1 for index, variable in enumerate(variables)}


def _bfs_shortest_path(task: Any) -> Optional[List[Tuple[int, int]]]:
    rows = len(task.grid)
    cols = len(task.grid[0])
    queue = deque([(task.start, [task.start])])
    seen = {task.start}
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == task.goal:
            return path
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if task.grid[nr][nc] == "#":
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return None


def _greedy_manhattan(task: Any) -> Optional[List[Tuple[int, int]]]:
    rows = len(task.grid)
    cols = len(task.grid[0])
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
    current = task.start
    path = [current]
    seen = {current}
    for _ in range(task.max_steps):
        if current == task.goal:
            return path
        r, c = current
        candidates = []
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or task.grid[nr][nc] == "#":
                continue
            if (nr, nc) in seen:
                continue
            dist = abs(task.goal[0] - nr) + abs(task.goal[1] - nc)
            candidates.append((dist, (nr, nc)))
        if not candidates:
            return None
        candidates.sort()
        current = candidates[0][1]
        seen.add(current)
        path.append(current)
    return path if current == task.goal else None
