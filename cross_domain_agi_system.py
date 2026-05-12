"""Cross-domain AGI boundary probe for the RSI-NAS integration.

This is a breakthrough attempt beyond the previous narrow character-LM NAS
validation. It adds independent domains:

* small algorithmic program synthesis
* interventional causal influence discovery
* grid-world planning

The code is intentionally strict about the final claim. Passing these toy
domains demonstrates cross-domain machinery, not AGI.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from local_agi_architect import run_local_agi_architect
from adaptive_domain_programs import AdaptiveDomainProgramSynthesizer


JsonValue = Any


@dataclass
class DomainResult:
    domain: str
    task_name: str
    passed: bool
    score: float
    evidence: Dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class AlgorithmicProgram:
    name: str
    complexity: int
    func: Callable[[Any], Any]

    def run(self, value: Any) -> Any:
        return self.func(value)


@dataclass
class AlgorithmicTask:
    name: str
    train: List[Tuple[Any, Any]]
    test: List[Tuple[Any, Any]]


class AlgorithmicProgramSynthesizer:
    """Enumerate a small DSL and select the simplest program fitting examples."""

    def __init__(self) -> None:
        self.programs = self._build_library()

    def _build_library(self) -> List[AlgorithmicProgram]:
        def safe_len(x: Any) -> int:
            return len(x)

        def safe_sum(x: Any) -> Any:
            return sum(x)

        def count_even(x: Sequence[int]) -> int:
            return sum(1 for item in x if item % 2 == 0)

        def pair_add(x: Sequence[int]) -> int:
            return x[0] + x[1]

        def pair_mul(x: Sequence[int]) -> int:
            return x[0] * x[1]

        programs = [
            AlgorithmicProgram("identity", 1, lambda x: x),
            AlgorithmicProgram("reverse_list", 2, lambda x: list(reversed(x))),
            AlgorithmicProgram("sort_list", 2, lambda x: sorted(x)),
            AlgorithmicProgram("sum_list", 2, safe_sum),
            AlgorithmicProgram("len_list", 2, safe_len),
            AlgorithmicProgram("max_list", 2, max),
            AlgorithmicProgram("min_list", 2, min),
            AlgorithmicProgram("first", 2, lambda x: x[0]),
            AlgorithmicProgram("last", 2, lambda x: x[-1]),
            AlgorithmicProgram("increment_int", 2, lambda x: x + 1),
            AlgorithmicProgram("double_int", 2, lambda x: x * 2),
            AlgorithmicProgram("square_int", 2, lambda x: x * x),
            AlgorithmicProgram("pair_add", 3, pair_add),
            AlgorithmicProgram("pair_mul", 3, pair_mul),
            AlgorithmicProgram("map_increment", 3, lambda x: [item + 1 for item in x]),
            AlgorithmicProgram("map_double", 3, lambda x: [item * 2 for item in x]),
            AlgorithmicProgram("filter_even", 3, lambda x: [item for item in x if item % 2 == 0]),
            AlgorithmicProgram("filter_positive", 3, lambda x: [item for item in x if item > 0]),
            AlgorithmicProgram("count_even", 3, count_even),
            AlgorithmicProgram("sum_squares", 4, lambda x: sum(item * item for item in x)),
        ]
        return sorted(programs, key=lambda program: (program.complexity, program.name))

    def synthesize(self, examples: Sequence[Tuple[Any, Any]]) -> Optional[AlgorithmicProgram]:
        for program in self.programs:
            if self._fits(program, examples):
                return program
        return None

    def _fits(self, program: AlgorithmicProgram, examples: Sequence[Tuple[Any, Any]]) -> bool:
        for input_value, expected in examples:
            try:
                observed = program.run(input_value)
            except Exception:
                return False
            if observed != expected:
                return False
        return True

    def solve(self, task: AlgorithmicTask) -> DomainResult:
        program = self.synthesize(task.train)
        if program is None:
            return DomainResult(
                domain="algorithmic_synthesis",
                task_name=task.name,
                passed=False,
                score=0.0,
                evidence={"reason": "no program fit training examples"},
            )

        passed = 0
        predictions = []
        for input_value, expected in task.test:
            try:
                observed = program.run(input_value)
            except Exception as exc:
                observed = f"error:{type(exc).__name__}"
            predictions.append({
                "input": input_value,
                "expected": expected,
                "observed": observed,
            })
            if observed == expected:
                passed += 1
        score = passed / max(1, len(task.test))
        return DomainResult(
            domain="algorithmic_synthesis",
            task_name=task.name,
            passed=score == 1.0,
            score=score,
            evidence={
                "program": program.name,
                "complexity": program.complexity,
                "predictions": predictions,
            },
        )


def algorithmic_tasks() -> List[AlgorithmicTask]:
    return [
        AlgorithmicTask(
            "reverse_unseen_lists",
            train=[([1, 2, 3], [3, 2, 1]), ([5, 6], [6, 5])],
            test=[([9, 8, 7, 6], [6, 7, 8, 9]), ([0, -1, 4], [4, -1, 0])],
        ),
        AlgorithmicTask(
            "count_even_numbers",
            train=[([1, 2, 3, 4], 2), ([2, 8, 10], 3)],
            test=[([1, 3, 5], 0), ([0, 2, 5, 7, 8], 3)],
        ),
        AlgorithmicTask(
            "map_increment_unseen",
            train=[([1, 2], [2, 3]), ([0, -1, 5], [1, 0, 6])],
            test=[([10, 20], [11, 21]), ([-5, -4], [-4, -3])],
        ),
        AlgorithmicTask(
            "pair_addition",
            train=[((1, 2), 3), ((-4, 10), 6)],
            test=[((100, 23), 123), ((-8, -9), -17)],
        ),
        AlgorithmicTask(
            "sum_squares",
            train=[([1, 2, 3], 14), ([4, 0], 16)],
            test=[([2, 5], 29), ([3, 3, 1], 19)],
        ),
    ]


@dataclass
class CausalTask:
    name: str
    variables: Tuple[str, ...]
    expected_influences: List[Tuple[str, str]]
    equations: Callable[[Dict[str, int]], Dict[str, int]]


class CausalInfluenceEngine:
    """Estimate interventional influence by comparing do(X=0) vs do(X=1)."""

    def estimate(self, task: CausalTask) -> List[Tuple[str, str]]:
        influences: List[Tuple[str, str]] = []
        for source in task.variables:
            for target in task.variables:
                if source == target:
                    continue
                delta = self._intervention_delta(task, source, target)
                if delta > 0.2:
                    influences.append((source, target))
        return sorted(influences)

    def _intervention_delta(self, task: CausalTask, source: str, target: str) -> float:
        other_variables = [var for var in task.variables if var != source]
        diffs: List[int] = []
        for assignment in self._binary_assignments(other_variables):
            low = dict(assignment)
            high = dict(assignment)
            low[source] = 0
            high[source] = 1
            low_out = task.equations(low)
            high_out = task.equations(high)
            diffs.append(int(low_out[target] != high_out[target]))
        return sum(diffs) / max(1, len(diffs))

    def _binary_assignments(self, variables: Sequence[str]) -> Iterable[Dict[str, int]]:
        total = 1 << len(variables)
        for mask in range(total):
            yield {
                variable: (mask >> index) & 1
                for index, variable in enumerate(variables)
            }

    def solve(self, task: CausalTask) -> DomainResult:
        observed = self.estimate(task)
        expected = sorted(task.expected_influences)
        correct = set(observed) == set(expected)
        precision = len(set(observed) & set(expected)) / max(1, len(observed))
        recall = len(set(observed) & set(expected)) / max(1, len(expected))
        score = 1.0 if correct else round(0.5 * (precision + recall), 4)
        return DomainResult(
            domain="causal_intervention",
            task_name=task.name,
            passed=correct,
            score=score,
            evidence={
                "expected_influences": expected,
                "observed_influences": observed,
            },
        )


def causal_tasks() -> List[CausalTask]:
    return [
        CausalTask(
            name="chain_influence",
            variables=("A", "B", "C"),
            expected_influences=[("A", "B"), ("B", "C")],
            equations=lambda v: {"A": v["A"], "B": v["A"], "C": v["B"]},
        ),
        CausalTask(
            name="fork_influence",
            variables=("A", "B", "C"),
            expected_influences=[("A", "B"), ("A", "C")],
            equations=lambda v: {"A": v["A"], "B": v["A"], "C": 1 - v["A"]},
        ),
        CausalTask(
            name="xor_collider_influence",
            variables=("A", "B", "C"),
            expected_influences=[("A", "C"), ("B", "C")],
            equations=lambda v: {"A": v["A"], "B": v["B"], "C": v["A"] ^ v["B"]},
        ),
    ]


@dataclass
class GridTask:
    name: str
    grid: Tuple[str, ...]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    max_steps: int


class GridPlanner:
    """Breadth-first planning over unseen grid worlds."""

    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def solve_path(self, task: GridTask) -> Optional[List[Tuple[int, int]]]:
        rows = len(task.grid)
        cols = len(task.grid[0])
        queue = deque([(task.start, [task.start])])
        seen = {task.start}
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == task.goal:
                return path
            for dr, dc in self.moves:
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

    def solve(self, task: GridTask) -> DomainResult:
        path = self.solve_path(task)
        passed = path is not None and len(path) - 1 <= task.max_steps
        return DomainResult(
            domain="grid_planning",
            task_name=task.name,
            passed=bool(passed),
            score=1.0 if passed else 0.0,
            evidence={
                "path": path,
                "path_length": None if path is None else len(path) - 1,
                "max_steps": task.max_steps,
            },
        )


def grid_tasks() -> List[GridTask]:
    return [
        GridTask(
            "corridor_detour",
            grid=(".....", ".###.", "...#.", ".#...", "....."),
            start=(0, 0),
            goal=(4, 4),
            max_steps=8,
        ),
        GridTask(
            "narrow_gate",
            grid=("..#...", "..#.#.", "....#.", ".####.", "......"),
            start=(0, 0),
            goal=(4, 5),
            max_steps=11,
        ),
        GridTask(
            "blocked_center",
            grid=("......", ".####.", ".#..#.", ".#..#.", ".####.", "......"),
            start=(0, 0),
            goal=(5, 5),
            max_steps=10,
        ),
    ]


@dataclass
class CrossDomainReport:
    final_verdict: str
    toy_cross_domain_success: bool
    agi_claim_verified: bool
    domain_results: List[DomainResult]
    gates: List[DomainResult]
    local_inventory_summary: Optional[Dict[str, Any]]
    failure_reasons: List[str]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "final_verdict": self.final_verdict,
            "toy_cross_domain_success": self.toy_cross_domain_success,
            "agi_claim_verified": self.agi_claim_verified,
            "domain_results": [asdict(result) for result in self.domain_results],
            "gates": [asdict(result) for result in self.gates],
            "local_inventory_summary": self.local_inventory_summary,
            "failure_reasons": self.failure_reasons,
        }


def run_cross_domain_probe(args: argparse.Namespace) -> CrossDomainReport:
    domain_results: List[DomainResult] = []
    adaptive = AdaptiveDomainProgramSynthesizer()
    for task in algorithmic_tasks():
        result = adaptive.solve_algorithmic(task)
        domain_results.append(DomainResult(
            domain="algorithmic_synthesis",
            task_name=task.name,
            passed=result.ok,
            score=result.test_score,
            evidence={
                "selected_program": result.program.to_dict(),
                "predictions": result.predictions,
                "behavior_signature": result.behavior_signature,
            },
        ))

    for task in causal_tasks():
        result = adaptive.solve_causal(task)
        domain_results.append(DomainResult(
            domain="causal_intervention",
            task_name=task.name,
            passed=result.ok,
            score=result.test_score,
            evidence={
                "selected_program": result.program.to_dict(),
                "predictions": result.predictions,
                "behavior_signature": result.behavior_signature,
            },
        ))

    for task in grid_tasks():
        result = adaptive.solve_grid(task)
        domain_results.append(DomainResult(
            domain="grid_planning",
            task_name=task.name,
            passed=result.ok,
            score=result.test_score,
            evidence={
                "selected_program": result.program.to_dict(),
                "predictions": result.predictions,
                "behavior_signature": result.behavior_signature,
            },
        ))

    local_summary = None
    if args.inventory:
        local_args = argparse.Namespace(
            inventory=args.inventory,
            max_files=args.max_files,
            candidates=args.candidates,
            d_model=args.d_model,
            train_steps=args.train_steps,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_params=args.max_params,
            device=args.device,
            no_evaluate=args.no_evaluate_local,
        )
        local_report = run_local_agi_architect(local_args)
        local_summary = {
            "signals_scanned": local_report.get("signals_scanned"),
            "syntax_valid_signals": local_report.get("syntax_valid_signals"),
            "registered_generated_modules": local_report.get("registered_generated_modules"),
            "candidate_count": len(local_report.get("candidates", [])),
        }

    by_domain: Dict[str, List[DomainResult]] = {}
    for result in domain_results:
        by_domain.setdefault(result.domain, []).append(result)

    gates = [
        DomainResult(
            domain="meta_gate",
            task_name=f"{domain}_all_tasks_passed",
            passed=all(result.passed for result in rows),
            score=sum(result.score for result in rows) / max(1, len(rows)),
            evidence={
                "passed": sum(1 for result in rows if result.passed),
                "total": len(rows),
            },
        )
        for domain, rows in sorted(by_domain.items())
    ]
    gates.extend([
        DomainResult(
            domain="agi_gate",
            task_name="unified_learning_not_handcrafted_adapters",
            passed=False,
            score=0.0,
            evidence={
                "observed": "bounded domain-program candidates are sandboxed and selected, with failure grammar feedback",
                "missing": "a learned policy that creates genuinely new primitives or domain programs from experience",
                "referenced_recent_repo_ideas": [
                    "DeepNeural-AutoExploration/operator_dsl.py",
                    "DeepNeural-AutoExploration/candidate_sandbox.py",
                    "DeepNeural-AutoExploration/failure_grammar.py",
                    "OMEGA-THDSE/benchmarks/runner.py",
                ],
            },
        ),
        DomainResult(
            domain="agi_gate",
            task_name="autonomous_goal_formation_and_long_horizon_tool_use",
            passed=False,
            score=0.0,
            evidence={
                "observed": "predefined benchmark objectives",
                "missing": "self-proposed goals, tool plans, and verification in unseen environments",
            },
        ),
        DomainResult(
            domain="agi_gate",
            task_name="open_ended_recursive_generalization",
            passed=False,
            score=0.0,
            evidence={
                "observed": "bounded local inventory NAS plus toy-domain solvers",
                "missing": "recursive improvement that transfers to new task families without manual adapter insertion",
            },
        ),
    ])

    toy_success = all(
        gate.passed
        for gate in gates
        if gate.domain == "meta_gate"
    )
    agi_verified = all(gate.passed for gate in gates)
    if agi_verified:
        verdict = "AGI_CLAIM_VERIFIED"
    elif toy_success:
        verdict = "CROSS_DOMAIN_TOY_SUCCESS_BUT_AGI_FAILED"
    else:
        verdict = "CROSS_DOMAIN_ATTEMPT_FAILED"

    failure_reasons = [
        f"{gate.task_name}: {gate.evidence.get('missing')}"
        for gate in gates
        if not gate.passed and gate.evidence.get("missing")
    ]
    return CrossDomainReport(
        final_verdict=verdict,
        toy_cross_domain_success=toy_success,
        agi_claim_verified=agi_verified,
        domain_results=domain_results,
        gates=gates,
        local_inventory_summary=local_summary,
        failure_reasons=failure_reasons,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=None)
    parser.add_argument("--max-files", type=int, default=160)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-params", type=int, default=300_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-evaluate-local", action="store_true")
    parser.add_argument("--json-output", default="cross_domain_agi_report.json")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = run_cross_domain_probe(args)
    output = Path(args.json_output)
    output.write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"final_verdict={report.final_verdict}")
    print(f"toy_cross_domain_success={report.toy_cross_domain_success}")
    print(f"agi_claim_verified={report.agi_claim_verified}")
    for gate in report.gates:
        state = "PASS" if gate.passed else "FAIL"
        print(f"{state} {gate.domain}:{gate.task_name} score={gate.score:.3f}")
    print(f"report={output}")
    return 0 if report.agi_claim_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
