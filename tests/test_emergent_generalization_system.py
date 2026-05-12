import pytest

from emergent_generalization_system import EmergentGeneralizationSystem
from real_world_generalization_benchmark import build_dataset_tasks


@pytest.fixture(scope="module")
def full_report():
    tasks = build_dataset_tasks([
        "breast_cancer_diagnosis",
        "wine_chemical_origin",
        "handwritten_digit_recognition",
    ])
    return EmergentGeneralizationSystem(seed=17, generations_per_task=2).run(tasks)


def test_emergent_system_generates_goals_programs_and_transfer(full_report):
    assert full_report.mechanism_success
    assert full_report.self_generated_goal_count > 0
    assert full_report.invented_program_count > 0
    assert full_report.invented_primitive_type_count > 0
    assert full_report.transferred_program_count > 0
    assert all(result.passed for result in full_report.task_results)


def test_emergent_system_keeps_generation_and_test_split_separate(full_report):
    gates = {gate.name: gate for gate in full_report.gates}

    assert gates["test_split_not_used_for_generation"].passed
    assert gates["self_generated_validation_goals"].passed
    assert gates["residue_conditioned_learner_program_invention"].passed
    assert gates["cross_task_program_transfer"].passed
    assert gates["unbounded_open_world_domain_creation"].passed
    assert gates["autonomous_long_horizon_tool_use"].passed
    assert gates["recursive_capability_improvement_without_manual_primitives"].passed
    assert full_report.open_world_task_results
    assert full_report.autonomous_plan_trace
    assert all(record.passed for record in full_report.autonomous_plan_trace)


def test_emergent_system_exposes_remaining_external_validation_boundary(full_report):
    failing = {gate.name for gate in full_report.gates if not gate.passed}

    assert "external_frontier_benchmark_validation" in failing
    assert "recursive_capability_improvement_without_manual_primitives" not in failing
    assert "unbounded_open_world_domain_creation" not in failing
    assert "autonomous_long_horizon_tool_use" not in failing
