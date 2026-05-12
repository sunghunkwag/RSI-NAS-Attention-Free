from real_world_generalization_benchmark import (
    RealWorldGeneralizationEvaluator,
    build_dataset_tasks,
)


def test_real_world_benchmark_beats_baseline_on_real_datasets():
    tasks = build_dataset_tasks(["breast_cancer_diagnosis", "wine_chemical_origin"])
    report = RealWorldGeneralizationEvaluator(seed=17).evaluate(tasks)

    assert report.real_benchmark_success
    assert report.final_verdict == "REAL_BENCHMARK_GENERALIZATION_PASSED_BUT_AGI_FAILED"
    assert not report.agi_claim_verified
    assert all(result.passed for result in report.dataset_results)
    assert all(result.candidate_count >= 5 for result in report.dataset_results)
    assert all(
        result.test["balanced_accuracy"] > result.baseline["balanced_accuracy"]
        for result in report.dataset_results
    )


def test_real_world_benchmark_keeps_agi_boundary_explicit():
    tasks = build_dataset_tasks(["breast_cancer_diagnosis"])
    report = RealWorldGeneralizationEvaluator(seed=17).evaluate(tasks)
    failing_gate_names = {gate.name for gate in report.gates if not gate.passed}

    assert "novel_domain_without_predefined_candidate_library" in failing_gate_names
    assert "autonomous_goal_formation" in failing_gate_names
    assert "recursive_self_improvement_on_real_tasks" in failing_gate_names
