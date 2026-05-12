from eie_real_world_instrument_evolver import EIERealWorldInstrumentEvolver
from real_world_generalization_benchmark import build_dataset_tasks


def test_eie_real_world_instrument_evolves_from_residue():
    tasks = build_dataset_tasks(["breast_cancer_diagnosis", "handwritten_digit_recognition"])
    report = EIERealWorldInstrumentEvolver(seed=17, generations=2).evaluate(tasks)

    assert report.eie_real_benchmark_success
    assert report.instrument_evolved
    assert not report.agi_claim_verified
    assert report.final_verdict == "EIE_REAL_WORLD_INSTRUMENT_EVOLVED_BUT_AGI_FAILED"
    assert report.final_instrument["noise_stress_weight"] > 0.0
    assert report.final_instrument["feature_dropout_stress_weight"] > 0.0
    assert any(cycle.patch and cycle.patch.source_residue_ids for cycle in report.evolution_cycles)


def test_eie_real_world_uses_probationary_evaluator_acceptance():
    tasks = build_dataset_tasks(["handwritten_digit_recognition"])
    report = EIERealWorldInstrumentEvolver(seed=17, generations=2).evaluate(tasks)
    probation = [
        cycle.probation_decision
        for cycle in report.evolution_cycles
        if cycle.probation_decision is not None
    ]
    gate_by_name = {gate.name: gate for gate in report.gates}

    assert probation
    assert all(decision.accepted for decision in probation)
    assert all(decision.adversarial_checks["test_split_not_used"] for decision in probation)
    assert all(decision.adversarial_checks["source_residue_required"] for decision in probation)
    assert gate_by_name["probationary_evaluator_acceptance"].passed


def test_eie_real_world_keeps_agi_failure_boundary():
    tasks = build_dataset_tasks(["breast_cancer_diagnosis"])
    report = EIERealWorldInstrumentEvolver(seed=17, generations=2).evaluate(tasks)
    failing_gate_names = {gate.name for gate in report.gates if not gate.passed}

    assert "learner_primitive_invention" in failing_gate_names
    assert "autonomous_goal_formation" in failing_gate_names
    assert "open_ended_recursive_self_improvement" in failing_gate_names
