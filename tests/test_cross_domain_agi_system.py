import argparse

from cross_domain_agi_system import (
    AlgorithmicProgramSynthesizer,
    CausalInfluenceEngine,
    GridPlanner,
    algorithmic_tasks,
    causal_tasks,
    grid_tasks,
    run_cross_domain_probe,
)


def test_algorithmic_synthesizer_solves_all_toy_tasks():
    synthesizer = AlgorithmicProgramSynthesizer()

    results = [synthesizer.solve(task) for task in algorithmic_tasks()]

    assert all(result.passed for result in results)
    assert {result.evidence["program"] for result in results} >= {
        "reverse_list",
        "count_even",
        "map_increment",
        "sum_list",
        "sum_squares",
    }


def test_causal_engine_recovers_interventional_influences():
    engine = CausalInfluenceEngine()

    results = [engine.solve(task) for task in causal_tasks()]

    assert all(result.passed for result in results)


def test_grid_planner_solves_unseen_mazes():
    planner = GridPlanner()

    results = [planner.solve(task) for task in grid_tasks()]

    assert all(result.passed for result in results)


def test_cross_domain_probe_passes_toy_domains_but_not_agi():
    args = argparse.Namespace(
        inventory=None,
        max_files=10,
        candidates=1,
        d_model=8,
        train_steps=1,
        seq_len=8,
        batch_size=1,
        max_params=100_000,
        device="cpu",
        no_evaluate_local=True,
    )

    report = run_cross_domain_probe(args)

    assert report.toy_cross_domain_success
    assert not report.agi_claim_verified
    assert report.final_verdict == "CROSS_DOMAIN_TOY_SUCCESS_BUT_AGI_FAILED"
    assert "selected_program" in report.domain_results[0].evidence
