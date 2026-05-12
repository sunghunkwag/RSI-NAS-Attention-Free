from adaptive_domain_programs import AdaptiveDomainProgramSynthesizer, DomainProgram, DomainPrimitive
from cross_domain_agi_system import algorithmic_tasks, causal_tasks, grid_tasks


def test_adaptive_program_synthesizer_solves_algorithmic_tasks():
    synth = AdaptiveDomainProgramSynthesizer()

    results = [synth.solve_algorithmic(task) for task in algorithmic_tasks()]

    assert all(result.ok for result in results)
    assert all(result.program.source_hash for result in results)
    assert all(result.behavior_signature for result in results)


def test_adaptive_program_synthesizer_solves_causal_and_grid_tasks():
    synth = AdaptiveDomainProgramSynthesizer()

    causal = [synth.solve_causal(task) for task in causal_tasks()]
    grid = [synth.solve_grid(task) for task in grid_tasks()]

    assert all(result.ok for result in causal)
    assert all(result.ok for result in grid)
    assert {result.program.program_id for result in grid}.issubset(
        {"grid_bfs_shortest_path", "grid_greedy_manhattan"}
    )
    assert "grid_bfs_shortest_path" in {result.program.program_id for result in grid}


def test_failure_grammar_records_rejected_candidates():
    synth = AdaptiveDomainProgramSynthesizer()
    task = algorithmic_tasks()[0]
    bad = DomainProgram(
        "bad_identity",
        "algorithmic_synthesis",
        (DomainPrimitive("identity"),),
        complexity=1,
    )

    result = synth.sandbox.evaluate_algorithmic(bad, task)
    rule = synth.failure_grammar.update(result)

    assert not result.ok
    assert rule is not None
    assert synth.failure_grammar.penalty(bad) > 0.0
