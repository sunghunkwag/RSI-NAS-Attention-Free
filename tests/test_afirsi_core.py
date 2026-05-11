import pytest

from afirsi_core import (
    Evaluator,
    FailureResidue,
    FailureResidueLedger,
    InstrumentMutationContract,
    InstrumentPatch,
    ObservationChannel,
    OperatorGenerator,
    ProblemSpaceVersionGraph,
)
from rsi_nas import (
    ArchitectureGenome,
    ArchitectureGrammar,
    ArchitectureMeta,
    GatedFFN,
    LayerGene,
    ModuleRegistry,
    ModuleSpec,
)


D = 16


def _core_residue(residue_id="r1"):
    return FailureResidue(
        residue_id=residue_id,
        run_id="test-run",
        generation=3,
        residue_type="PRUNING_PROPAGATION_RACE",
        subject_type="generated_module",
        subject_id="gen_a",
        triggering_event="prune_attempt",
        observed_evidence={
            "birth_generation": 3,
            "age": 0,
            "evaluations": 0,
            "archive_insertions": 0,
            "elite_uses": 0,
            "prune_attempts": 1,
            "protected_prune_attempts": 0,
            "source_action": "compose:gen_a",
        },
        missing_evidence={
            "required_age_generations": 9,
            "observed_age_generations": 0,
            "required_evaluations": 2,
            "observed_evaluations": 0,
            "under_age": True,
            "under_evaluated": True,
        },
        suspected_instrument_failure="pruning before evidence",
        problem_space_version="psv-0",
        evaluator_version="evaluator.pruning_propagation_race.v1",
        observation_channel_version="observation.generated_module_lifecycle.v1",
        proposed_mutation_targets=[
            "generated_grace_generations",
            "generated_min_evaluations",
            "generated_probe_rate",
        ],
        severity="high",
        confidence=0.95,
    )


def _generated_record(name="gen_a", birth_generation=3):
    registry = ModuleRegistry()
    spec = ModuleSpec(
        name=name,
        builder=lambda d, **kw: GatedFFN(d),
        default_kwargs={},
        param_cost=1.0,
        is_generated=True,
    )
    registry.register(
        spec,
        birth_generation=birth_generation,
        source_action=f"compose:{name}",
    )
    return registry, registry.generated_record(name)


def test_failure_residue_ledger_records_and_queries_unresolved():
    ledger = FailureResidueLedger()
    residue = ledger.record(_core_residue())

    assert ledger.query_by_type("PRUNING_PROPAGATION_RACE") == [residue]
    assert ledger.unresolved_residues() == [residue]
    assert residue.addressed is False


def test_failure_residue_ledger_marks_addressed_by_patch_id():
    ledger = FailureResidueLedger()
    residue = ledger.record(_core_residue())

    ledger.mark_addressed(residue.residue_id, "patch-1")

    assert ledger.unresolved_residues() == []
    assert ledger.get(residue.residue_id).addressed_by_patch_id == "patch-1"


def test_observation_channel_captures_generated_module_lifecycle():
    registry, record = _generated_record()
    generated_genome = ArchitectureGenome(
        layers=[LayerGene("gen_a")],
        d_model=D,
    )
    registry.record_genome_evaluation(
        genome=generated_genome,
        generation=4,
        fitness=0.25,
        inserted=True,
    )
    registry.record_elite_usage([generated_genome, generated_genome], generation=5)
    channel = ObservationChannel()

    observation = channel.observe_generated_module(
        record=record,
        generation=5,
        triggering_event="prune_attempt",
        policy={
            "generated_grace_generations": 9,
            "generated_min_evaluations": 2,
            "generated_probe_rate": 1.0,
        },
        problem_space_version="psv-1",
    )

    assert observation.birth_generation == 3
    assert observation.age == 2
    assert observation.evaluations == 1
    assert observation.archive_insertions == 1
    assert observation.elite_uses == 2
    assert observation.protected_prune_attempts == 0
    assert observation.best_fitness == 0.25
    assert observation.last_evaluated_generation == 4


def test_evaluator_emits_pruning_propagation_race_for_under_observed_module():
    _, record = _generated_record(birth_generation=5)
    record.prune_attempts = 1
    observation = ObservationChannel().observe_generated_module(
        record=record,
        generation=5,
        triggering_event="prune_attempt",
        policy={
            "generated_grace_generations": 0,
            "generated_min_evaluations": 0,
            "generated_probe_rate": 0.0,
        },
        problem_space_version="psv-0",
    )
    evaluator = Evaluator(expansion_interval=5, min_evaluations=2)

    residues = evaluator.evaluate([observation])

    assert len(residues) == 1
    assert residues[0].residue_type == "PRUNING_PROPAGATION_RACE"
    assert residues[0].module_name == "gen_a"
    assert residues[0].missing_evidence["required_age_generations"] == 15
    assert residues[0].missing_evidence["required_evaluations"] == 2


def test_instrument_mutation_contract_rejects_forbidden_patches():
    patch = InstrumentPatch(
        patch_id="bad",
        source_residue_ids=["r1"],
        parent_version_id="psv-0",
        target_updates={"mean_bpc_delta": 100.0},
    )

    with pytest.raises(ValueError):
        InstrumentMutationContract().validate_patch(patch)


def test_instrument_mutation_contract_accepts_allowed_policy_patch():
    patch = InstrumentPatch(
        patch_id="good",
        source_residue_ids=["r1"],
        parent_version_id="psv-0",
        target_updates={
            "generated_grace_generations": 10,
            "generated_min_evaluations": 2,
            "generated_probe_rate": 1.0,
            "clean_probe_first_eval": True,
            "evaluator_terms": ["generated_evidence_window"],
            "archive_insertion_priority": (
                "prefer_under_evaluated_generated_modules"
            ),
            "meta_eval_gain": 0.35,
        },
    )

    InstrumentMutationContract().validate_patch(patch)


def test_problem_space_version_graph_records_parent_child_lineage():
    graph = ProblemSpaceVersionGraph(root_policy={"generated_probe_rate": 0.0})
    child = graph.create_child(
        parent_version_id="psv-0",
        patch_id="patch-1",
        residue_ids=["r1"],
        observation_channel_version="obs.v1",
        evaluator_version="eval.v1",
        instrument_policy={"generated_probe_rate": 1.0},
    )

    lineage = graph.lineage(child.version_id)

    assert [version.version_id for version in lineage] == ["psv-0", child.version_id]
    assert child.parent_version_id == "psv-0"
    assert child.patch_id == "patch-1"
    assert child.residue_ids == ["r1"]


def test_operator_generator_produces_patch_from_real_residue():
    residue = _core_residue()
    generator = OperatorGenerator()

    patches = generator.generate(
        [residue],
        instrument_policy={
            "generated_grace_generations": 0,
            "generated_min_evaluations": 0,
            "generated_probe_rate": 0.0,
        },
        parent_version_id="psv-0",
    )

    assert len(patches) == 1
    patch = patches[0]
    assert patch.source_residue_ids == [residue.residue_id]
    assert patch.target_updates["generated_grace_generations"] == 9
    assert patch.target_updates["generated_min_evaluations"] == 2
    assert patch.target_updates["generated_probe_rate"] == 1.0
    assert "archive_insertions" not in patch.target_updates
    InstrumentMutationContract().validate_patch(patch)


def test_afirsi_core_integrates_with_rsi_nas_pruning_race_path():
    registry = ModuleRegistry()
    grammar = ArchitectureGrammar(registry)
    meta = ArchitectureMeta(
        registry,
        grammar,
        enable_eie=True,
        expansion_interval=5,
        generated_min_evaluations=2,
    )
    meta.set_generation(5)
    spec = ModuleSpec(
        name="race_gen",
        builder=lambda d, **kw: GatedFFN(d),
        default_kwargs={},
        param_cost=1.0,
        is_generated=True,
    )
    registry.register(
        spec,
        birth_generation=5,
        source_action="compose:race_gen",
    )
    parent_version = meta.problem_space_graph.active_version_id

    pruned = meta.prune_unused(
        [ArchitectureGenome(layers=[LayerGene("nca_step")], d_model=D)],
        generation=5,
    )

    record = registry.generated_record("race_gen")
    residue = meta.pruning_residues[0]
    child_version = meta.problem_space_graph.current_version()
    instrumented = meta.instrument_candidate(
        ArchitectureGenome(layers=[LayerGene("nca_step")], d_model=D)
    )

    assert pruned == []
    assert registry.get("race_gen") is not None
    assert record.prune_attempts == 1
    assert record.protected_from_prune == 1
    assert residue.residue_type == "PRUNING_PROPAGATION_RACE"
    assert residue.addressed is True
    assert residue.addressed_by_patch_id == child_version.patch_id
    assert len(meta.failure_residue_ledger.query_by_type(
        "PRUNING_PROPAGATION_RACE"
    )) == 1
    assert child_version.parent_version_id == parent_version
    assert child_version.residue_ids == [residue.residue_id]
    assert (
        child_version.active_instrument_policy["pruning_policy"][
            "generated_probe_rate"
        ]
        == 1.0
    )
    assert meta.pruning_policy.generated_grace_generations == 15
    assert meta.pruning_policy.generated_min_evaluations == 2
    assert meta.pruning_policy.generated_probe_rate == 1.0
    assert any(layer.module_name == "race_gen" for layer in instrumented.layers)
