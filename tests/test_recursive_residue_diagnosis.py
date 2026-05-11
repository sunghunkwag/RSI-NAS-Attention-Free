import argparse

import pytest

from afirsi_core import InstrumentMutationContract, OperatorGenerator as CoreOperatorGenerator
from bounded_rsi import (
    BoundedRSIKernel,
    RecursivePolicyVersion,
    RecursiveValidationResult,
    make_seed_policy,
)
from omega_adapter import OMEGAInstrumentGenerator
from omega_adapter.schemas import InstrumentPatch, StructuredAFIRSIResidue


def _small_args(**overrides):
    values = {
        "cycles": 2,
        "seeds": "7",
        "holdout_seeds": "23",
        "generations": 2,
        "population_size": 2,
        "d_model": 16,
        "train_steps": 1,
        "expansion_interval": 1,
        "pruning_interval": 1,
        "generated_min_evaluations": 1,
        "max_candidates": 2,
        "report_path": "bounded_rsi_diagnosis_test_report.json",
        "json": False,
        "verbose": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _policy(policy_id="P1", parent_policy_id="P0"):
    candidate = make_seed_policy()
    return RecursivePolicyVersion(
        policy_id=policy_id,
        policy_name=candidate.name,
        parent_policy_id=parent_policy_id,
        problem_space_version_id="psv-1",
        source_residue_ids=["r0"],
        source_patch_id="p0",
        eie_config={
            "grace_multiplier": 1.0,
            "probe_rate": 0.5,
            "clean_probe_first_eval": False,
            "meta_eval_gain": 0.1,
            "meta_archive_gain": 0.5,
            "meta_fitness_gain": 1.0,
            "meta_exploration_floor": 0.25,
        },
        generated_min_evaluations=1,
        creation_cycle=1,
        validation_summary={},
        accepted=True,
        rejection_reasons=[],
        lineage_depth=1,
        candidate=candidate,
    )


def _summary(name="candidate", delta=0.01, evals=3, archive=1):
    return {
        "name": name,
        "paired_or_holdout_seeds": 1,
        "mean_best_bpc": 7.0,
        "parent_mean_best_bpc": 7.1,
        "mean_bpc_delta": delta,
        "total_generated_evaluations": evals,
        "total_generated_archive_insertions": archive,
        "total_eie_residues": 1,
        "total_instrumented_candidates": 1,
        "total_meta_operator_policy_updates": 1,
        "total_instrument_mutations": 1,
        "eie_wins": 1 if delta > 0 else 0,
        "comparison": "parent",
        "counter_integrity_verified": True,
        "counter_integrity_violations": [],
    }


def _row(seed=7, evals=3, archive=1, weights=None):
    return {
        "seed": seed,
        "policy_id": "P1",
        "policy_name": "seed_policy",
        "best_bpc": 7.0,
        "generated_modules": 1,
        "eie_residues": 1,
        "instrument_mutations": 1,
        "instrumented_candidates": 1,
        "protected_pruning_attempts": 1,
        "meta_operator_policy_updates": 1,
        "meta_operator_weights": weights or {"compose": 0.5, "specialize": 0.5},
        "generated_evaluations": evals,
        "generated_archive_insertions": archive,
        "generated_records": [
            {
                "name": "gen_a",
                "evaluations": evals,
                "archive_insertions": archive,
                "birth_generation": 1,
                "source_action": "compose:gen_a",
                "elite_uses": 0,
                "best_fitness": 0.0,
                "prune_attempts": 1,
                "protected_from_prune": 1,
            }
        ],
        "problem_space_version": "psv-1",
        "failure_residue_ledger_entries": 1,
    }


def _source_residue(residue_type="PRUNING_PROPAGATION_RACE", residue_id="r1"):
    return {
        "residue_id": residue_id,
        "generation": 1,
        "seed": 7,
        "condition": "P1",
        "recursive_policy_id": "P1",
        "recursive_parent_policy_id": "P0",
        "residue_type": residue_type,
        "generated_module_name": "gen_a",
        "source_action": "compose:gen_a",
        "birth_generation": 1,
        "evaluations": 1,
        "archive_insertions": 0,
        "elite_uses": 0,
        "best_fitness": 0.0,
        "prune_attempts": 1,
        "protected_from_prune": 1,
        "age_at_prune_attempt": 0,
        "current_policy_snapshot": {},
        "observed_failure_signature": [residue_type],
        "requested_instrument_classes": ["evaluation_budget"],
    }


def _structured_residue(residue_type, residue_id):
    return StructuredAFIRSIResidue(
        residue_id=residue_id,
        generation=2,
        seed=7,
        condition="P1",
        residue_type=residue_type,
        generated_module_name=f"{residue_type}_subject",
        source_action="recursive_diagnostic",
        birth_generation=0,
        evaluations=3,
        archive_insertions=0,
        elite_uses=0,
        best_fitness=0.0,
        prune_attempts=0,
        protected_from_prune=0,
        age_at_prune_attempt=0,
        current_policy_snapshot={},
        observed_failure_signature=[residue_type],
        requested_instrument_classes=["evaluator_terms"],
        subject_type="recursive_policy_execution",
        parent_policy_id="P1",
        cycle_index=2,
    )


def _patch(patch_id="patch-a", updates=None, scaffold="clean_probe_until_min_evaluations"):
    return InstrumentPatch(
        patch_id=patch_id,
        candidate_name=f"candidate_{patch_id}",
        source_residue_ids=["r1"],
        parent_policy_name="P1",
        target_updates=updates or {
            "generated_min_evaluations": 2,
            "probe_rate": 1.0,
            "clean_probe_first_eval": True,
            "generated_module_scaffold_strategy": scaffold,
        },
        candidate_scoring_coefficients={},
        evaluator_terms=["generated_evidence_window"],
        archive_insertion_priority="prefer_under_evaluated_generated_modules",
        generated_module_scaffold_strategy=scaffold,
        constraints_interpreted=[],
        rationale=[],
        provenance={
            "patch_family": "no_eval_residue_budget",
            "parent_residue_types": ["PRUNING_PROPAGATION_RACE"],
        },
    )


def _result(**overrides):
    values = {
        "candidate_policy_id": "P1->C2.1",
        "candidate_policy_name": "candidate",
        "parent_policy_id": "P1",
        "comparison_parent_policy_id": "P1",
        "source_residue_ids": ["r1"],
        "patch_id": "patch-a",
        "contract_validated": True,
        "paired_evaluated": True,
        "holdout_evaluated": True,
        "paired_summary": _summary(delta=-0.01),
        "holdout_summary": _summary(name="holdout", delta=-0.02),
        "recursive_score": -0.1,
        "holdout_recursive_score": -0.1,
        "patch_family": "no_eval_residue_budget",
        "patch_fields_changed": [
            "clean_probe_first_eval",
            "generated_min_evaluations",
            "generated_module_scaffold_strategy",
            "probe_rate",
        ],
        "patch_equivalence_group": "eq-1",
        "patch_novel": True,
        "rejection_reasons": ["holdout recursive improvement score is not robust"],
    }
    values.update(overrides)
    return RecursiveValidationResult(**values)


@pytest.fixture(scope="module")
def bounded_report():
    return BoundedRSIKernel(_small_args()).run().to_json_dict()


def test_p2_rejection_creates_structured_diagnostic_record():
    kernel = BoundedRSIKernel(_small_args())
    diagnostics = []
    result = _result()

    kernel._finalize_rejected_result(
        result=result,
        parent=_policy(),
        cycle_index=2,
        source_residues=[_source_residue()],
        patch=_patch(),
        diagnostic_records=diagnostics,
    )

    assert diagnostics
    assert diagnostics[0].parent_policy_id == "P1"
    assert diagnostics[0].candidate_policy_id == "P1->C2.1"
    assert diagnostics[0].rejection_reasons[0]["code"]
    assert result.structured_rejection_reasons[0]["message"].startswith("holdout")


def test_archive_stagnation_emitted_from_execution_summary():
    kernel = BoundedRSIKernel(_small_args())
    residues = kernel._augment_parent_residues(
        cycle_index=2,
        parent=_policy(),
        parent_paired_summary=_summary(evals=4, archive=0),
        parent_paired_rows=[_row(evals=4, archive=0)],
        source_residues=[_source_residue()],
    )

    assert any(residue["residue_type"] == "ARCHIVE_STAGNATION" for residue in residues)


def test_evaluator_noise_or_overfit_emitted_when_holdout_fails():
    kernel = BoundedRSIKernel(_small_args())
    diagnostics = []
    result = _result(
        paired_summary=_summary(delta=0.03),
        holdout_summary=_summary(name="holdout", delta=-0.04),
        recursive_score=-0.2,
        holdout_recursive_score=-0.2,
        rejection_reasons=["holdout BPC regressed against parent"],
    )

    kernel._finalize_rejected_result(
        result=result,
        parent=_policy(),
        cycle_index=2,
        source_residues=[_source_residue()],
        patch=_patch(),
        diagnostic_records=diagnostics,
    )

    assert "EVALUATOR_NOISE_OR_OVERFIT" in {d.residue_type for d in diagnostics}


def test_operator_generator_mode_collapse_emitted_for_repeated_equivalent_patch():
    kernel = BoundedRSIKernel(_small_args())
    diagnostics = []
    result = _result(
        patch_novel=False,
        equivalent_failed_patch_ids=["failed-patch"],
        novelty_penalty=0.1,
        rejection_reasons=["patch repeats a recently failed equivalent signature"],
    )

    kernel._finalize_rejected_result(
        result=result,
        parent=_policy(),
        cycle_index=2,
        source_residues=[_source_residue()],
        patch=_patch(),
        diagnostic_records=diagnostics,
    )

    assert "OPERATOR_GENERATOR_MODE_COLLAPSE" in {d.residue_type for d in diagnostics}


def test_patch_effectiveness_failure_emitted_when_valid_patch_does_not_improve():
    kernel = BoundedRSIKernel(_small_args())
    diagnostics = []
    result = _result(
        paired_summary=_summary(delta=-0.02),
        holdout_summary=_summary(name="holdout", delta=-0.01),
        recursive_score=-0.2,
        holdout_recursive_score=-0.2,
    )

    kernel._finalize_rejected_result(
        result=result,
        parent=_policy(),
        cycle_index=2,
        source_residues=[_source_residue()],
        patch=_patch(),
        diagnostic_records=diagnostics,
    )

    assert "PATCH_EFFECTIVENESS_FAILURE" in {d.residue_type for d in diagnostics}


def test_operator_generator_produces_different_patch_families_for_residue_types():
    generator = OMEGAInstrumentGenerator()
    cfg = make_seed_policy().config
    archive_patch = generator.generate(
        [_structured_residue("ARCHIVE_STAGNATION", "r-archive")],
        cfg,
        1,
        cycle=2,
        parent_policy_name="P1",
    )[0]
    overfit_patch = generator.generate(
        [_structured_residue("EVALUATOR_NOISE_OR_OVERFIT", "r-overfit")],
        cfg,
        1,
        cycle=2,
        parent_policy_name="P1",
    )[0]

    assert archive_patch.target_updates != overfit_patch.target_updates
    assert archive_patch.provenance["patch_family"] != overfit_patch.provenance["patch_family"]
    assert "archive_priority_terms" in archive_patch.target_updates
    assert "evaluator_terms" in overfit_patch.target_updates
    InstrumentMutationContract().validate_patch(archive_patch)
    InstrumentMutationContract().validate_patch(overfit_patch)


def test_core_operator_generator_handles_second_order_residue_types():
    core_residue = _structured_residue("META_OPERATOR_IMBALANCE", "r-meta")
    from afirsi_core import FailureResidue

    failure = FailureResidue(
        residue_id=core_residue.residue_id,
        generation=2,
        residue_type=core_residue.residue_type,
        subject_type="recursive_policy_execution",
        subject_id="P1",
        triggering_event="recursive_diagnostic",
        observed_evidence={"evaluations": 3, "archive_insertions": 1},
        missing_evidence={},
        suspected_instrument_failure="operator imbalance",
        problem_space_version="psv-1",
        evaluator_version="recursive",
        observation_channel_version="recursive",
        proposed_mutation_targets=["meta_operator_weighting_coefficients"],
    )

    patches = CoreOperatorGenerator().generate([failure], parent_version_id="psv-1")

    assert patches
    assert patches[0].target_updates["meta_operator_weighting_coefficients"]
    InstrumentMutationContract().validate_patch(patches[0])


def test_equivalent_failed_patch_is_penalized_by_behavior_not_patch_id():
    kernel = BoundedRSIKernel(_small_args())
    parent = _policy()
    residues = [_source_residue()]
    patch_a = _patch("patch-a")
    first = kernel._patch_metadata(patch_a, parent, residues)
    kernel.failed_patch_memory.append({
        "patch_id": "patch-a",
        "equivalence_group": first["patch_equivalence_group"],
        "patch_family": first["patch_family"],
        "parent_policy_id": "P1",
        "cycle_index": 2,
        "residue_types": ["PRUNING_PROPAGATION_RACE"],
        "rejection_reasons": [],
    })
    patch_b = _patch("different-id-same-behavior")

    second = kernel._patch_metadata(patch_b, parent, residues)

    assert patch_a.patch_id != patch_b.patch_id
    assert second["patch_novel"] is False
    assert second["novelty_penalty"] > 0.0
    assert second["equivalent_failed_patch_ids"] == ["patch-a"]


def test_p2_candidates_are_generated_from_p1_derived_residues(bounded_report):
    cycle2 = bounded_report["cycles"][1]

    assert cycle2["parent_policy_id"] == "P1"
    assert all(
        residue["recursive_policy_id"] == "P1"
        for residue in cycle2["source_residues"]
    )


def test_p2_candidates_are_compared_against_p1_not_p0(bounded_report):
    cycle2 = bounded_report["cycles"][1]

    assert all(
        result["comparison_parent_policy_id"] == "P1"
        for result in cycle2["validation_results"]
        if result["paired_evaluated"]
    )


def test_bounded_success_requires_accepted_p2(bounded_report):
    cycle2 = bounded_report["cycles"][1]

    if bounded_report["bounded_recursive_success"]:
        assert cycle2["accepted_policy_id"] == "P2"
        assert bounded_report["accepted_policy_count"] >= 2
    else:
        assert cycle2["accepted_policy_id"] != "P2"


def test_holdout_gate_cannot_be_lowered_to_accept_paired_only_gain():
    kernel = BoundedRSIKernel(_small_args())
    result = _result(
        paired_summary=_summary(delta=0.05),
        holdout_summary=None,
        holdout_evaluated=False,
        recursive_score=1.0,
        holdout_recursive_score=-999.0,
        rejection_reasons=[],
    )

    failures = kernel._acceptance_failures(result)

    assert "holdout validation was not run" in failures


@pytest.mark.parametrize(
    "field_name",
    ["total_generated_evaluations", "total_generated_archive_insertions"],
)
def test_fake_activity_counters_are_rejected_by_score_integrity(field_name):
    kernel = BoundedRSIKernel(_small_args())
    summary = _summary()
    summary[field_name] = 999
    summary["counter_integrity_verified"] = False
    summary["counter_integrity_violations"] = [{"field": field_name}]

    with pytest.raises(ValueError):
        kernel.recursive_improvement_score(
            summary,
            valid_patch_rate=1.0,
            residue_to_patch_conversion_success=1.0,
            invalid_patch_count=0,
            holdout_summary=None,
        )


def test_repeated_patch_templates_cannot_clear_acceptance_without_novelty():
    kernel = BoundedRSIKernel(_small_args())
    result = _result(
        paired_summary=_summary(delta=0.05),
        holdout_summary=_summary(name="holdout", delta=0.05),
        recursive_score=1.0,
        holdout_recursive_score=1.0,
        patch_novel=False,
        novelty_penalty=0.20,
        rejection_reasons=[],
    )

    failures = kernel._acceptance_failures(result)

    assert "patch repeats a recently failed equivalent signature" in failures
