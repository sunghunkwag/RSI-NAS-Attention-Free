import argparse

import pytest

from afirsi_core import InstrumentMutationContract, ProblemSpaceVersionGraph
from bounded_rsi import (
    BoundedRSIKernel,
    RecursiveValidationResult,
    make_seed_policy,
)
from omega_adapter.schemas import InstrumentPatch


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
        "report_path": "bounded_rsi_test_report.json",
        "json": False,
        "verbose": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _positive_summary(name="candidate", delta=0.01):
    return {
        "name": name,
        "mean_best_bpc": 7.0,
        "parent_mean_best_bpc": 7.1,
        "mean_bpc_delta": delta,
        "total_generated_evaluations": 2,
        "total_generated_archive_insertions": 1,
        "total_eie_residues": 1,
        "total_instrumented_candidates": 1,
        "total_meta_operator_policy_updates": 1,
        "total_instrument_mutations": 1,
        "eie_wins": 1,
        "comparison": "parent",
    }


@pytest.fixture(scope="module")
def bounded_report():
    return BoundedRSIKernel(_small_args()).run().to_json_dict()


def test_p0_produces_real_residues_through_live_rsi_execution(bounded_report):
    cycle1 = bounded_report["cycles"][0]

    assert cycle1["parent_policy_id"] == "P0"
    assert cycle1["source_residue_ids"]
    assert all(
        residue["recursive_policy_id"] == "P0"
        and residue["residue_type"] == "PRUNING_PROPAGATION_RACE"
        for residue in cycle1["source_residues"]
    )
    assert cycle1["parent_paired_rows"][0]["failure_residue_ledger_entries"] > 0


def test_candidate_p1_is_generated_from_p0_residues(bounded_report):
    cycle1 = bounded_report["cycles"][0]
    p1 = next(
        policy for policy in cycle1["candidate_policies"]
        if policy["policy_id"] == "P1"
    )

    assert p1["source_residue_ids"] == cycle1["source_residue_ids"]
    assert p1["source_patch_id"] in cycle1["patch_ids"]
    assert p1["parent_policy_id"] == "P0"


def test_p1_cannot_be_accepted_without_contract_validation():
    kernel = BoundedRSIKernel(_small_args(cycles=1))
    result = RecursiveValidationResult(
        candidate_policy_id="C",
        candidate_policy_name="candidate",
        parent_policy_id="P0",
        comparison_parent_policy_id="P0",
        source_residue_ids=["r1"],
        patch_id="patch",
        contract_validated=False,
        paired_evaluated=True,
        holdout_evaluated=True,
        paired_summary=_positive_summary(),
        holdout_summary=_positive_summary(),
        recursive_score=1.0,
        holdout_recursive_score=1.0,
    )

    assert "patch was not contract validated" in kernel._acceptance_failures(result)


def test_p1_cannot_be_accepted_without_paired_seed_evaluation():
    kernel = BoundedRSIKernel(_small_args(cycles=1))
    result = RecursiveValidationResult(
        candidate_policy_id="C",
        candidate_policy_name="candidate",
        parent_policy_id="P0",
        comparison_parent_policy_id="P0",
        source_residue_ids=["r1"],
        patch_id="patch",
        contract_validated=True,
        paired_evaluated=False,
        holdout_evaluated=True,
        paired_summary=_positive_summary(),
        holdout_summary=_positive_summary(),
        recursive_score=1.0,
        holdout_recursive_score=1.0,
    )

    assert "paired-seed validation was not run" in kernel._acceptance_failures(result)


def test_p1_cannot_be_accepted_without_holdout_validation():
    kernel = BoundedRSIKernel(_small_args(cycles=1))
    result = RecursiveValidationResult(
        candidate_policy_id="C",
        candidate_policy_name="candidate",
        parent_policy_id="P0",
        comparison_parent_policy_id="P0",
        source_residue_ids=["r1"],
        patch_id="patch",
        contract_validated=True,
        paired_evaluated=True,
        holdout_evaluated=False,
        paired_summary=_positive_summary(),
        holdout_summary=None,
        recursive_score=1.0,
        holdout_recursive_score=-999.0,
    )

    assert "holdout validation was not run" in kernel._acceptance_failures(result)


def test_accepted_p1_becomes_parent_of_cycle_2(bounded_report):
    assert bounded_report["p1_became_parent_of_cycle_2"] is True
    assert bounded_report["cycles"][1]["parent_policy_id"] == "P1"
    assert bounded_report["cycles"][0]["accepted_policy_id"] == "P1"


def test_cycle_2_generates_p2_candidates_from_p1_derived_residues(bounded_report):
    cycle2 = bounded_report["cycles"][1]

    assert bounded_report["p2_generated_from_p1_derived_residues"] is True
    assert cycle2["source_residue_ids"]
    assert all(
        residue["recursive_policy_id"] == "P1"
        for residue in cycle2["source_residues"]
    )
    for candidate in cycle2["candidate_policies"]:
        assert candidate["source_residue_ids"] == cycle2["source_residue_ids"]


def test_p2_is_compared_against_p1_not_p0(bounded_report):
    cycle2 = bounded_report["cycles"][1]

    assert bounded_report["p2_compared_against_p1"] is True
    assert all(
        result["comparison_parent_policy_id"] == "P1"
        for result in cycle2["validation_results"]
        if result["paired_evaluated"]
    )


def test_recursive_lineage_records_p0_to_p1_to_p2_attempt(bounded_report):
    lineage = bounded_report["full_policy_lineage"]

    assert [policy["policy_id"] for policy in lineage] == ["P0", "P1", "P2"]
    assert lineage[1]["parent_policy_id"] == "P0"
    assert lineage[2]["parent_policy_id"] == "P1"
    assert bounded_report["cycles"][1]["patch_ids"]


class _InvalidPatchGenerator:
    def generate(self, residues, parent_config, parent_generated_min_evaluations, cycle, parent_policy_name):
        return [
            InstrumentPatch(
                patch_id="bad-patch",
                candidate_name="bad_candidate",
                source_residue_ids=[r.residue_id for r in residues],
                parent_policy_name=parent_policy_name,
                target_updates={"mean_bpc_delta": 999.0},
                candidate_scoring_coefficients={},
                evaluator_terms=[],
                archive_insertion_priority="baseline",
                generated_module_scaffold_strategy="baseline",
                constraints_interpreted=[],
                rationale=[],
                provenance={"source": "test"},
            )
        ]


def test_invalid_patches_do_not_create_accepted_policy_versions():
    graph = ProblemSpaceVersionGraph()
    report = BoundedRSIKernel(
        _small_args(cycles=1, max_candidates=1),
        generator=_InvalidPatchGenerator(),
        problem_space_graph=graph,
    ).run().to_json_dict()

    assert report["accepted_policy_count"] == 0
    assert report["cycles"][0]["candidate_policies"][0]["accepted"] is False
    assert "contract validation failed" in (
        report["cycles"][0]["validation_results"][0]["rejection_reasons"][0]
    )


def test_invalid_patches_do_not_create_accepted_problem_space_versions():
    graph = ProblemSpaceVersionGraph()
    BoundedRSIKernel(
        _small_args(cycles=1, max_candidates=1),
        generator=_InvalidPatchGenerator(),
        problem_space_graph=graph,
    ).run()

    assert graph.children_of("psv-0") == []


def test_rejected_candidates_do_not_get_problem_space_versions(bounded_report):
    rejected = [
        policy for cycle in bounded_report["cycles"]
        for policy in cycle["candidate_policies"]
        if not policy["accepted"]
    ]

    assert rejected
    assert all(policy["problem_space_version_id"] is None for policy in rejected)


def test_generated_evaluations_and_archive_insertions_come_from_engine_records(bounded_report):
    all_rows = []
    for cycle in bounded_report["cycles"]:
        all_rows.extend(cycle["parent_paired_rows"])
        all_rows.extend(cycle["parent_holdout_rows"])
        for result in cycle["validation_results"]:
            all_rows.extend(result["paired_rows"])
            all_rows.extend(result["holdout_rows"])

    assert all_rows
    assert any(row["generated_evaluations"] > 0 for row in all_rows)
    assert any(row["generated_archive_insertions"] > 0 for row in all_rows)
    for row in all_rows:
        assert row["generated_evaluations"] == sum(
            record["evaluations"] for record in row["generated_records"]
        )
        assert row["generated_archive_insertions"] == sum(
            record["archive_insertions"] for record in row["generated_records"]
        )


def test_feedback_is_required_for_later_cycle_parent_and_residue_source(bounded_report):
    cycle1, cycle2 = bounded_report["cycles"]

    assert cycle1["accepted_policy_id"] == "P1"
    assert cycle2["parent_policy_id"] == cycle1["accepted_policy_id"]
    assert all(
        residue["recursive_policy_id"] == cycle1["accepted_policy_id"]
        for residue in cycle2["source_residues"]
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "mean_bpc_delta",
        "generated_archive_insertions",
        "generated_evaluations",
        "success",
        "delete_residue",
        "bypass_build_train_evaluate",
        "hardcoded_seed",
    ],
)
def test_negative_control_contract_rejects_forbidden_patch_fields(field_name):
    patch = InstrumentPatch(
        patch_id=f"bad-{field_name}",
        candidate_name="bad",
        source_residue_ids=["r1"],
        parent_policy_name="P0",
        target_updates={field_name: 1},
        candidate_scoring_coefficients={},
        evaluator_terms=[],
        archive_insertion_priority="baseline",
        generated_module_scaffold_strategy="baseline",
        constraints_interpreted=[],
        rationale=[],
        provenance={"source": "test"},
    )

    with pytest.raises(ValueError):
        InstrumentMutationContract().validate_patch(patch)


def test_direct_policy_mutation_without_contract_validation_cannot_be_accepted():
    kernel = BoundedRSIKernel(_small_args(cycles=1))
    result = RecursiveValidationResult(
        candidate_policy_id="direct",
        candidate_policy_name="direct_mutation",
        parent_policy_id="P0",
        comparison_parent_policy_id="P0",
        source_residue_ids=["r1"],
        patch_id="direct-patch",
        contract_validated=False,
        paired_evaluated=True,
        holdout_evaluated=True,
        paired_summary=_positive_summary(),
        holdout_summary=_positive_summary(),
        recursive_score=10.0,
        holdout_recursive_score=10.0,
    )

    assert "patch was not contract validated" in kernel._acceptance_failures(result)


def test_child_version_is_created_only_for_accepted_patch(bounded_report):
    accepted_versions = [
        policy["problem_space_version_id"]
        for policy in bounded_report["full_policy_lineage"]
        if policy["policy_id"] != "P0"
    ]
    rejected_versions = [
        policy["problem_space_version_id"]
        for cycle in bounded_report["cycles"]
        for policy in cycle["candidate_policies"]
        if not policy["accepted"]
    ]

    assert accepted_versions == ["psv-1", "psv-2"]
    assert all(version is None for version in rejected_versions)


def test_seed_policy_starts_as_p0_without_source_residue_shortcut():
    candidate = make_seed_policy()

    assert candidate.patch.source_residue_ids == []
    assert candidate.generated_min_evaluations == 1
    assert candidate.config.probe_rate == 0.5
