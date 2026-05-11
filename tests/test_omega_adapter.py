import logging
import random

import numpy as np
import torch

from omega_adapter import OMEGAInstrumentGenerator, apply_instrument_patch, export_residues
from omega_adapter.schemas import StructuredAFIRSIResidue
from open_ended_rsi_omega import (
    build_provenance_chain,
    final_validation_valid,
    passes_acceptance_gate,
)
from rsi_nas import EIEConfig, build_rsi_nas


def _sample_residue(
    residue_id="r-no-eval",
    signatures=None,
    requested=None,
    evaluations=0,
    archive_insertions=0,
    elite_uses=0,
    best_fitness=0.0,
    protected_from_prune=1,
):
    signatures = signatures or [
        "PRUNING_PROPAGATION_RACE",
        "premature_prune_before_evaluation",
    ]
    requested = requested or [
        "evaluation_budget",
        "generated_module_scaffold_strategy",
        "pruning_grace",
    ]
    return StructuredAFIRSIResidue(
        residue_id=residue_id,
        generation=1,
        seed=7,
        condition="test",
        residue_type="PRUNING_PROPAGATION_RACE",
        generated_module_name=f"gen_{residue_id}",
        source_action="compose:gen",
        birth_generation=1,
        evaluations=evaluations,
        archive_insertions=archive_insertions,
        elite_uses=elite_uses,
        best_fitness=best_fitness,
        prune_attempts=1,
        protected_from_prune=protected_from_prune,
        age_at_prune_attempt=0,
        current_policy_snapshot={
            "eie_config": {
                "grace_multiplier": 1.0,
                "probe_rate": 0.5,
                "clean_probe_first_eval": False,
                "meta_eval_gain": 0.1,
                "meta_archive_gain": 0.5,
                "meta_fitness_gain": 1.0,
                "meta_exploration_floor": 0.25,
            },
            "pruning_policy": {
                "generated_grace_generations": 0,
                "generated_min_evaluations": 0,
                "generated_probe_rate": 0.0,
                "mutation_count": 0,
            },
        },
        observed_failure_signature=signatures,
        requested_instrument_classes=requested,
    )


def _valid_summary(name="candidate", score=1.0):
    return {
        "name": name,
        "mechanism_valid": True,
        "mean_bpc_delta": 0.01,
        "score": score,
        "total_generated_evaluations": 5,
        "total_generated_archive_insertions": 2,
        "total_meta_operator_policy_updates": 1,
        "total_eie_residues": 1,
    }


def test_residue_export_contains_real_generated_module_evidence():
    logging.getLogger().setLevel(logging.WARNING)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    engine = build_rsi_nas(
        d_model=16,
        train_steps=1,
        expansion_interval=1,
        pruning_interval=1,
        generated_min_evaluations=1,
        enable_eie=True,
        eie_config=EIEConfig(
            grace_multiplier=1.0,
            probe_rate=1.0,
            clean_probe_first_eval=True,
        ),
    )
    engine.run(generations=2, population_size=2)
    residues = export_residues(engine, seed=7, condition="test")

    assert len(residues) >= 1
    residue = residues[0]
    assert residue.residue_id.startswith("afirsi-residue-")
    assert residue.residue_type == "PRUNING_PROPAGATION_RACE"
    assert residue.generated_module_name.startswith(("seq_", "fused_", "spec_"))
    assert residue.prune_attempts >= 1
    assert residue.protected_from_prune >= 1
    assert "current_policy_snapshot" not in residue.observed_failure_signature
    assert "pruning_grace" in residue.requested_instrument_classes


def test_empty_residue_produces_no_success_candidate():
    generator = OMEGAInstrumentGenerator()
    candidates = generator.generate([], EIEConfig(), 1, cycle=1, parent_policy_name="p")

    assert candidates == []


def test_omega_generator_is_residue_conditioned():
    generator = OMEGAInstrumentGenerator()
    no_eval = [_sample_residue()]
    archive_starved = [_sample_residue(
        residue_id="r-archive",
        signatures=[
            "PRUNING_PROPAGATION_RACE",
            "archive_starvation_after_evaluation",
        ],
        requested=["archive_insertion_priority", "meta_archive_weighting"],
        evaluations=3,
        archive_insertions=0,
    )]

    no_eval_patches = generator.generate(no_eval, EIEConfig(), 1, 1, "parent")
    archive_patches = generator.generate(archive_starved, EIEConfig(), 1, 1, "parent")

    assert {p.patch_id for p in no_eval_patches} != {
        p.patch_id for p in archive_patches
    }
    assert no_eval_patches[0].target_updates["generated_min_evaluations"] > 1
    assert archive_patches[0].target_updates["meta_archive_gain"] >= 2.0


def test_different_residues_produce_different_policy_patches():
    generator = OMEGAInstrumentGenerator()
    residue_a = _sample_residue(residue_id="a", protected_from_prune=1)
    residue_b = _sample_residue(
        residue_id="b",
        signatures=[
            "PRUNING_PROPAGATION_RACE",
            "fitness_without_elite_use",
        ],
        requested=["fitness_weighting", "elite_use_weighting"],
        evaluations=4,
        archive_insertions=1,
        elite_uses=0,
        best_fitness=0.25,
    )

    patch_a = generator.generate([residue_a], EIEConfig(), 1, 1, "p")[0]
    patch_b = generator.generate([residue_b], EIEConfig(), 1, 1, "p")[0]

    assert patch_a.target_updates != patch_b.target_updates
    assert patch_a.constraints_interpreted != patch_b.constraints_interpreted


def test_policy_patch_changes_actual_eie_config():
    generator = OMEGAInstrumentGenerator()
    parent = EIEConfig(
        grace_multiplier=1.0,
        probe_rate=0.5,
        clean_probe_first_eval=False,
        meta_eval_gain=0.1,
        meta_archive_gain=0.5,
        meta_fitness_gain=1.0,
        meta_exploration_floor=0.25,
    )
    patch = generator.generate([_sample_residue()], parent, 1, 1, "parent")[0]
    candidate = apply_instrument_patch("parent", parent, 1, patch)

    assert candidate.config.probe_rate == 1.0
    assert candidate.config.clean_probe_first_eval is True
    assert candidate.config.grace_multiplier > parent.grace_multiplier
    assert candidate.generated_min_evaluations > 1


def test_open_ended_omega_requires_holdout_validation():
    paired = _valid_summary(score=2.0)
    holdout = _valid_summary(name="holdout", score=1.0)
    holdout["mechanism_valid"] = False
    champion = _valid_summary(name="champion", score=1.0)

    accepted, failures = passes_acceptance_gate(paired, holdout, champion)

    assert accepted is False
    assert "holdout mechanism invalid" in failures


def test_no_static_three_candidate_cheat():
    generator = OMEGAInstrumentGenerator()
    residues = [
        _sample_residue(residue_id="a"),
        _sample_residue(
            residue_id="b",
            signatures=[
                "PRUNING_PROPAGATION_RACE",
                "archive_starvation_after_evaluation",
            ],
            requested=["archive_insertion_priority", "meta_archive_weighting"],
            evaluations=2,
        ),
        _sample_residue(
            residue_id="c",
            signatures=[
                "PRUNING_PROPAGATION_RACE",
                "fitness_without_elite_use",
            ],
            requested=["fitness_weighting", "elite_use_weighting"],
            evaluations=3,
            archive_insertions=1,
            best_fitness=0.2,
        ),
    ]
    patches = generator.generate(residues, EIEConfig(), 1, 1, "parent")
    old_names = {
        "cycle1_clean_probe_archive",
        "cycle1_fitness_weighted",
        "cycle1_evidence_budget",
    }

    assert len(patches) >= 3
    assert old_names.isdisjoint({patch.candidate_name for patch in patches})
    assert len({patch.patch_id for patch in patches}) == len(patches)


def test_no_success_without_archive_insertions():
    paired = _valid_summary(score=2.0)
    paired["total_generated_archive_insertions"] = 0
    holdout = _valid_summary(name="holdout", score=1.0)
    champion = _valid_summary(name="champion", score=1.0)

    accepted, failures = passes_acceptance_gate(paired, holdout, champion)

    assert accepted is False
    assert "paired missing archive insertions" in failures


def test_no_success_without_meta_policy_updates():
    paired = _valid_summary(score=2.0)
    paired["total_meta_operator_policy_updates"] = 0
    holdout = _valid_summary(name="holdout", score=1.0)
    champion = _valid_summary(name="champion", score=1.0)

    accepted, failures = passes_acceptance_gate(paired, holdout, champion)

    assert accepted is False
    assert "paired missing meta-policy updates" in failures


def test_no_success_without_real_exported_residue():
    champion = _valid_summary(name="champion", score=2.0)

    accepted = final_validation_valid(
        accepted_count=1,
        required_accepted_improvements=1,
        bootstrap_residue_count=0,
        champion_summary=champion,
    )

    assert accepted is False


def test_no_success_without_generated_module_evaluations():
    champion = _valid_summary(name="champion", score=2.0)
    champion["total_generated_evaluations"] = 0

    accepted = final_validation_valid(
        accepted_count=1,
        required_accepted_improvements=1,
        bootstrap_residue_count=2,
        champion_summary=champion,
    )

    assert accepted is False


def test_candidate_names_do_not_determine_success():
    paired = _valid_summary(name="cycle1_clean_probe_archive", score=2.0)
    paired["total_generated_archive_insertions"] = 0
    holdout = _valid_summary(name="cycle1_clean_probe_archive", score=2.0)
    champion = _valid_summary(name="champion", score=1.0)

    accepted, failures = passes_acceptance_gate(paired, holdout, champion)

    assert accepted is False
    assert "paired missing archive insertions" in failures


def test_report_contains_provenance_chain():
    generator = OMEGAInstrumentGenerator()
    patch = generator.generate([_sample_residue()], EIEConfig(), 1, 1, "parent")[0]
    candidate = apply_instrument_patch("parent", EIEConfig(), 1, patch)
    paired = _valid_summary(name=candidate.name, score=2.0)
    holdout = _valid_summary(name=candidate.name, score=1.0)

    chain = build_provenance_chain(
        candidate,
        paired,
        holdout,
        accepted=True,
        current_champion_name="parent",
    )

    assert chain["residue_ids"] == patch.source_residue_ids
    assert chain["generated_patch_id"] == patch.patch_id
    assert chain["paired_evaluation"] == candidate.name
    assert chain["holdout_evaluation"] == candidate.name
    assert chain["decision"] == "eligible"
    assert chain["next_parent_policy"] == candidate.name
