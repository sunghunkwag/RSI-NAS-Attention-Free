"""Residue-conditioned OMEGA-style instrument synthesis."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from statistics import mean
from typing import Dict, Iterable, List

from rsi_nas import EIEConfig

from .schemas import InstrumentPatch, StructuredAFIRSIResidue


class OMEGAInstrumentGenerator:
    """Generate EIE instrument patches from observed failure residue content.

    This mirrors the OMEGA-THDSE style at the adapter level: residue facts are
    converted into symbolic constraints, the missing instrument class is
    interpreted, and deterministic patches are synthesized from those facts.
    No candidate is emitted when no real residue is provided.
    """

    def generate(
        self,
        residues: List[StructuredAFIRSIResidue],
        parent_config: EIEConfig,
        parent_generated_min_evaluations: int,
        cycle: int,
        parent_policy_name: str,
    ) -> List[InstrumentPatch]:
        if not residues:
            return []

        facts = self._interpret_residues(residues)
        patches: List[InstrumentPatch] = []
        if facts["premature_no_eval_count"] > 0:
            patches.append(self._no_eval_patch(
                residues, facts, parent_config,
                parent_generated_min_evaluations, cycle, parent_policy_name,
            ))
        if facts["archive_starvation_count"] > 0:
            patches.append(self._archive_starvation_patch(
                residues, facts, parent_config,
                parent_generated_min_evaluations, cycle, parent_policy_name,
            ))
        if facts["fitness_without_elite_count"] > 0:
            patches.append(self._fitness_no_elite_patch(
                residues, facts, parent_config,
                parent_generated_min_evaluations, cycle, parent_policy_name,
            ))
        if facts["repeated_protection_count"] >= 2:
            patches.append(self._residue_pressure_patch(
                residues, facts, parent_config,
                parent_generated_min_evaluations, cycle, parent_policy_name,
            ))

        unique: Dict[str, InstrumentPatch] = {}
        for patch in patches:
            unique[patch.patch_id] = patch
        return list(unique.values())

    def _interpret_residues(
        self, residues: Iterable[StructuredAFIRSIResidue]
    ) -> Dict:
        rows = list(residues)
        signatures = Counter(
            sig for row in rows for sig in row.observed_failure_signature
        )
        requested = Counter(
            cls for row in rows for cls in row.requested_instrument_classes
        )
        source_kinds = Counter(
            row.source_action.split(":", 1)[0] for row in rows
        )
        no_eval = [
            row for row in rows
            if "premature_prune_before_evaluation"
            in row.observed_failure_signature
        ]
        archive_starved = [
            row for row in rows
            if "archive_starvation_after_evaluation"
            in row.observed_failure_signature
        ]
        fitness_no_elite = [
            row for row in rows
            if "fitness_without_elite_use" in row.observed_failure_signature
        ]
        repeated = [
            row for row in rows
            if "repeated_prune_protection" in row.observed_failure_signature
        ]

        return {
            "total_residues": len(rows),
            "signature_counts": dict(signatures),
            "requested_instrument_counts": dict(requested),
            "source_kind_counts": dict(source_kinds),
            "premature_no_eval_count": len(no_eval),
            "archive_starvation_count": len(archive_starved),
            "fitness_without_elite_count": len(fitness_no_elite),
            "repeated_protection_count": len(repeated),
            "max_age_at_prune": max(row.age_at_prune_attempt for row in rows),
            "mean_evaluations": mean(row.evaluations for row in rows),
            "mean_archive_insertions": mean(row.archive_insertions for row in rows),
            "max_best_fitness": max(row.best_fitness for row in rows),
            "residue_ids": [row.residue_id for row in rows],
        }

    def _base_provenance(
        self,
        residues: List[StructuredAFIRSIResidue],
        facts: Dict,
        cycle: int,
        parent_policy_name: str,
        interpretation: str,
    ) -> Dict:
        return {
            "source": "OMEGA-THDSE-style deterministic symbolic synthesis",
            "omega_style": {
                "residue_facts": facts,
                "constraint_interpretation": interpretation,
                "causal_input_residue_ids": [r.residue_id for r in residues],
            },
            "cycle": cycle,
            "parent_policy_name": parent_policy_name,
        }

    def _patch_id(self, interpretation: str, updates: Dict, residue_ids: List[str]) -> str:
        payload = {
            "interpretation": interpretation,
            "updates": updates,
            "residue_ids": residue_ids,
        }
        digest = hashlib.blake2b(
            json.dumps(payload, sort_keys=True).encode("utf-8"),
            digest_size=5,
        ).hexdigest()
        return f"omega-{interpretation}-{digest}"

    def _make_patch(
        self,
        residues: List[StructuredAFIRSIResidue],
        facts: Dict,
        cycle: int,
        parent_policy_name: str,
        interpretation: str,
        target_updates: Dict,
        scoring_coefficients: Dict[str, float],
        evaluator_terms: List[str],
        archive_priority: str,
        scaffold_strategy: str,
        constraints: List[str],
        rationale: List[str],
    ) -> InstrumentPatch:
        residue_ids = [r.residue_id for r in residues]
        patch_id = self._patch_id(interpretation, target_updates, residue_ids)
        return InstrumentPatch(
            patch_id=patch_id,
            candidate_name=f"cycle{cycle}_{interpretation}_{patch_id.rsplit('-', 1)[-1]}",
            source_residue_ids=residue_ids,
            parent_policy_name=parent_policy_name,
            target_updates=target_updates,
            candidate_scoring_coefficients=scoring_coefficients,
            evaluator_terms=evaluator_terms,
            archive_insertion_priority=archive_priority,
            generated_module_scaffold_strategy=scaffold_strategy,
            constraints_interpreted=constraints,
            rationale=rationale,
            provenance=self._base_provenance(
                residues, facts, cycle, parent_policy_name, interpretation
            ),
        )

    def _no_eval_patch(
        self,
        residues: List[StructuredAFIRSIResidue],
        facts: Dict,
        cfg: EIEConfig,
        parent_min_evals: int,
        cycle: int,
        parent_policy_name: str,
    ) -> InstrumentPatch:
        no_eval_ratio = facts["premature_no_eval_count"] / facts["total_residues"]
        target_min = max(parent_min_evals + 2, 3 if no_eval_ratio >= 0.25 else 2)
        target_grace = max(cfg.grace_multiplier, 4.0 if no_eval_ratio >= 0.25 else 3.0)
        updates = {
            "grace_multiplier": round(target_grace, 4),
            "probe_rate": 1.0,
            "clean_probe_first_eval": True,
            "meta_eval_gain": max(cfg.meta_eval_gain, 0.35),
            "meta_archive_gain": max(cfg.meta_archive_gain, 2.0),
            "generated_min_evaluations": target_min,
            "generated_module_scaffold_strategy": "clean_probe_until_min_evaluations",
            "archive_insertion_priority": "prefer_under_evaluated_generated_modules",
        }
        return self._make_patch(
            residues, facts, cycle, parent_policy_name,
            "no_eval_residue_budget",
            updates,
            {
                "generated_archive_insertions": 0.0030,
                "generated_evaluations": 0.0005,
                "meta_operator_policy_updates": 0.0004,
            },
            ["generated_evaluation_budget", "archive_evidence_bonus"],
            "prefer_under_evaluated_generated_modules",
            "clean_probe_until_min_evaluations",
            [
                "if residue.evaluations == 0 before prune: require clean probe",
                "if prune age is inside evidence window: increase grace budget",
                "if real archive evidence exists: reward archive insertions",
            ],
            [
                "Premature pruning before evaluation means the instrument lacks an evidence budget.",
                "The patch increases generated-module evaluation pressure and keeps real SGD evaluation mandatory.",
            ],
        )

    def _archive_starvation_patch(
        self,
        residues: List[StructuredAFIRSIResidue],
        facts: Dict,
        cfg: EIEConfig,
        parent_min_evals: int,
        cycle: int,
        parent_policy_name: str,
    ) -> InstrumentPatch:
        pressure = math.log1p(facts["archive_starvation_count"])
        updates = {
            "probe_rate": 1.0,
            "clean_probe_first_eval": True,
            "meta_archive_gain": round(max(cfg.meta_archive_gain, 2.0 + pressure), 4),
            "meta_eval_gain": round(max(cfg.meta_eval_gain, 0.25 + 0.05 * pressure), 4),
            "generated_min_evaluations": max(parent_min_evals + 1, 2),
            "archive_insertion_priority": "raise_starved_generated_modules",
        }
        return self._make_patch(
            residues, facts, cycle, parent_policy_name,
            "archive_starvation_reweight",
            updates,
            {
                "generated_archive_insertions": 0.0035,
                "generated_evaluations": 0.0003,
            },
            ["archive_starvation_penalty", "generated_archive_bonus"],
            "raise_starved_generated_modules",
            "clean_probe_then_archive_reweight",
            [
                "if evaluations > 0 and archive_insertions == 0: raise archive weighting",
                "paired and holdout seeds must still use real archive insertion evidence",
            ],
            [
                "Evaluated generated modules that never enter MAP-Elites indicate archive starvation.",
                "The patch shifts the meta-policy toward archive-producing generation operators.",
            ],
        )

    def _fitness_no_elite_patch(
        self,
        residues: List[StructuredAFIRSIResidue],
        facts: Dict,
        cfg: EIEConfig,
        parent_min_evals: int,
        cycle: int,
        parent_policy_name: str,
    ) -> InstrumentPatch:
        updates = {
            "probe_rate": 1.0,
            "meta_fitness_gain": round(max(cfg.meta_fitness_gain, 5.0), 4),
            "meta_archive_gain": round(max(cfg.meta_archive_gain, 1.75), 4),
            "meta_exploration_floor": round(max(cfg.meta_exploration_floor * 0.8, 0.10), 4),
            "generated_min_evaluations": max(parent_min_evals, 1),
            "archive_insertion_priority": "prefer_high_fitness_low_elite_use",
        }
        return self._make_patch(
            residues, facts, cycle, parent_policy_name,
            "fitness_without_elite_bridge",
            updates,
            {
                "generated_archive_insertions": 0.0020,
                "eie_wins": 0.0060,
            },
            ["fitness_without_elite_use", "elite_use_gap"],
            "prefer_high_fitness_low_elite_use",
            "mixed_probe_with_fitness_reweight",
            [
                "if best_fitness > 0 and elite_uses == 0: raise fitness weighting",
                "do not keep protection unless BPC and holdout evidence remain valid",
            ],
            [
                "Some generated modules show fitness evidence but fail to become elite components.",
                "The patch raises fitness weighting while preserving archive and holdout gates.",
            ],
        )

    def _residue_pressure_patch(
        self,
        residues: List[StructuredAFIRSIResidue],
        facts: Dict,
        cfg: EIEConfig,
        parent_min_evals: int,
        cycle: int,
        parent_policy_name: str,
    ) -> InstrumentPatch:
        pressure = min(2.0, facts["repeated_protection_count"] / max(1, facts["total_residues"]))
        updates = {
            "grace_multiplier": round(max(cfg.grace_multiplier, 3.0 + pressure), 4),
            "probe_rate": 1.0,
            "meta_eval_gain": round(max(cfg.meta_eval_gain, 0.30 + pressure * 0.10), 4),
            "generated_min_evaluations": max(parent_min_evals + 1, 2),
            "candidate_scoring_coefficients": "residue_pressure_active",
        }
        return self._make_patch(
            residues, facts, cycle, parent_policy_name,
            "residue_pressure_disagreement",
            updates,
            {
                "meta_operator_policy_updates": 0.0008,
                "generated_evaluations": 0.0004,
            },
            ["residue_pressure", "evaluator_disagreement"],
            "prefer_residue_reducing_generated_modules",
            "probe_then_compare_evaluator_disagreement",
            [
                "if repeated protection accumulates: add evaluator-disagreement pressure",
                "policy updates must be observed; direct success flags remain forbidden",
            ],
            [
                "Repeated protection means the instrument keeps seeing the same failure boundary.",
                "The patch rewards real policy updates and generated-module evaluations.",
            ],
        )
