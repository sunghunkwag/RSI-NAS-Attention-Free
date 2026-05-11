"""Residue-conditioned AFIRSI instrument patch generation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

from .mutation_contract import InstrumentPatch
from .provenance import ProvenanceRecord
from .residue import FailureResidue


class OperatorGenerator:
    """Proposes instrument patches from residue without mutating live state."""

    def __init__(self, version: str = "operator_generator.multi_bottleneck.v2") -> None:
        self.version = version

    def generate(
        self,
        residues: Iterable[FailureResidue],
        instrument_policy: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
        parent_version_id: str = "psv-0",
    ) -> List[InstrumentPatch]:
        all_rows = list(residues)
        if not all_rows:
            return []
        rows = [
            residue
            for residue in all_rows
            if residue.residue_type == "PRUNING_PROPAGATION_RACE"
        ]

        policy = dict(instrument_policy or {})
        patches: List[InstrumentPatch] = []
        if rows:
            patches.append(self._pruning_race_patch(
                rows=rows,
                policy=policy,
                config=config,
                parent_version_id=parent_version_id,
            ))
        for residue_type in sorted({r.residue_type for r in all_rows}):
            if residue_type == "PRUNING_PROPAGATION_RACE":
                continue
            typed = [r for r in all_rows if r.residue_type == residue_type]
            patch = self._second_order_patch(
                residue_type=residue_type,
                rows=typed,
                parent_version_id=parent_version_id,
            )
            if patch is not None:
                patches.append(patch)
        unique: Dict[str, InstrumentPatch] = {}
        for patch in patches:
            unique[patch.patch_id] = patch
        return list(unique.values())

    def _pruning_race_patch(
        self,
        rows: List[FailureResidue],
        policy: Dict[str, Any],
        config: Optional[Any],
        parent_version_id: str,
    ) -> InstrumentPatch:
        target_grace = max(
            [int(policy.get("generated_grace_generations", 0))]
            + [
                int(residue.missing_evidence.get("required_age_generations", 0))
                for residue in rows
            ]
        )
        target_evals = max(
            [int(policy.get("generated_min_evaluations", 0))]
            + [
                int(residue.missing_evidence.get("required_evaluations", 0))
                for residue in rows
            ]
        )
        target_probe = max(
            float(policy.get("generated_probe_rate", 0.0)),
            float(getattr(config, "probe_rate", 1.0)),
        )

        updates: Dict[str, Any] = {
            "generated_grace_generations": target_grace,
            "generated_min_evaluations": target_evals,
            "generated_probe_rate": round(min(1.0, max(0.0, target_probe)), 4),
        }
        if any(residue.evaluations == 0 for residue in rows):
            updates["clean_probe_first_eval"] = bool(
                getattr(config, "clean_probe_first_eval", True)
            )
            updates["generated_module_scaffold_strategy"] = (
                "clean_probe_until_min_evaluations"
            )
        updates["evaluator_terms"] = ["generated_evidence_window"]
        updates["archive_insertion_priority"] = (
            "prefer_under_evaluated_generated_modules"
        )

        residue_ids = [residue.residue_id for residue in rows]
        patch_id = self._patch_id("pruning_race_evidence_budget", updates, residue_ids)
        provenance = ProvenanceRecord(
            source="afirsi_core.OperatorGenerator",
            residue_ids=residue_ids,
            problem_space_version=parent_version_id,
            generator_version=self.version,
            details={
                "residue_types": sorted({residue.residue_type for residue in rows}),
                "subjects": [residue.subject_id for residue in rows],
                "missing_evidence": [residue.missing_evidence for residue in rows],
            },
        )
        return InstrumentPatch(
            patch_id=patch_id,
            candidate_name=f"pruning_race_evidence_budget_{patch_id.rsplit('-', 1)[-1]}",
            source_residue_ids=residue_ids,
            parent_version_id=parent_version_id,
            target_updates=updates,
            evaluator_terms=["generated_evidence_window"],
            archive_insertion_priority="prefer_under_evaluated_generated_modules",
            generated_module_scaffold_strategy=(
                "clean_probe_until_min_evaluations"
            ),
            rationale=[
                "A generated module reached a prune attempt before the active evidence window was satisfied.",
                "The patch increases real evaluation pressure without fabricating archive or fitness evidence.",
            ],
            provenance=provenance.to_dict(),
        )

    def _second_order_patch(
        self,
        residue_type: str,
        rows: List[FailureResidue],
        parent_version_id: str,
    ) -> Optional[InstrumentPatch]:
        recipes: Dict[str, Dict[str, Any]] = {
            "ARCHIVE_STAGNATION": {
                "updates": {
                    "archive_insertion_priority": "novelty_diversity_generated_modules",
                    "archive_priority_terms": ["novelty_cell_coverage"],
                    "meta_archive_gain": 3.0,
                    "meta_exploration_floor": 0.4,
                },
                "evaluator_terms": ["archive_stagnation", "novelty_cell_coverage"],
                "scaffold": "archive_diversity_probe",
            },
            "OPERATOR_GENERATOR_MODE_COLLAPSE": {
                "updates": {
                    "candidate_scoring_coefficients": "require_patch_novelty",
                    "meta_exploration_floor": 0.45,
                    "evaluator_terms": ["patch_equivalence_penalty"],
                },
                "evaluator_terms": ["patch_equivalence_penalty"],
                "scaffold": "diversified_patch_family_probe",
            },
            "EVALUATOR_NOISE_OR_OVERFIT": {
                "updates": {
                    "candidate_scoring_coefficients": "holdout_weighted_recursive_score",
                    "evaluator_terms": ["holdout_robustness"],
                },
                "evaluator_terms": ["holdout_robustness"],
                "scaffold": "paired_holdout_stability_probe",
            },
            "POLICY_SATURATION": {
                "updates": {
                    "candidate_scoring_coefficients": "new_observation_channel_search",
                    "evaluator_terms": ["non_pruning_bottleneck_search"],
                    "archive_priority_terms": ["underexplored_behavior_cells"],
                },
                "evaluator_terms": ["non_pruning_bottleneck_search"],
                "scaffold": "discover_next_bottleneck",
            },
            "SCAFFOLD_BIAS": {
                "updates": {
                    "clean_probe_first_eval": False,
                    "generated_module_scaffold_strategy": "mixed_genome_transfer_probe",
                    "candidate_scoring_coefficients": "mixed_genome_transfer_required",
                },
                "evaluator_terms": ["mixed_genome_transfer"],
                "scaffold": "mixed_genome_transfer_probe",
            },
            "META_OPERATOR_IMBALANCE": {
                "updates": {
                    "meta_operator_weighting_coefficients": "balanced_operator_family_floor",
                    "meta_exploration_floor": 0.5,
                    "evaluator_terms": ["operator_weight_entropy"],
                },
                "evaluator_terms": ["operator_weight_entropy"],
                "scaffold": "balanced_operator_family_probe",
            },
            "PATCH_EFFECTIVENESS_FAILURE": {
                "updates": {
                    "candidate_scoring_coefficients": "avoid_failed_causal_signature",
                    "evaluator_terms": ["patch_effectiveness_delta"],
                },
                "evaluator_terms": ["patch_effectiveness_delta"],
                "scaffold": "alternate_failed_signature_probe",
            },
        }
        recipe = recipes.get(residue_type)
        if recipe is None:
            return None
        residue_ids = [residue.residue_id for residue in rows]
        patch_id = self._patch_id(
            residue_type.lower(),
            recipe["updates"],
            residue_ids,
        )
        provenance = ProvenanceRecord(
            source="afirsi_core.OperatorGenerator",
            residue_ids=residue_ids,
            problem_space_version=parent_version_id,
            generator_version=self.version,
            details={
                "residue_types": [residue_type],
                "subjects": [residue.subject_id for residue in rows],
            },
        )
        return InstrumentPatch(
            patch_id=patch_id,
            candidate_name=f"{residue_type.lower()}_{patch_id.rsplit('-', 1)[-1]}",
            source_residue_ids=residue_ids,
            parent_version_id=parent_version_id,
            target_updates=recipe["updates"],
            evaluator_terms=recipe["evaluator_terms"],
            archive_insertion_priority=recipe["updates"].get(
                "archive_insertion_priority",
                "second_order_bottleneck",
            ),
            generated_module_scaffold_strategy=recipe["scaffold"],
            rationale=[
                f"{residue_type} was derived from observed execution or validation evidence.",
                "The patch changes instrument policy only and still requires validation.",
            ],
            provenance=provenance.to_dict(),
        )

    def _patch_id(
        self,
        interpretation: str,
        updates: Dict[str, Any],
        residue_ids: List[str],
    ) -> str:
        payload = {
            "interpretation": interpretation,
            "updates": updates,
            "residue_ids": residue_ids,
            "version": self.version,
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.blake2b(raw, digest_size=6).hexdigest()
        return f"afirsi-patch-{interpretation}-{digest}"
