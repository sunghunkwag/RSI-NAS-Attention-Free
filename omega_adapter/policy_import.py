"""Convert OMEGA instrument patches into executable EIE policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from rsi_nas import EIEConfig

from .schemas import InstrumentMutationContract, InstrumentPatch


DEFAULT_SCORING_COEFFICIENTS = {
    "mean_bpc_delta": 1.0,
    "generated_archive_insertions": 0.0010,
    "generated_evaluations": 0.0002,
    "meta_operator_policy_updates": 0.0002,
    "eie_wins": 0.0050,
}


@dataclass(frozen=True)
class OMEGAPolicyCandidate:
    name: str
    config: EIEConfig
    generated_min_evaluations: int
    patch: InstrumentPatch

    @property
    def scoring_coefficients(self) -> Dict[str, float]:
        coeffs = dict(DEFAULT_SCORING_COEFFICIENTS)
        coeffs.update(self.patch.candidate_scoring_coefficients)
        return coeffs

    def to_json_dict(self) -> Dict:
        return {
            "name": self.name,
            "config": asdict(self.config),
            "generated_min_evaluations": self.generated_min_evaluations,
            "patch": self.patch.to_json_dict(),
            "scoring_coefficients": self.scoring_coefficients,
        }


def _bounded_float(value, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def apply_instrument_patch(
    parent_name: str,
    parent_config: EIEConfig,
    parent_generated_min_evaluations: int,
    patch: InstrumentPatch,
    contract: InstrumentMutationContract | None = None,
) -> OMEGAPolicyCandidate:
    """Apply a validated patch to produce an executable EIE policy."""

    active_contract = contract or InstrumentMutationContract()
    active_contract.validate_patch(patch)

    cfg = asdict(parent_config)
    generated_min = int(parent_generated_min_evaluations)

    for field, value in patch.target_updates.items():
        if field == "generated_min_evaluations":
            generated_min = max(0, min(8, int(value)))
        elif field == "clean_probe_first_eval":
            cfg[field] = bool(value)
        elif field == "probe_rate":
            cfg[field] = _bounded_float(value, 0.0, 1.0)
        elif field == "meta_exploration_floor":
            cfg[field] = _bounded_float(value, 0.0, 2.0)
        elif field in {
            "grace_multiplier",
            "meta_eval_gain",
            "meta_archive_gain",
            "meta_fitness_gain",
        }:
            cfg[field] = _bounded_float(value, 0.0, 12.0)

    return OMEGAPolicyCandidate(
        name=patch.candidate_name,
        config=EIEConfig(**cfg),
        generated_min_evaluations=generated_min,
        patch=patch,
    )


def score_summary(summary: Dict, coefficients: Dict[str, float]) -> float:
    """Score a validated candidate without fabricating task performance."""

    if not summary["mechanism_valid"]:
        return -999.0
    return round(
        coefficients.get("mean_bpc_delta", 1.0) * summary["mean_bpc_delta"]
        + coefficients.get("generated_archive_insertions", 0.0)
        * summary["total_generated_archive_insertions"]
        + coefficients.get("generated_evaluations", 0.0)
        * summary["total_generated_evaluations"]
        + coefficients.get("meta_operator_policy_updates", 0.0)
        * summary["total_meta_operator_policy_updates"]
        + coefficients.get("eie_wins", 0.0) * summary["eie_wins"],
        6,
    )
