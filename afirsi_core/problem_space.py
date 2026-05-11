"""Version graph for evolving AFIRSI problem/instrument spaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProblemSpaceVersion:
    version_id: str
    parent_version_id: Optional[str]
    patch_id: Optional[str]
    residue_ids: List[str]
    active_observation_channel_version: str
    active_evaluator_version: str
    active_instrument_policy: Dict[str, Any]
    sequence_number: int


class ProblemSpaceVersionGraph:
    """Records parent-child lineage as AFIRSI patches change instruments."""

    def __init__(
        self,
        root_policy: Optional[Dict[str, Any]] = None,
        observation_channel_version: str = "observation.generated_module_lifecycle.v1",
        evaluator_version: str = "evaluator.pruning_propagation_race.v1",
    ) -> None:
        root = ProblemSpaceVersion(
            version_id="psv-0",
            parent_version_id=None,
            patch_id=None,
            residue_ids=[],
            active_observation_channel_version=observation_channel_version,
            active_evaluator_version=evaluator_version,
            active_instrument_policy=dict(root_policy or {}),
            sequence_number=0,
        )
        self._versions: Dict[str, ProblemSpaceVersion] = {"psv-0": root}
        self._children: Dict[str, List[str]] = {"psv-0": []}
        self.active_version_id = "psv-0"

    def current_version(self) -> ProblemSpaceVersion:
        return self._versions[self.active_version_id]

    def get(self, version_id: str) -> ProblemSpaceVersion:
        try:
            return self._versions[version_id]
        except KeyError as exc:
            raise KeyError(f"Unknown problem-space version: {version_id}") from exc

    def create_child(
        self,
        parent_version_id: str,
        patch_id: str,
        residue_ids: List[str],
        observation_channel_version: str,
        evaluator_version: str,
        instrument_policy: Dict[str, Any],
    ) -> ProblemSpaceVersion:
        self.get(parent_version_id)
        version_id = f"psv-{len(self._versions)}"
        version = ProblemSpaceVersion(
            version_id=version_id,
            parent_version_id=parent_version_id,
            patch_id=patch_id,
            residue_ids=list(residue_ids),
            active_observation_channel_version=observation_channel_version,
            active_evaluator_version=evaluator_version,
            active_instrument_policy=dict(instrument_policy),
            sequence_number=len(self._versions),
        )
        self._versions[version_id] = version
        self._children.setdefault(parent_version_id, []).append(version_id)
        self._children.setdefault(version_id, [])
        self.active_version_id = version_id
        return version

    def lineage(self, version_id: Optional[str] = None) -> List[ProblemSpaceVersion]:
        cursor = version_id or self.active_version_id
        rows: List[ProblemSpaceVersion] = []
        while cursor is not None:
            version = self.get(cursor)
            rows.append(version)
            cursor = version.parent_version_id
        return list(reversed(rows))

    def children_of(self, version_id: str) -> List[ProblemSpaceVersion]:
        self.get(version_id)
        return [self._versions[child] for child in self._children[version_id]]
