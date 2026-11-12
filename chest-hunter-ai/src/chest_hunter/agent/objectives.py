"""Per-session bookkeeping of which chest locations are still worth visiting.

The candidate set is loaded from SQLite once per session
(``ChestRepository.get_unvisited_chests``); afterwards this in-memory tracker
mirrors exactly what the agent has recorded, so the planner never queries the
database on a simulation tick.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from chest_hunter.navigation.route_planner import Candidate


class TargetStatus(StrEnum):
    PENDING = "pending"
    VISITED = "visited"          # presence verified and an observation recorded
    UNREACHABLE = "unreachable"  # no path exists (not observed, nothing recorded)
    ABANDONED = "abandoned"      # recovery attempts exhausted


class ObjectiveTracker:
    """Candidate targets and their status within one session."""

    def __init__(self, candidates: Iterable[Candidate]) -> None:
        self._candidates: dict[int, Candidate] = {}
        for candidate in sorted(candidates, key=lambda c: c.chest_id):
            self._candidates[candidate.chest_id] = candidate
        self._status = {cid: TargetStatus.PENDING for cid in self._candidates}
        self._attempts: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._candidates)

    def candidate(self, chest_id: int) -> Candidate:
        return self._candidates[chest_id]

    def status(self, chest_id: int) -> TargetStatus:
        return self._status[chest_id]

    def pending(self) -> list[Candidate]:
        return [c for cid, c in self._candidates.items()
                if self._status[cid] is TargetStatus.PENDING]

    def count(self, status: TargetStatus) -> int:
        return sum(1 for s in self._status.values() if s is status)

    @property
    def done(self) -> bool:
        return not any(s is TargetStatus.PENDING for s in self._status.values())

    def _resolve(self, chest_id: int, status: TargetStatus) -> None:
        if self._status[chest_id] is not TargetStatus.PENDING:
            raise ValueError(
                f"Chest location {chest_id} is already {self._status[chest_id].value}."
            )
        self._status[chest_id] = status

    def mark_visited(self, chest_id: int) -> None:
        self._resolve(chest_id, TargetStatus.VISITED)

    def mark_unreachable(self, chest_id: int) -> None:
        self._resolve(chest_id, TargetStatus.UNREACHABLE)

    def mark_abandoned(self, chest_id: int) -> None:
        self._resolve(chest_id, TargetStatus.ABANDONED)

    def record_attempt(self, chest_id: int) -> int:
        """Count one recovery attempt for a target and return the new total."""
        self._attempts[chest_id] = self._attempts.get(chest_id, 0) + 1
        return self._attempts[chest_id]

    def attempts(self, chest_id: int) -> int:
        return self._attempts.get(chest_id, 0)
