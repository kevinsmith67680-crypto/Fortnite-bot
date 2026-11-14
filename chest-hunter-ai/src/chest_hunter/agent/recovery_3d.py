"""Classification and bookkeeping of navigation/interaction failures (Phase 5).

Every recovery of the visual agent is mapped to one failure class. Each class has a
bounded strategy (``VisualAgent3D`` with ``Agent3DConfig.smart_recovery``) and a budget;
the per-target attempt limit and the episode-wide recovery budget of the Phase 4
controller still apply on top, so no strategy can loop forever. Outcomes are recorded:
a recovery counts as *resolved* when the agent afterwards reaches or opens the same target.

* path_blocked (no progress on a route or approach, approach timeout): back off along the
  trail, note the obstacle, replan.
* target_lost (the chest track disappeared): scan around the last reliable estimate, once.
* localisation_uncertainty (target estimate too uncertain near the goal): stop and collect
  more observations, at most twice per target.
* interaction_failure (interrupted hold, chest still closed, opening not confirmed): recheck
  distance and alignment; look back to confirm an opening the agent was notified about.
* camera_alignment_failure (the interaction prompt never appeared): step closer, then
  approach from another side.
* target_unreachable (no path): mark unreachable at once, no retries.
* model_inference_failure (learned policy error, timeout or invalid output): deterministic
  control (``PolicySelector``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Final

FAILURE_CLASSES: Final = ("path_blocked", "target_lost", "localisation_uncertainty",
                          "interaction_failure", "camera_alignment_failure",
                          "target_unreachable", "model_inference_failure")

REASON_CLASS: Final = {
    "stuck": "path_blocked", "approach_blocked": "path_blocked",
    "approach_timeout": "path_blocked", "target_lost": "target_lost",
    "localisation_uncertainty": "localisation_uncertainty",
    "interaction_failed": "interaction_failure", "interaction_interrupted": "interaction_failure",
    "verification_inconclusive": "interaction_failure",
    "alignment_failed": "camera_alignment_failure", "unreachable": "target_unreachable",
    "model_failure": "model_inference_failure",
}

#: Episode-wide budget per class for the smart strategies (after that the Phase 4
#: behaviour - retry, then abandon the target - takes over).
CLASS_BUDGET: Final = {"path_blocked": 40, "target_lost": 15, "localisation_uncertainty": 15,
                       "interaction_failure": 20, "camera_alignment_failure": 20,
                       "target_unreachable": 1000, "model_inference_failure": 1000}


def classify(reason: str | None) -> str:
    return REASON_CLASS.get(reason or "", "path_blocked")


@dataclass(slots=True)
class RecoveryLog:
    attempts: Counter[str] = field(default_factory=Counter)
    resolved: Counter[str] = field(default_factory=Counter)
    smart_used: Counter[str] = field(default_factory=Counter)
    pending: dict[object, list[str]] = field(default_factory=dict)

    def record(self, target_key: object, cls: str) -> None:
        self.attempts[cls] += 1
        self.pending.setdefault(target_key, []).append(cls)

    def resolve(self, target_key: object) -> None:
        for cls in self.pending.pop(target_key, []):
            self.resolved[cls] += 1

    def can_use(self, cls: str) -> bool:
        return self.smart_used[cls] < CLASS_BUDGET[cls]

    def use(self, cls: str) -> None:
        self.smart_used[cls] += 1

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {cls: {"attempts": self.attempts[cls], "resolved": self.resolved[cls],
                      "smart_strategy_used": self.smart_used[cls]}
                for cls in FAILURE_CLASSES if self.attempts[cls] or self.smart_used[cls]}
