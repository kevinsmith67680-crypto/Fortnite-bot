"""Recording of control demonstrations in the research environment.

A demonstration episode is a directory::

    episode_00042/
      episode.json        metadata: ids, map, seed, source, settings, counts, outcome, version
      steps.npz           one row per simulator step (compact NumPy arrays, see STEP_FIELDS)
      frames/000017.png   downscaled camera observations (lossless PNG, de-duplicated)

Every simulator step is stored (so an episode can be replayed exactly from its reset);
steps that were taken right after a fresh camera observation also reference that frame and
form the (observation, action) training samples. The observation is the one the controller
had *before* choosing the action: ``frame_time == action_time`` is checked on recording and
again by the dataset loader.

Only what the research environment produces is stored: camera frames from the
environment's own render, the controller's commands and the environment's telemetry. No
desktop capture, audio, user names or host information.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from chest_hunter.unreal_bridge.observation_adapter import Observation3D, Telemetry

RECORDER_VERSION: Final = 1
#: Order of the action vector stored per step (continuous first, then binary).
ACTION_FIELDS: Final = ("move_forward", "move_right", "look_yaw", "look_pitch", "jump", "interact")
CONTINUOUS_ACTIONS: Final = ACTION_FIELDS[:4]
BINARY_ACTIONS: Final = ACTION_FIELDS[4:]
#: Coarse controller phase labels (expert demonstrations only; human episodes use -1).
PHASES: Final = ("scan", "navigate", "approach", "align", "interact", "verify", "recover")
OUTCOMES: Final = ("none", "chest_opened", "interaction_failed")
SOURCE_HUMAN: Final = "human"
SOURCE_SCRIPTED: Final = "synthetic-scripted-expert"
#: Default stored frame size (width, height): enough for the compact policy, small on disk.
FRAME_SIZE: Final = (128, 72)

_STATE_PHASE: Final = {
    "OBSERVING": "scan", "SCANNING": "scan", "EXPLORING": "navigate",
    "TARGET_DETECTED": "navigate", "PLANNING_ROUTE": "navigate", "NAVIGATING": "navigate",
    "APPROACHING": "approach", "ALIGNING": "align", "INTERACTING": "interact",
    "VERIFYING_OPEN": "verify", "RECOVERING": "recover",
}


class RecordingError(RuntimeError):
    """Misuse of the recorder (wrong state, misaligned observation, bad action)."""


def phase_of(state: object) -> int:
    """Phase index for a controller state (``-1`` when unknown, e.g. human control)."""
    name = _STATE_PHASE.get(str(state))
    return PHASES.index(name) if name is not None else -1


def action_vector(action: dict[str, Any]) -> np.ndarray:
    """Validated action dict -> vector in ACTION_FIELDS order (float64: replay is exact)."""
    vec = np.zeros(len(ACTION_FIELDS), dtype=np.float64)
    for i, key in enumerate(ACTION_FIELDS):
        value = action.get(key, False if key in BINARY_ACTIONS else 0.0)
        if key in BINARY_ACTIONS:
            if not isinstance(value, bool):
                raise RecordingError(f"{key} must be a boolean")
            vec[i] = 1.0 if value else 0.0
        else:
            v = float(value)
            if not math.isfinite(v) or not -1.0 <= v <= 1.0:
                raise RecordingError(f"{key} must be within [-1, 1], got {value!r}")
            vec[i] = v
    return vec


def downscale(frame: np.ndarray, size: tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """Area-averaged downscale of a BGR frame to ``size`` = (width, height)."""
    if frame.shape[1] == size[0] and frame.shape[0] == size[1]:
        return np.ascontiguousarray(frame)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def proprioception(telemetry: Telemetry | None) -> np.ndarray:
    """The agent's own body state used as policy input: camera pitch and speed (scaled)."""
    if telemetry is None:
        return np.zeros(2, dtype=np.float32)
    v = telemetry.odometry.velocity
    speed = math.hypot(v[0], v[1])
    return np.array([telemetry.camera.pitch_deg / 60.0, speed / 7.0], dtype=np.float32)


@dataclass(slots=True)
class _Step:
    tick_start: int
    ticks: int
    action_time: float
    action: np.ndarray
    observed: bool
    frame_ref: int
    frame_id: int
    frame_time: float
    proprio: np.ndarray
    odometry_after: tuple[float, float, float]
    phase: int
    outcome: int
    reward: bool


class DemonstrationRecorder:
    """Records one episode at a time: ``start`` -> ``record_step`` ... -> ``stop``.

    ``pause``/``resume`` suspend recording. Steps reported while paused are not stored and
    mark the episode as not exactly replayable (the teleoperation window instead stops
    stepping the environment while paused, which keeps episodes replayable).
    """

    def __init__(self, root: str | Path, *, frame_size: tuple[int, int] = FRAME_SIZE,
                 png_compression: int = 3, max_steps: int = 200_000) -> None:
        self.root = Path(root)
        self.frame_size = frame_size
        self.png_compression = int(png_compression)
        self.max_steps = max_steps
        self.state = "idle"
        self.episode_dir: Path | None = None
        self._meta: dict[str, Any] = {}
        self._steps: list[_Step] = []
        self._hashes: dict[bytes, int] = {}
        self._frames_written = 0
        self._skipped_while_paused = 0
        self._last_tick_end: int | None = None
        self._last_telemetry: Telemetry | None = None
        self.bytes_written = 0

    # -- controls -----------------------------------------------------------------------------

    def start(self, *, episode_id: int, map_id: str, seed: int, episode: int, source: str,
              settings: dict[str, Any] | None = None, telemetry: Telemetry | None = None,
              initial_observation: bool = False) -> Path:
        """Begin an episode. ``initial_observation``: the controller requested a frame with
        GET_OBSERVATION right after the reset (replay must do the same)."""
        if self.state in ("recording", "paused"):
            raise RecordingError("a recording is already in progress; stop it first")
        if source not in (SOURCE_HUMAN, SOURCE_SCRIPTED):
            raise RecordingError(f"source must be {SOURCE_HUMAN!r} or {SOURCE_SCRIPTED!r}")
        directory = self.root / f"episode_{episode_id:05d}"
        if directory.exists():
            raise RecordingError(f"{directory} already exists; demonstrations are never "
                                 "overwritten")
        (directory / "frames").mkdir(parents=True)
        self.episode_dir = directory
        self._meta = {
            "format": "chest-hunter-demonstration/1", "recorder_version": RECORDER_VERSION,
            "episode_id": episode_id, "map_id": map_id, "seed": seed, "episode": episode,
            "source": source, "synthetic": source != SOURCE_HUMAN,
            "settings": dict(settings or {}), "frame_size": list(self.frame_size),
            "action_fields": list(ACTION_FIELDS), "phases": list(PHASES),
            "outcomes": list(OUTCOMES), "initial_observation": initial_observation,
        }
        self._steps = []
        self._hashes = {}
        self._frames_written = 0
        self._skipped_while_paused = 0
        self._last_tick_end = telemetry.tick if telemetry is not None else None
        self._last_telemetry = telemetry
        self.state = "recording"
        return directory

    def pause(self) -> None:
        if self.state != "recording":
            raise RecordingError(f"cannot pause while {self.state}")
        self.state = "paused"

    def resume(self) -> None:
        if self.state != "paused":
            raise RecordingError(f"cannot resume while {self.state}")
        self.state = "recording"

    @property
    def active(self) -> bool:
        return self.state in ("recording", "paused")

    # -- data ---------------------------------------------------------------------------------

    def record_step(self, *, action: dict[str, Any], ticks: int, action_time: float,
                    telemetry_after: Telemetry, observation: Observation3D | None = None,
                    observed: bool = False, phase: int = -1) -> None:
        """Store one simulator step.

        ``observation`` is the camera frame the controller had just received when it chose
        ``action`` (``None`` if it acted on older information); ``observed`` says whether
        this step asked the environment for a new frame (needed for exact replay).
        """
        if self.state == "paused":
            self._skipped_while_paused += 1
            self._last_telemetry = telemetry_after
            self._last_tick_end = telemetry_after.tick
            return
        if self.state != "recording":
            raise RecordingError(f"cannot record while {self.state}")
        if len(self._steps) >= self.max_steps:
            raise RecordingError(f"episode exceeds max_steps={self.max_steps}")
        vec = action_vector(action)
        if not 1 <= ticks <= 30:
            raise RecordingError("ticks must be within 1-30")
        frame_ref, frame_id, frame_time = -1, -1, math.nan
        if observation is not None:
            if abs(observation.sim_time - action_time) > 1e-6:
                raise RecordingError(
                    f"observation at t={observation.sim_time:.4f} s does not precede the action "
                    f"at t={action_time:.4f} s (misaligned sample)")
            frame_ref = self._store_frame(observation.frame)
            frame_id, frame_time = observation.frame_id, observation.sim_time
        tick_start = telemetry_after.tick - ticks if self._last_tick_end is None \
            else self._last_tick_end
        outcome, reward = 0, False
        for event in telemetry_after.events:
            if event.get("type") == "chest_opened_notification":
                outcome, reward = 1, bool(event.get("reward"))
            elif event.get("type") == "interaction_failed" and outcome == 0:
                outcome = 2
        self._steps.append(_Step(tick_start, ticks, action_time, vec, observed, frame_ref,
                                 frame_id, frame_time, proprioception(self._last_telemetry),
                                 telemetry_after.odometry.position, phase, outcome, reward))
        self._last_tick_end = telemetry_after.tick
        self._last_telemetry = telemetry_after

    def _store_frame(self, frame: np.ndarray) -> int:
        assert self.episode_dir is not None
        small = downscale(frame, self.frame_size)
        digest = hashlib.blake2b(small.tobytes(), digest_size=16).digest()
        if digest in self._hashes:                       # identical frame: reference it
            return self._hashes[digest]
        index = self._frames_written
        path = self.episode_dir / "frames" / f"{index:06d}.png"
        ok, data = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression])
        if not ok:
            raise RecordingError("PNG encoding failed")
        path.write_bytes(data.tobytes())
        self.bytes_written += len(data)
        self._hashes[digest] = index
        self._frames_written += 1
        return index

    def stop(self, *, outcome: dict[str, Any] | None = None) -> Path:
        """Write ``steps.npz`` and ``episode.json``; returns the episode directory."""
        if not self.active or self.episode_dir is None:
            raise RecordingError(f"cannot stop while {self.state}")
        steps = self._steps
        n = len(steps)
        arrays: dict[str, Any] = {
            "tick_start": np.array([s.tick_start for s in steps], dtype=np.int64),
            "ticks": np.array([s.ticks for s in steps], dtype=np.int16),
            "action_time": np.array([s.action_time for s in steps], dtype=np.float64),
            "action": np.stack([s.action for s in steps]) if n
            else np.zeros((0, len(ACTION_FIELDS)), np.float64),
            "observed": np.array([s.observed for s in steps], dtype=bool),
            "frame_ref": np.array([s.frame_ref for s in steps], dtype=np.int32),
            "frame_id": np.array([s.frame_id for s in steps], dtype=np.int64),
            "frame_time": np.array([s.frame_time for s in steps], dtype=np.float64),
            "proprio": np.stack([s.proprio for s in steps]) if n else np.zeros((0, 2), np.float32),
            "odometry_after": np.array([s.odometry_after for s in steps], dtype=np.float64)
            .reshape(n, 3),
            "phase": np.array([s.phase for s in steps], dtype=np.int8),
            "outcome": np.array([s.outcome for s in steps], dtype=np.int8),
            "reward": np.array([s.reward for s in steps], dtype=bool),
        }
        np.savez_compressed(self.episode_dir / "steps.npz", **arrays)
        samples = int((arrays["frame_ref"] >= 0).sum())
        self._meta.update({
            "steps": n, "samples": samples, "frames": self._frames_written,
            "duplicate_frames_skipped": samples - self._frames_written,
            "steps_skipped_while_paused": self._skipped_while_paused,
            "replayable": self._skipped_while_paused == 0,
            "sim_seconds": round(float(arrays["action_time"][-1]) if n else 0.0, 4),
            "chests_opened": int((arrays["outcome"] == 1).sum()),
            "outcome": dict(outcome or {}),
        })
        (self.episode_dir / "episode.json").write_text(json.dumps(self._meta, indent=1) + "\n",
                                                       encoding="utf-8")
        directory = self.episode_dir
        self.state = "stopped"
        self._steps = []
        self._hashes = {}
        return directory


def next_episode_id(root: str | Path) -> int:
    """Smallest unused episode id in a dataset directory (ids are never reused)."""
    root = Path(root)
    ids = [int(p.name.split("_")[1]) for p in root.glob("episode_*") if p.name[8:].isdigit()]
    return max(ids, default=-1) + 1


def load_episode(directory: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    directory = Path(directory)
    meta = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    with np.load(directory / "steps.npz") as data:
        arrays = {k: data[k] for k in data.files}
    return meta, arrays


def iter_samples(directory: str | Path) -> list[dict[str, Any]]:
    """Human-readable view of an episode's training samples (one dict per sample)."""
    directory = Path(directory)
    meta, a = load_episode(directory)
    out = []
    for i in np.nonzero(a["frame_ref"] >= 0)[0]:
        sample: dict[str, Any] = {"episode_id": meta["episode_id"], "map_id": meta["map_id"],
                                  "frame_id": int(a["frame_id"][i]),
                                  "sim_time": float(a["action_time"][i]),
                                  "observation_path": f"frames/{int(a['frame_ref'][i]):06d}.png"}
        for j, key in enumerate(ACTION_FIELDS):
            value = float(a["action"][i, j])
            sample[key] = bool(value > 0.5) if key in BINARY_ACTIONS else value
        out.append(sample)
    return out


@dataclass(frozen=True, slots=True)
class ReplayReport:
    steps: int
    frames_compared: int
    frame_mismatches: int
    max_position_error: float

    @property
    def exact(self) -> bool:
        return self.frame_mismatches == 0 and self.max_position_error < 1e-6


def replay_episode(directory: str | Path, client: Any) -> ReplayReport:
    """Re-run a recorded episode through a bridge client and compare it with the recording.

    The client must be connected to a server that can reset the same map; the episode is
    reset with the recorded seed and settings, every recorded action is applied with its
    recorded tick count, and frames are re-requested on the same steps.
    """
    meta, a = load_episode(directory)
    if not meta.get("replayable", False):
        raise RecordingError("this episode was recorded with pauses and is not replayable")
    s = meta.get("settings", {})
    client.reset_episode(meta["map_id"], seed=meta["seed"], episode=meta["episode"],
                         width=int(s.get("width", 640)), height=int(s.get("height", 360)),
                         fov_deg=float(s.get("fov_deg", 90.0)),
                         graphics_quality=str(s.get("graphics_quality", "low")),
                         max_ticks=s.get("max_ticks"), navigation=str(s.get("navigation",
                                                                             "navmesh")))
    size = (int(meta["frame_size"][0]), int(meta["frame_size"][1]))
    frames_dir = Path(directory) / "frames"
    compared = mismatches = 0
    max_err = 0.0
    pending: Observation3D | None = None
    if meta.get("initial_observation"):
        pending = client.get_observation()
    for i in range(len(a["ticks"])):
        if a["frame_ref"][i] >= 0:
            compared += 1
            stored = cv2.imread(str(frames_dir / f"{int(a['frame_ref'][i]):06d}.png"))
            if pending is None or stored is None or not np.array_equal(
                    downscale(pending.frame, size), stored):
                mismatches += 1
        vec = a["action"][i]
        action = {k: float(vec[j]) for j, k in enumerate(CONTINUOUS_ACTIONS)}
        action |= {k: bool(vec[4 + j] > 0.5) for j, k in enumerate(BINARY_ACTIONS)}
        result = client.step(action, ticks=int(a["ticks"][i]), observe=bool(a["observed"][i]))
        pending = result.observation
        err = math.dist(result.telemetry.odometry.position, a["odometry_after"][i].tolist())
        max_err = max(max_err, err)
    return ReplayReport(len(a["ticks"]), compared, mismatches, max_err)
