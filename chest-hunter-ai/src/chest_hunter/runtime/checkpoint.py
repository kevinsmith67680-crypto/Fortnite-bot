"""Lightweight runtime checkpoints for long experiments (Phase 6).

A checkpoint is one JSON document::

    {"format": "chest-hunter-checkpoint", "version": 1,
     "kind": "endurance", "created": ..., "updated": ...,
     "config": {...}, "config_hash": "...", "agent_version": "...", "policy_version": "...",
     "seed": 42, "seed_sequence": "...",
     "last_completed_episode": 57, "completed": 58,
     "results_file": "episodes.jsonl", "aggregate": {...},
     "checksum": "<sha256 of the document without this field>"}

* **Atomic writes:** the document goes to a temporary file in the same directory, is
  flushed and ``fsync``-ed, and replaces the checkpoint with ``os.replace``; the previous
  valid checkpoint is kept as ``<name>.bak``. A crash leaves either the old or the new
  checkpoint, never a half-written one.
* **Validation:** a checkpoint whose checksum, format or version does not match is
  corrupted. Loading falls back to the backup; if both are unusable,
  :class:`CheckpointError` is raised (never a silent restart that would discard results).
* **Compatibility:** resuming requires the same ``config_hash`` and ``agent_version``;
  anything else is refused with the differing fields named.
* Per-episode results live in an append-only JSON-lines file (one ``fsync``-ed line per
  episode). :func:`read_results` keeps every complete record, drops a torn last line
  (reported), and removes duplicate episode ids (first record wins).

Nothing here restores a partially executed simulation: an interrupted episode is run again
from its start (the research environment has no deterministic mid-episode restore).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

FORMAT: Final = "chest-hunter-checkpoint"
VERSION: Final = 1


class CheckpointError(Exception):
    """A checkpoint (and its backup) cannot be used, or does not match this run."""


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(config).encode("utf-8")).hexdigest()[:16]


def _checksum(doc: dict[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k != "checksum"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    with contextlib.suppress(OSError):       # make the rename durable where supported
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


@dataclass
class Checkpoint:
    kind: str
    config: dict[str, Any]
    agent_version: str
    policy_version: str
    seed: int
    seed_sequence: str
    last_completed_episode: int = -1
    completed: int = 0
    results_file: str = "episodes.jsonl"
    aggregate: dict[str, Any] | None = None
    created: str = ""
    updated: str = ""

    @property
    def config_hash(self) -> str:
        return config_hash(self.config)

    def to_document(self) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        doc = {"format": FORMAT, "version": VERSION, "kind": self.kind,
               "created": self.created or now, "updated": now, "config": self.config,
               "config_hash": self.config_hash, "agent_version": self.agent_version,
               "policy_version": self.policy_version, "seed": self.seed,
               "seed_sequence": self.seed_sequence,
               "last_completed_episode": self.last_completed_episode,
               "completed": self.completed, "results_file": self.results_file,
               "aggregate": self.aggregate or {}}
        doc["checksum"] = _checksum(doc)
        return doc

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> Checkpoint:
        if doc.get("format") != FORMAT or doc.get("version") != VERSION:
            raise CheckpointError(f"unsupported checkpoint format/version "
                                  f"({doc.get('format')!r}, {doc.get('version')!r})")
        if doc.get("checksum") != _checksum(doc):
            raise CheckpointError("checksum mismatch (the file is corrupted or was edited)")
        if config_hash(doc["config"]) != doc.get("config_hash"):
            raise CheckpointError("config hash does not match the stored configuration")
        return cls(kind=doc["kind"], config=doc["config"], agent_version=doc["agent_version"],
                   policy_version=doc["policy_version"], seed=int(doc["seed"]),
                   seed_sequence=doc["seed_sequence"],
                   last_completed_episode=int(doc["last_completed_episode"]),
                   completed=int(doc["completed"]), results_file=doc["results_file"],
                   aggregate=dict(doc.get("aggregate") or {}), created=doc["created"],
                   updated=doc["updated"])


def save_checkpoint(path: str | Path, checkpoint: Checkpoint) -> None:
    target = Path(path)
    doc = checkpoint.to_document()
    checkpoint.created, checkpoint.updated = doc["created"], doc["updated"]
    if target.exists():
        try:                                   # keep the previous *valid* checkpoint
            Checkpoint.from_document(json.loads(target.read_text(encoding="utf-8")))
            shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
        except (OSError, ValueError, KeyError, CheckpointError):
            pass
    atomic_write_text(target, json.dumps(doc, indent=1, sort_keys=True) + "\n")


def load_checkpoint(path: str | Path) -> tuple[Checkpoint | None, list[str]]:
    """(checkpoint, warnings). ``None`` when no checkpoint exists at all. Falls back to the
    backup when the main file is corrupted; raises :class:`CheckpointError` when a
    checkpoint exists but neither it nor its backup can be used."""
    target = Path(path)
    backup = target.with_suffix(target.suffix + ".bak")
    if not target.exists() and not backup.exists():
        return None, []
    warnings: list[str] = []
    errors: list[str] = []
    for candidate in (target, backup):
        if not candidate.exists():
            continue
        try:
            cp = Checkpoint.from_document(json.loads(candidate.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError, CheckpointError) as exc:
            errors.append(f"{candidate.name}: {exc}")
            warnings.append(f"checkpoint {candidate.name} is unusable: {exc}")
            continue
        if candidate is backup:
            warnings.append("using the backup checkpoint (the main one is corrupted)")
        return cp, warnings
    raise CheckpointError("no usable checkpoint: " + "; ".join(errors))


def check_compatible(cp: Checkpoint, *, config: dict[str, Any], agent_version: str) -> None:
    problems = []
    if cp.agent_version != agent_version:
        problems.append(f"agent version {cp.agent_version} != {agent_version}")
    if cp.config_hash != config_hash(config):
        differing = sorted(k for k in set(cp.config) | set(config)
                           if cp.config.get(k) != config.get(k))
        problems.append("configuration differs in: " + ", ".join(differing))
    if problems:
        raise CheckpointError("cannot resume: " + "; ".join(problems))


# -- per-episode results (append-only JSON lines) --------------------------------------------

def append_result(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, default=str) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_results(path: str | Path, *, key: str = "episode"
                 ) -> tuple[list[dict[str, Any]], list[str]]:
    """(records, warnings): complete records in file order, without duplicates."""
    target = Path(path)
    if not target.exists():
        return [], []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    seen: set[Any] = set()
    raw = target.read_text(encoding="utf-8")
    lines = raw.split("\n")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        last = number == len(lines)              # no newline after it: possibly torn
        try:
            record = json.loads(line)
        except ValueError:
            warnings.append(f"{target.name} line {number}: "
                            + ("incomplete last record dropped" if last else
                               "unreadable record skipped"))
            continue
        if record.get(key) in seen:
            warnings.append(f"{target.name} line {number}: duplicate {key} "
                            f"{record.get(key)} ignored")
            continue
        seen.add(record.get(key))
        records.append(record)
    return records, warnings


def rewrite_results(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Rewrite the results file atomically (used to drop a torn last line on resume)."""
    text = "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in records)
    atomic_write_text(Path(path), text)
