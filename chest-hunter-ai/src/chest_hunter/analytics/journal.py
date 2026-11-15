"""Append-only journal of finished benchmark rows (one JSON line per episode job).

Long benchmarks write every row as soon as its episode finishes, so an interrupted run keeps
what it measured and ``--resume`` plays only the missing jobs. The first line records the
run's parameters; a resume with different parameters is refused (rows of different runs
are never mixed).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from chest_hunter.exceptions import ValidationError

FORMAT = "chest-hunter-row-journal/1"


class RowJournal:
    def __init__(self, path: Path, params: dict[str, Any], *, resume: bool) -> None:
        self.path = path
        self.params = json.loads(json.dumps(params, sort_keys=True, default=str))
        self.rows: dict[str, dict[str, Any]] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.exists():
            with path.open(encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            header = json.loads(lines[0]) if lines else {}
            if header.get("format") != FORMAT or header.get("params") != self.params:
                raise ValidationError(f"{path} belongs to a run with other parameters; "
                                      "resume with the same options or start a new run")
            for line in lines[1:]:
                try:
                    entry = json.loads(line)
                except ValueError:                   # a line cut short by an interruption
                    continue
                self.rows[entry["key"]] = entry["row"]
        else:
            path.write_text(json.dumps({"format": FORMAT, "params": self.params}) + "\n",
                            encoding="utf-8")

    @staticmethod
    def key(*parts: Any) -> str:
        return json.dumps([str(p) for p in parts])

    def done(self, key: str) -> bool:
        return key in self.rows

    def add(self, key: str, row: dict[str, Any]) -> None:
        self.rows[key] = row
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "row": row}, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
