"""Deployment startup checks (Phase 7).

``check-runtime`` runs every check and reports each with an actionable remedy; nothing is
modified (the database is opened read-only, a temporary file is written and removed to test
output directories). Checks:

* Python version and the pinned dependency versions (``requirements-lock.txt``);
* configuration file version;
* SQLite schema version;
* learned-model compatibility (format, input size, loadable backend);
* camera observation dimensions;
* exploration-memory version;
* research-bridge protocol version and the environment version the server reports
  (the reference arena is started for the check; ``--backend unreal`` connects to the
  configured loopback address with a short timeout);
* required output directories.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sqlite3
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MIN_PYTHON = (3, 11)
TESTED_PYTHON = ("3.11", "3.12", "3.13")
POLICY_FORMAT = "chest-hunter-policy/1"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    remedy: str = ""
    warning: bool = False          # ok, but worth attention

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _locked_versions(path: Path) -> dict[str, str]:
    """``name -> version`` from requirements-lock.txt lines applying to this interpreter."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    major_minor = sys.version_info[:2]
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        spec, _, marker = line.partition(";")
        name, version = (s.strip() for s in spec.split("==", 1))
        marker = marker.strip()
        if marker:
            # only the python_version markers this file uses: < "X.Y" / >= "X.Y"
            op, _, ver = marker.replace("python_version", "").strip().partition(" ")
            want = tuple(int(x) for x in ver.strip().strip('"').split("."))
            if (op == "<" and not major_minor < want) or (op == ">=" and not major_minor >= want):
                continue
        out[name.lower()] = version
    return out


def check_python(base_dir: Path) -> list[Check]:
    checks = []
    ver = ".".join(map(str, sys.version_info[:3]))
    ok = sys.version_info[:2] >= MIN_PYTHON
    tested = ".".join(map(str, sys.version_info[:2])) in TESTED_PYTHON
    checks.append(Check("python", ok, f"Python {ver} ({platform.machine()}, "
                        f"{platform.system()})" + ("" if tested else "; not a tested version"),
                        f"install Python {'/'.join(TESTED_PYTHON)}" if not ok or not tested
                        else "", warning=ok and not tested))
    locked = _locked_versions(base_dir / "requirements-lock.txt")
    for dist, required in (("numpy", True), ("opencv-python-headless", True),
                           ("onnxruntime", False)):
        try:
            have = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            checks.append(Check(f"dependency:{dist}", not required,
                                "not installed" + ("" if required else
                                                   " (the NumPy policy backend is used)"),
                                "python -m pip install -r requirements-lock.txt",
                                warning=not required))
            continue
        want = locked.get(dist)
        same = want is None or want == have
        checks.append(Check(f"dependency:{dist}", True,
                            f"{have}" + ("" if same else f" (pinned {want})"),
                            "" if same else "python -m pip install -r requirements-lock.txt",
                            warning=not same))
    return checks


def check_configuration(settings: Any) -> Check:
    from chest_hunter.config import CONFIG_VERSION

    v = settings.config_version
    if v == CONFIG_VERSION:
        return Check("configuration", True, f"{settings.config_path} (version {v})")
    return Check("configuration", True,
                 f"{settings.config_path} has config_version {v}; this program writes "
                 f"{CONFIG_VERSION} (older files are read with defaults for new settings)",
                 f'add "config_version": {CONFIG_VERSION} after reviewing the new sections',
                 warning=True)


def check_database(settings: Any) -> Check:
    from chest_hunter.database import SCHEMA_VERSION

    path = Path(settings.database.path)
    if not path.exists():
        return Check("database", True, f"{path} does not exist yet",
                     "run 'python -m chest_hunter init-db' before simulate/benchmark commands "
                     "that record sessions", warning=True)
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            version = int(con.execute("PRAGMA user_version").fetchone()[0])
        finally:
            con.close()
    except sqlite3.Error as exc:
        return Check("database", False, f"{path}: {exc}",
                     "check the file is a SQLite database (or move it away and run init-db)")
    if version == SCHEMA_VERSION:
        return Check("database", True, f"{path}: schema version {version}")
    if version < SCHEMA_VERSION:
        return Check("database", True, f"{path}: schema version {version}, current "
                     f"{SCHEMA_VERSION} (migrated automatically on first use)",
                     "run 'python -m chest_hunter init-db' to migrate now", warning=True)
    return Check("database", False, f"{path}: schema version {version} is newer than this "
                 f"program ({SCHEMA_VERSION})", "update the program, or use another --db")


def check_model(settings: Any, model_dir: Path | None) -> list[Check]:
    if model_dir is None:
        return [Check("model", True, "no model configured (deterministic policy only)",
                      warning=True)]
    meta_path = model_dir / "policy.json"
    if not meta_path.exists():
        return [Check("model", False, f"{model_dir}: policy.json missing",
                      "train and export a policy (train-policy, export-policy) or pass "
                      "--model")]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checks = []
    fmt = meta.get("format")
    if fmt != POLICY_FORMAT:
        checks.append(Check("model:format", False, f"format {fmt!r}, expected {POLICY_FORMAT}",
                            "re-export the model with this version (export-policy)"))
        return checks
    arch = meta.get("architecture", {})
    checks.append(Check("model:format", True, f"{meta.get('model_version')} ({fmt}, input "
                        f"{arch.get('width')}x{arch.get('height')}, {meta.get('parameters')} "
                        "parameters)"))
    try:
        from chest_hunter.learning.learned_policy import LearnedPolicy

        policy = LearnedPolicy(model_dir)
        checks.append(Check("model:load", True, f"backend {policy.backend}, version "
                            f"{policy.version}"))
    except Exception as exc:                       # report every load problem
        checks.append(Check("model:load", False, f"{type(exc).__name__}: {exc}",
                            "install onnxruntime (requirements-lock.txt) or re-export"))
    return checks


def check_observation(settings: Any) -> list[Check]:
    from chest_hunter.config import SUPPORTED_FRAME_SIZES

    out = []
    c3, e = settings.simulation_3d, settings.exploration
    for name, size in (("observation", (c3.observation_width, c3.observation_height)),
                       ("observation:benchmark", (e.benchmark_observation_width,
                                                  e.benchmark_observation_height))):
        ok = size in SUPPORTED_FRAME_SIZES
        out.append(Check(name, ok, f"{size[0]}x{size[1]}",
                         "" if ok else "use one of " + ", ".join(
                             f"{w}x{h}" for w, h in SUPPORTED_FRAME_SIZES)))
    return out


def check_memory(settings: Any) -> Check:
    from chest_hunter.exploration.self_improving import MEMORY_VERSION, ExplorationMemory
    from chest_hunter.runtime.configurations import AGENT_VERSION
    from chest_hunter.simulation3d.generator7 import GENERATOR_VERSION

    path = settings.exploration_memory_path()
    if not path.exists():
        return Check("exploration_memory", True, f"{path} does not exist (fixed default "
                     "strategy)", "train it with 'python -m chest_hunter train-exploration'",
                     warning=True)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        mem = ExplorationMemory.load(path, agent_version=AGENT_VERSION,
                                     generator_version=GENERATOR_VERSION)
    except (ValueError, KeyError, OSError) as exc:
        return Check("exploration_memory", False, f"{path}: {exc}",
                     "move the file away and retrain (train-exploration)")
    if mem.updates == 0 and raw.get("updates"):
        return Check("exploration_memory", True,
                     f"{path} was trained for agent {raw.get('agent_version')} / generator "
                     f"{raw.get('generator_version')}; ignored for {AGENT_VERSION} / "
                     f"{GENERATOR_VERSION}", "retrain with train-exploration", warning=True)
    return Check("exploration_memory", True, f"{path}: format version {MEMORY_VERSION}, "
                 f"{mem.updates} training episode(s), families {sorted(mem.categories)}")


def check_bridge(settings: Any, backend: str) -> list[Check]:
    from chest_hunter.unreal_bridge.client import BridgeClient
    from chest_hunter.unreal_bridge.mock_server import ENVIRONMENT_VERSION, MockUnrealServer
    from chest_hunter.unreal_bridge.protocol import PROTOCOL_VERSION

    server = None
    try:
        if backend == "reference":
            server = MockUnrealServer().start()
            host, port = "127.0.0.1", server.port
        else:
            host, port = settings.unreal_bridge.host, settings.unreal_bridge.port
        client = BridgeClient(host, port, timeout=3.0, connect_attempts=1)
        try:
            info = client.connect()
        finally:
            client.close()
    except Exception as exc:
        return [Check("bridge", False, f"{backend} research server: {type(exc).__name__}: {exc}",
                      "start the Unreal research build (docs/phase4_3d_architecture.md) or use "
                      "--backend reference" if backend == "unreal"
                      else "check that loopback TCP is allowed")]
    finally:
        if server is not None:
            server.stop()
    checks = [Check("bridge:protocol", info.get("version") == PROTOCOL_VERSION,
                    f"{info.get('server')} ({info.get('engine')}) on {host}:{port}, protocol "
                    f"{info.get('protocol')} v{info.get('version')}",
                    "" if info.get("version") == PROTOCOL_VERSION else
                    "use a client and server of the same protocol version")]
    env = info.get("environment_version")
    if env is None:
        checks.append(Check("bridge:environment", True, "the server does not report an "
                            "environment version (Unreal plug-in predates Phase 7)",
                            "Phase 7 procedural maps need the exported layouts on the server",
                            warning=True))
    else:
        same = backend != "reference" or env == ENVIRONMENT_VERSION
        checks.append(Check("bridge:environment", same, f"environment {env}",
                            "" if same else f"expected {ENVIRONMENT_VERSION}"))
    return checks


def check_output_dirs(settings: Any) -> list[Check]:
    out = []
    base = Path(settings.base_dir)
    for name, path in (("output:benchmark_results", base / "benchmark_results"),
                       ("output:database_dir", Path(settings.database.path).parent),
                       ("output:exploration_memory_dir",
                        settings.exploration_memory_path().parent)):
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path, prefix=".write-test-"):
                pass
            out.append(Check(name, True, f"{path} is writable"))
        except OSError as exc:
            out.append(Check(name, False, f"{path}: {exc}",
                             "create the directory or give the user write access"))
    return out


def run_startup_checks(settings: Any, *, backend: str = "reference",
                       model_dir: Path | None = None) -> list[Check]:
    checks: list[Check] = []
    checks += check_python(Path(settings.base_dir))
    checks.append(check_configuration(settings))
    checks.append(check_database(settings))
    checks += check_model(settings, model_dir)
    checks += check_observation(settings)
    checks.append(check_memory(settings))
    checks += check_bridge(settings, backend)
    checks += check_output_dirs(settings)
    return checks


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "FAIL" if not c.ok else ("warn" if c.warning else " ok ")
        lines.append(f"[{mark}] {c.name}: {c.detail}")
        if c.remedy and (not c.ok or c.warning):
            lines.append(f"       -> {c.remedy}")
    failed = sum(not c.ok for c in checks)
    lines.append(f"{len(checks)} checks, {failed} failed, "
                 f"{sum(c.ok and c.warning for c in checks)} warning(s)")
    return "\n".join(lines)
