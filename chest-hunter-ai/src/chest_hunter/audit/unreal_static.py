"""Phase 8 Unreal Engine integration audit - what can be established without Unreal.

Unreal Engine is not installed in the audit environment, so nothing here compiles, loads or
runs the Unreal project. The audit checks the project statically, runs the existing
source-consistency tests (``tests/test_unreal_sources.py``: protocol constants, command
names, error codes, sizes, action keys, chest and character dimensions, the C++ random roll
compiled with the host compiler, loopback-only binding, privileged-command gating), lists
what is blocked, and gives the exact commands for a machine with Unreal Engine 5.4.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

BLOCKED = [
    "compiling the ChestHunterResearch C++ module (UnrealBuildTool)",
    "running the C++ automation tests (ResearchProtocolTests.cpp)",
    "creating /Game/Maps/ResearchArena3D with Scripts/create_research_map.py",
    "loading the research map, spawning the third-person character and the chests",
    "camera capture (SceneCapture2D render target) and frame encoding",
    "the C++ bridge server accepting the Python client (HELLO, RESET_EPISODE, STEP, ...)",
    "Unreal interaction validation, chest state change and the opened-chest mesh",
    "episode reset and termination inside Unreal",
    "every Python benchmark against --backend unreal",
    "Unreal frame time, GPU load and memory",
]

COMMANDS = [
    "# On a machine with Unreal Engine 5.4 (Windows or Linux), from the repository root:",
    "python -m chest_hunter export-arena-3d --map research_arena_3d "
    "--map p7_complex_test_000_unfamiliar",
    "<UE>/Engine/Build/BatchFiles/Build.(bat|sh) ChestHunterResearchEditor <Platform> "
    "Development -Project=$PWD/unreal/ChestHunterResearch/ChestHunterResearch.uproject",
    "<UE>/Engine/Binaries/<Platform>/UnrealEditor-Cmd "
    "unreal/ChestHunterResearch/ChestHunterResearch.uproject -run=pythonscript "
    "-script=$PWD/unreal/ChestHunterResearch/Scripts/create_research_map.py -unattended",
    "<UE>/Engine/Binaries/<Platform>/UnrealEditor-Cmd "
    "unreal/ChestHunterResearch/ChestHunterResearch.uproject "
    "-ExecCmds=\"Automation RunTests ChestHunter; Quit\" -unattended -nullrhi",
    "<UE>/Engine/Binaries/<Platform>/UnrealEditor "
    "unreal/ChestHunterResearch/ChestHunterResearch.uproject -game "
    "-ResearchEvaluation -windowed -ResX=640 -ResY=360   "
    "# bridge on by default (-ResearchBridgeOff disables); -ResearchEvaluation implies the "
    "oracle channel, for scoring only",
    "python -m chest_hunter check-runtime --backend unreal",
    "python -m chest_hunter simulate-3d --backend unreal --agent hybrid --map research_arena_3d"
    " --evaluate",
    "python -m chest_hunter simulate-3d --backend unreal --exploration active "
    "--map p7_complex_test_000_unfamiliar --evaluate",
]


def _unreal_installed() -> dict[str, Any]:
    names = ("UnrealEditor", "UnrealEditor-Cmd", "UE4Editor")
    found = {n: shutil.which(n) for n in names if shutil.which(n)}
    env = {k: v for k, v in os.environ.items() if k.upper().startswith(("UE_", "UNREAL"))}
    return {"executables_on_path": found, "environment": sorted(env),
            "installed": bool(found)}


def audit(root: Path) -> dict[str, Any]:
    proj = root / "unreal" / "ChestHunterResearch"
    out: dict[str, Any] = {"project_dir": str(proj)}
    uproject = proj / "ChestHunterResearch.uproject"
    data = json.loads(uproject.read_text(encoding="utf-8"))
    out["uproject"] = {"valid_json": True, "engine": data.get("EngineAssociation"),
                       "modules": [m["Name"] for m in data.get("Modules", [])],
                       "plugins": {p["Name"]: p.get("Enabled") for p in data.get("Plugins", [])}}
    build = (proj / "Source" / "ChestHunterResearch" / "ChestHunterResearch.Build.cs").read_text()
    out["build_dependencies"] = re.findall(r'"([A-Za-z]+)"', build.split("AddRange", 1)[-1])
    sources = sorted(proj.rglob("*.cpp")) + sorted(proj.rglob("*.h"))
    out["cpp_files"] = len(sources)
    out["cpp_lines"] = sum(p.read_text(encoding="utf-8").count("\n") for p in sources)
    out["binaries_present"] = (proj / "Binaries").exists()
    out["maps_present"] = sorted(str(p.relative_to(proj)) for p in proj.rglob("*.umap"))
    out["uassets_present"] = len(list(proj.rglob("*.uasset")))
    refs = sorted({m for p in sources for m in re.findall(r'TEXT\("(/(?:Game|Engine)/[^"]+)"\)',
                                                        p.read_text(encoding="utf-8"))})
    out["asset_references"] = refs
    out["asset_references_outside_engine_content"] = [r for r in refs
                                                      if not r.startswith("/Engine/")]
    out["layouts"] = sorted(p.stem for p in (proj / "Content" / "ResearchData").glob("*.json"))
    ini = (proj / "Config" / "DefaultEngine.ini").read_text(encoding="utf-8")
    m = re.search(r"GameDefaultMap=(\S+)", ini)
    out["default_map"] = m.group(1) if m else None
    out["default_map_exists"] = bool(out["maps_present"])
    out["unreal_engine"] = _unreal_installed()
    out["host_cpp_compiler"] = shutil.which("g++") or shutil.which("clang++")
    proc = subprocess.run([sys.executable, "-m", "unittest", "tests.test_unreal_sources", "-v"],
                          cwd=root, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(root / "src")))
    lines = proc.stderr.splitlines()
    out["source_consistency_tests"] = {
        "returncode": proc.returncode,
        "results": [ln for ln in lines if ln.endswith((" ok", "FAIL", "ERROR"))
                    or "skipped" in ln],
        "summary": [ln for ln in lines if ln.startswith(("Ran ", "OK", "FAILED"))]}
    out["blocked"] = BLOCKED
    out["commands_for_unreal_machine"] = COMMANDS
    out["verdict"] = ("UNVERIFIED: the Unreal project is source and configuration only (no "
                      "binaries, no map asset); it has never been compiled or run. Protocol "
                      "and parameter consistency with the Python side is checked statically.")
    return out
