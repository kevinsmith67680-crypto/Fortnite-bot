"""Consistency checks between the Unreal C++ sources and the Python side.

Unreal Engine is not available in CI, so the C++ module cannot be built or run here. These
tests check what can be checked without it:

* protocol constants, command names, error codes, observation sizes and navigation modes in
  the C++ sources match the Python protocol;
* chest dimensions and character parameters match the reference arena;
* the deterministic SplitMix64/UnitRoll code is extracted VERBATIM from ResearchProtocol.cpp,
  compiled with the host C++ compiler and compared with the Python implementation on many
  inputs (skipped when no compiler is installed);
* security properties visible in the source: loopback-only binding, no console/script
  execution entry points, privileged commands gated by command-line switches.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from chest_hunter.simulation3d.arena import (
    CHEST_BODY_HEIGHT,
    CHEST_DEPTH,
    CHEST_LID_HEIGHT,
    CHEST_WIDTH,
)
from chest_hunter.simulation3d.character import CharacterParams
from chest_hunter.simulation3d.environment3d import splitmix64, unit_roll
from chest_hunter.unreal_bridge import protocol

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "unreal" / "ChestHunterResearch"
SOURCE = PROJECT / "Source" / "ChestHunterResearch"


def _read(relative: str) -> str:
    return (SOURCE / relative).read_text(encoding="utf-8")


def _all_cpp() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(SOURCE.rglob("*.[hc]pp"))
                     if "Tests" not in p.parts)


def test_project_layout() -> None:
    uproject = json.loads((PROJECT / "ChestHunterResearch.uproject").read_text())
    assert uproject["Modules"][0]["Name"] == "ChestHunterResearch"
    for name in ("ChestHunterResearch.Target.cs", "ChestHunterResearchEditor.Target.cs",
                 "ChestHunterResearch/ChestHunterResearch.Build.cs"):
        assert (PROJECT / "Source" / name).is_file()
    for cls in ("ResearchProtocol", "ResearchChest", "ResearchCharacter", "ResearchGameMode",
                "ResearchArenaBuilder", "ResearchBridgeServer", "ResearchBridgeSubsystem"):
        assert (SOURCE / "Public" / f"{cls}.h").is_file()
        assert (SOURCE / "Private" / f"{cls}.cpp").is_file()
    assert (SOURCE / "Private" / "Tests" / "ResearchProtocolTests.cpp").is_file()
    assert (PROJECT / "Content" / "ResearchData" / "research_arena_3d.json").is_file()


def test_protocol_constants_match() -> None:
    h = _read("Public/ResearchProtocol.h")
    assert "{ 'C', 'H', 'R', 'B' }" in h and protocol.MAGIC == b"CHRB"
    assert re.search(r"Version = (\d+);", h).group(1) == str(protocol.PROTOCOL_VERSION)  # type: ignore[union-attr]
    assert "MaxHeaderBytes = 64 * 1024" in h and protocol.MAX_HEADER_BYTES == 64 * 1024
    assert "MaxPayloadBytes = 8 * 1024 * 1024" in h
    assert protocol.MAX_PAYLOAD_BYTES == 8 * 1024 * 1024
    assert f"MaxTicksPerStep = {protocol.MAX_TICKS_PER_STEP};" in h
    assert f"MaxRays = {protocol.MAX_RAYS};" in h
    assert f'TEXT("{protocol.PROTOCOL_NAME}")' in h
    assert f"PrefixSize = {protocol.PREFIX.size};" in h


def test_error_codes_and_commands_match() -> None:
    h = _read("Public/ResearchProtocol.h")
    errors_block = h.split("namespace Errors")[1].split("}")[0]
    cpp_codes = set(re.findall(r'TEXT\("([a-z_]+)"\)', errors_block))
    assert cpp_codes <= set(protocol.ERROR_CODES)
    assert set(protocol.ERROR_CODES) - cpp_codes <= {"timeout"}       # sent as a literal
    assert 'TEXT("timeout")' in _read("Private/ResearchBridgeServer.cpp")
    fatal = re.search(r"bool IsFatal.*?\{(.*?)\}", _read("Private/ResearchProtocol.cpp"), re.S)
    from chest_hunter.unreal_bridge.mock_server import FATAL_CODES
    fatal_names = {"BadMagic": "bad_magic", "VersionMismatch": "version_mismatch",
                   "TooLarge": "too_large", "BadJson": "bad_json", "BadHeader": "bad_header"}
    assert fatal is not None
    assert {fatal_names[n] for n in re.findall(r"Errors::(\w+)", fatal.group(1))} == FATAL_CODES
    cpp = _read("Private/ResearchProtocol.cpp")
    block = cpp.split("const TArray<FString>& Commands()")[1].split("return Names;")[0]
    assert re.findall(r'TEXT\("([A-Z_]+)"\)', block) == list(protocol.COMMANDS)


def test_observation_sizes_navigation_modes_and_qualities_match() -> None:
    sub = _read("Private/ResearchBridgeSubsystem.cpp")
    sizes = re.findall(r"\{ (\d+), (\d+) \}", sub.split("SupportedSizes[][2] =")[1].split(";")[0])
    assert [(int(w), int(h)) for w, h in sizes] == list(protocol.SUPPORTED_SIZES)
    assert re.findall(r'TEXT\("(\w+)"\)', sub.split('TEXT("navigation"), {')[1].split("}")[0]) \
        == list(protocol.NAVIGATION_MODES)
    assert re.findall(r'TEXT\("(\w+)"\)', sub.split('TEXT("graphics_quality"), {')[1]
                      .split("}")[0]) == list(protocol.GRAPHICS_QUALITIES)
    ini = (PROJECT / "Config" / "DefaultEngine.ini").read_text()
    assert f"Port={protocol.DEFAULT_PORT}" in ini


def test_action_keys_match() -> None:
    cpp = _read("Private/ResearchProtocol.cpp")
    floats = re.findall(r'TEXT\("(\w+)"\)', cpp.split("FloatActions()")[1].split("return Keys;")[0])
    bools = re.findall(r'TEXT\("(\w+)"\)', cpp.split("BoolActions()")[1].split("return Keys;")[0])
    assert tuple(floats + bools) == protocol.ACTION_KEYS
    assert "bool bRequired = I < 2;" in cpp                         # jump and interact
    assert set(protocol.REQUIRED_ACTION_KEYS) == set(floats) | set(bools[:2])


def test_chest_dimensions_and_character_parameters_match() -> None:
    chest = _read("Public/ResearchChest.h")
    for name, value in (("WidthM", CHEST_WIDTH), ("DepthM", CHEST_DEPTH),
                        ("BodyHeightM", CHEST_BODY_HEIGHT), ("LidHeightM", CHEST_LID_HEIGHT)):
        assert re.search(rf"{name} = ([\d.]+)f;", chest).group(1) == f"{value:g}"  # type: ignore[union-attr]
    character = _read("Public/ResearchCharacter.h")
    p = CharacterParams()
    for cpp_name, value in (("MaxSpeed", p.max_speed), ("SprintSpeed", p.sprint_speed),
                            ("JumpSpeed", p.jump_speed), ("YawRate", p.yaw_rate),
                            ("PitchRate", p.pitch_rate), ("PitchMin", p.pitch_min),
                            ("PitchMax", p.pitch_max), ("PivotHeight", p.pivot_height),
                            ("ArmLength", p.arm_length), ("ShoulderOffset", p.shoulder_offset),
                            ("CapsuleRadius", p.radius), ("CapsuleHeight", p.height),
                            ("StepHeight", p.step_height)):
        found = re.search(rf"\b{cpp_name} = (-?[\d.]+)f;", character)
        assert found is not None and float(found.group(1)) == value, cpp_name
    ini = (PROJECT / "Config" / "DefaultEngine.ini").read_text()
    assert f"AgentRadius={p.radius * 100:.1f}" in ini
    assert f"AgentMaxStepHeight={p.step_height * 100:.1f}" in ini


def test_reference_vectors_in_the_cpp_tests_match_python() -> None:
    tests = _read("Private/Tests/ResearchProtocolTests.cpp")
    assert f"{splitmix64(0):#x}ull" in tests and f"{splitmix64(42):#x}ull" in tests
    for args in re.findall(r"UnitRoll\((\d+), (\d+), (\d+), (\d+)\), ([\d.]+)", tests):
        seed, episode, index, salt = (int(a) for a in args[:4])
        assert unit_roll(seed, episode, index, salt) == float(args[4])


def _extract(cpp: str, signature: str) -> str:
    start = cpp.index(signature)
    depth, i = 0, cpp.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(cpp[i], 0)
        i += 1
        if depth == 0:
            return cpp[start:i]


@pytest.mark.skipif(shutil.which("g++") is None and shutil.which("clang++") is None,
                    reason="no C++ compiler")
def test_cpp_randomness_is_bit_identical_to_python(tmp_path: Path) -> None:
    cpp = _read("Private/ResearchProtocol.cpp")
    body = _extract(cpp, "uint64 SplitMix64(uint64 Value)") + "\n" + \
        _extract(cpp, "double UnitRoll(int32 Seed, int32 Episode, int32 Index, int32 Salt)")
    rng = random.Random(1234)
    cases = [(rng.randrange(2 ** 31), rng.randrange(1000), rng.randrange(64), rng.randrange(1000))
             for _ in range(500)] + [(0, 0, 0, 0), (2 ** 31 - 1, 2 ** 31 - 1, 65535, 65535)]
    program = tmp_path / "roll.cpp"
    program.write_text(
        "#include <cstdint>\n#include <cstdio>\n"
        "typedef uint64_t uint64; typedef uint32_t uint32; typedef int32_t int32;\n"
        "namespace ResearchProtocol {\n" + body + "\n}\n"
        "int main() { int s, e, i, t;\n"
        "  printf(\"%llu %llu\\n\", (unsigned long long)ResearchProtocol::SplitMix64(0),"
        " (unsigned long long)ResearchProtocol::SplitMix64(42));\n"
        "  while (scanf(\"%d %d %d %d\", &s, &e, &i, &t) == 4)"
        " printf(\"%.17g\\n\", ResearchProtocol::UnitRoll(s, e, i, t)); }\n")
    compiler = shutil.which("g++") or shutil.which("clang++")
    exe = tmp_path / "roll"
    subprocess.run([compiler, "-std=c++17", "-O1", "-o", str(exe), str(program)], check=True,  # type: ignore[list-item]
                   capture_output=True, timeout=120)
    stdin = "\n".join(" ".join(str(v) for v in c) for c in cases)
    out = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, check=True,
                         timeout=60).stdout.split("\n")
    assert out[0].split() == [str(splitmix64(0)), str(splitmix64(42))]
    for case, line in zip(cases, out[1:], strict=False):
        assert float(line) == unit_roll(*case), case


def test_server_is_loopback_only_and_has_no_remote_execution() -> None:
    server = _read("Private/ResearchBridgeServer.cpp")
    assert "FIPv4Address(127, 0, 0, 1)" in server
    assert "IsLoopbackPeer" in server and "(Ip >> 24) == 127" in server
    assert "FIPv4Address::Any" not in server and "0.0.0.0" not in server
    everything = _all_cpp()
    for forbidden in ("ConsoleCommand", "ExecuteConsoleCommand", "FPlatformProcess::CreateProc",
                      "PythonScript", "IPythonScriptPlugin", "FFileHelper::SaveStringToFile"):
        assert forbidden not in everything, forbidden
    # GEngine->Exec is only used for fixed scalability settings with integer levels.
    for call in re.findall(r"GEngine->Exec\(.*", everything):
        assert 'TEXT("sg.' in call and "%d" in call


def test_privileged_commands_are_gated() -> None:
    sub = _read("Private/ResearchBridgeSubsystem.cpp")
    assert 'FParse::Param(Cmd, TEXT("ResearchEvaluation"))' in sub
    assert 'FParse::Param(Cmd, TEXT("ResearchOracle"))' in sub
    oracle = _extract(sub, "bool UResearchBridgeSubsystem::OracleState(")
    evaluation = _extract(sub, "bool UResearchBridgeSubsystem::Evaluation(")
    assert oracle.index("if (!bOracleEnabled)") < oracle.index("OracleState()")
    assert evaluation.index("if (!bEvaluationEnabled)") < evaluation.index("EvaluationLabels()")
    # Agent-facing handlers never touch privileged data.
    for handler in ("GetObservation(", "BeginStep(", "CompleteStep(", "NavQuery(", "NavRaycast(",
                    "EpisodeStatus(", "ResetEpisode("):
        text = _extract(sub, f"UResearchBridgeSubsystem::{handler}")
        assert "OracleState" not in text and "EvaluationLabels" not in text, handler


def test_agent_capture_is_a_scene_capture_not_a_screenshot() -> None:
    character = _read("Private/ResearchCharacter.cpp")
    assert "USceneCaptureComponent2D" in character and "CaptureScene()" in character
    everything = _all_cpp()
    for forbidden in ("FScreenshotRequest", "HighResShot", "ReadScreenPixels", "GetDesktop"):
        assert forbidden not in everything
    labels = _extract(_read("Private/ResearchGameMode.cpp"),
                      "bool AResearchGameMode::Observe(")
    assert "if (bEvaluationEnabled)" in labels           # labels computed only when allowed
