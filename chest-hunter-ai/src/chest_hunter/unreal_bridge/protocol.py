"""Wire protocol between the Python agent and the Unreal research environment.

Transport: one TCP connection on the loopback interface (127.0.0.1 only).

Message framing (identical in both directions)::

    offset  size  field
    0       4     magic  b"CHRB"
    4       2     protocol version (unsigned, big-endian)
    6       2     flags (reserved, 0)
    8       4     header length H (unsigned, big-endian), 2 <= H <= 64 KiB
    12      4     payload length P (unsigned, big-endian), P <= 8 MiB
    16      H     header: UTF-8 JSON object
    16 + H  P     payload: raw bytes (camera image), may be empty

Images are never encoded as JSON arrays: the header carries the metadata
(frame id, timestamp, size, encoding, camera intrinsics) and the payload the
pixels (``bgr8``: rows top to bottom, 3 bytes per pixel).

Requests carry ``{"command": ..., "request_id": n, "args": {...}}``;
responses ``{"type": "response", "command": ..., "request_id": n,
"status": "ok" | "error", ...}``. Every input is validated here, in one
place, by both the mock server and (mirrored in C++) the Unreal plugin.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final

MAGIC: Final = b"CHRB"
PROTOCOL_NAME: Final = "chest-hunter-bridge"
PROTOCOL_VERSION: Final = 1
PREFIX: Final = struct.Struct("!4sHHII")
MAX_HEADER_BYTES: Final = 64 * 1024
MAX_PAYLOAD_BYTES: Final = 8 * 1024 * 1024
DEFAULT_PORT: Final = 47820

COMMANDS: Final = ("HELLO", "RESET_EPISODE", "GET_OBSERVATION", "STEP", "GET_EPISODE_STATUS",
                   "STOP_EPISODE", "PING", "NAV_QUERY", "NAV_RAYCAST", "GET_ORACLE_STATE",
                   "GET_EVALUATION")
#: Commands that expose ground truth. Servers refuse them unless started in oracle
#: or evaluation mode; the agent's port never sends them.
PRIVILEGED_COMMANDS: Final = frozenset({"GET_ORACLE_STATE", "GET_EVALUATION"})

SUPPORTED_SIZES: Final = ((320, 180), (480, 270), (640, 360), (800, 450))
GRAPHICS_QUALITIES: Final = ("low", "medium", "high")
#: navmesh: path queries and local probes; local: local probes only (no global path
#: planning); none: no navigation service at all.
NAVIGATION_MODES: Final = ("navmesh", "local", "none")
MAX_RAYS: Final = 32
MAX_TICKS_PER_STEP: Final = 30

ERROR_CODES: Final = (
    "bad_magic", "version_mismatch", "too_large", "bad_json", "bad_header", "unknown_command",
    "invalid_argument", "invalid_action", "no_episode", "stale_episode", "stale_step",
    "out_of_order", "episode_ended", "navigation_disabled", "forbidden", "busy", "timeout",
    "internal",
)

_FLOAT_ACTIONS: Final = ("move_forward", "move_right", "look_yaw", "look_pitch")
_BOOL_ACTIONS: Final = ("jump", "interact", "sprint", "crouch")
ACTION_KEYS: Final = _FLOAT_ACTIONS + _BOOL_ACTIONS
REQUIRED_ACTION_KEYS: Final = ("move_forward", "move_right", "look_yaw", "look_pitch", "jump",
                               "interact")
IDLE_ACTION: Final[dict[str, Any]] = {k: 0.0 for k in _FLOAT_ACTIONS} | {
    "jump": False, "interact": False}


class ProtocolError(Exception):
    """A protocol violation, carrying a machine-readable ``code`` from :data:`ERROR_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class Message:
    header: dict[str, Any]
    payload: bytes = b""


# -- framing ------------------------------------------------------------------------------


def encode_message(header: dict[str, Any], payload: bytes = b"") -> bytes:
    try:
        raw = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("bad_header", f"header is not JSON-serialisable: {exc}") from exc
    if len(raw) > MAX_HEADER_BYTES:
        raise ProtocolError("too_large", f"header of {len(raw)} bytes exceeds {MAX_HEADER_BYTES}")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError("too_large",
                            f"payload of {len(payload)} bytes exceeds {MAX_PAYLOAD_BYTES}")
    return PREFIX.pack(MAGIC, PROTOCOL_VERSION, 0, len(raw), len(payload)) + raw + payload


def parse_prefix(prefix: bytes) -> tuple[int, int]:
    """Validate the 16-byte prefix; return (header length, payload length)."""
    if len(prefix) != PREFIX.size:
        raise ProtocolError("bad_header", "truncated message prefix")
    magic, version, _flags, hlen, plen = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError("bad_magic", f"expected {MAGIC!r}, got {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ProtocolError("version_mismatch",
                            f"peer speaks protocol version {version}, this side {PROTOCOL_VERSION}")
    if not 2 <= hlen <= MAX_HEADER_BYTES:
        raise ProtocolError("too_large" if hlen > MAX_HEADER_BYTES else "bad_header",
                            f"header length {hlen} outside [2, {MAX_HEADER_BYTES}]")
    if plen > MAX_PAYLOAD_BYTES:
        raise ProtocolError("too_large", f"payload length {plen} exceeds {MAX_PAYLOAD_BYTES}")
    return hlen, plen


def parse_header(raw: bytes) -> dict[str, Any]:
    try:
        header = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError("bad_json", f"header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ProtocolError("bad_header", "header must be a JSON object")
    return header


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not allowed")


def read_message(recv_exact: Callable[[int], bytes]) -> Message:
    """Read one framed message using ``recv_exact(n)`` (which returns exactly n bytes)."""
    hlen, plen = parse_prefix(recv_exact(PREFIX.size))
    header = parse_header(recv_exact(hlen))
    payload = recv_exact(plen) if plen else b""
    return Message(header, payload)


# -- request validation --------------------------------------------------------------------


def _number(name: str, value: Any, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("invalid_argument", f"{name} must be a number")
    v = float(value)
    if not math.isfinite(v) or not lo <= v <= hi:
        raise ProtocolError("invalid_argument", f"{name} must be within [{lo}, {hi}], got {value}")
    return v


def _integer(name: str, value: Any, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError("invalid_argument", f"{name} must be an integer")
    if not lo <= value <= hi:
        raise ProtocolError("invalid_argument", f"{name} must be within [{lo}, {hi}], got {value}")
    return value


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError("invalid_argument", f"{name} must be true or false")
    return value


def _string(name: str, value: Any, max_len: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_len:
        raise ProtocolError("invalid_argument", f"{name} must be a non-empty string")
    return value


def _vec3(name: str, value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ProtocolError("invalid_argument", f"{name} must be [x, y, z]")
    x, y, z = (_number(f"{name}[{i}]", v, -1e5, 1e5) for i, v in enumerate(value))
    return (x, y, z)


def validate_action(action: Any) -> dict[str, Any]:
    """Validate an action object; unknown keys, wrong types and out-of-range values fail.

    Analogue axes are normalised to [-1, 1]; ``look_yaw``/``look_pitch`` are
    rates (fractions of the configured maximum turn rate).
    """
    if not isinstance(action, dict):
        raise ProtocolError("invalid_action", "action must be a JSON object")
    unknown = sorted(set(action) - set(ACTION_KEYS))
    if unknown:
        raise ProtocolError("invalid_action", f"unknown action keys: {', '.join(unknown)}")
    missing = [k for k in REQUIRED_ACTION_KEYS if k not in action]
    if missing:
        raise ProtocolError("invalid_action", f"missing action keys: {', '.join(missing)}")
    out: dict[str, Any] = {}
    try:
        for key in _FLOAT_ACTIONS:
            out[key] = _number(key, action[key], -1.0, 1.0)
        for key in _BOOL_ACTIONS:
            out[key] = _boolean(key, action.get(key, False))
    except ProtocolError as exc:
        raise ProtocolError("invalid_action", exc.message) from exc
    return out


@dataclass(frozen=True, slots=True)
class Request:
    command: str
    request_id: int
    args: dict[str, Any] = field(default_factory=dict)


def validate_request(header: dict[str, Any]) -> Request:
    """Check the envelope and the arguments of every command."""
    command = header.get("command")
    if not isinstance(command, str):
        raise ProtocolError("bad_header", "missing 'command'")
    if command not in COMMANDS:
        raise ProtocolError("unknown_command", f"unknown command {command!r}")
    request_id = _integer("request_id", header.get("request_id"), 1, 2 ** 53)
    raw_args = header.get("args", {})
    if not isinstance(raw_args, dict):
        raise ProtocolError("invalid_argument", "args must be an object")
    allowed = _ARGUMENTS[command]
    unknown = sorted(set(raw_args) - set(allowed))
    if unknown:
        raise ProtocolError("invalid_argument",
                            f"{command} does not accept: {', '.join(unknown)}")
    args: dict[str, Any] = {}
    for key, (required, check) in allowed.items():
        if key not in raw_args:
            if required:
                raise ProtocolError("invalid_argument", f"{command} requires '{key}'")
            continue
        args[key] = check(key, raw_args[key])
    return Request(command, request_id, args)


def _check_size(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ProtocolError("invalid_argument", f"{name} must be {{width, height}}")
    w = _integer(f"{name}.width", value.get("width"), 1, 10_000)
    h = _integer(f"{name}.height", value.get("height"), 1, 10_000)
    if (w, h) not in SUPPORTED_SIZES:
        sizes = ", ".join(f"{a}x{b}" for a, b in SUPPORTED_SIZES)
        raise ProtocolError("invalid_argument", f"observation size {w}x{h} not supported ({sizes})")
    return (w, h)


def _bearings(name: str, value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_RAYS:
        raise ProtocolError("invalid_argument", f"{name} must be a list of 1-{MAX_RAYS} angles")
    return tuple(_number(f"{name}[{i}]", v, -720.0, 720.0) for i, v in enumerate(value))


def _choice(options: tuple[str, ...]) -> Callable[[str, Any], str]:
    def check(name: str, value: Any) -> str:
        if value not in options:
            raise ProtocolError("invalid_argument", f"{name} must be one of {', '.join(options)}")
        return str(value)
    return check


_Check = Callable[[str, Any], Any]
_ARGUMENTS: Final[dict[str, dict[str, tuple[bool, _Check]]]] = {
    "HELLO": {
        "protocol": (True, lambda n, v: _string(n, v)),
        "version": (True, lambda n, v: _integer(n, v, 0, 1000)),
        "client": (False, lambda n, v: _string(n, v)),
    },
    "RESET_EPISODE": {
        "map_id": (True, lambda n, v: _string(n, v)),
        "seed": (True, lambda n, v: _integer(n, v, 0, 2 ** 31 - 1)),
        "episode": (False, lambda n, v: _integer(n, v, 0, 2 ** 31 - 1)),
        "observation": (False, _check_size),
        "fov_deg": (False, lambda n, v: _number(n, v, 60.0, 110.0)),
        "graphics_quality": (False, _choice(GRAPHICS_QUALITIES)),
        "max_ticks": (False, lambda n, v: _integer(n, v, 1, 10_000_000)),
        "navigation": (False, _choice(NAVIGATION_MODES)),
    },
    "GET_OBSERVATION": {"episode_id": (True, lambda n, v: _string(n, v))},
    "STEP": {
        "episode_id": (True, lambda n, v: _string(n, v)),
        "step_id": (True, lambda n, v: _integer(n, v, 1, 2 ** 53)),
        "action": (True, lambda n, v: validate_action(v)),
        "ticks": (False, lambda n, v: _integer(n, v, 1, MAX_TICKS_PER_STEP)),
        "observe": (False, _boolean),
    },
    "GET_EPISODE_STATUS": {"episode_id": (True, lambda n, v: _string(n, v))},
    "STOP_EPISODE": {
        "episode_id": (True, lambda n, v: _string(n, v)),
        "reason": (False, lambda n, v: _string(n, v, 256)),
    },
    "PING": {"nonce": (False, lambda n, v: _integer(n, v, 0, 2 ** 53))},
    "NAV_QUERY": {
        "episode_id": (True, lambda n, v: _string(n, v)),
        "start": (True, _vec3),
        "goal": (True, _vec3),
        "goal_tolerance": (False, lambda n, v: _number(n, v, 0.1, 5.0)),
    },
    "NAV_RAYCAST": {
        "episode_id": (True, lambda n, v: _string(n, v)),
        "origin": (True, _vec3),
        "bearings_deg": (True, lambda n, v: _bearings(n, v)),
        "max_distance": (False, lambda n, v: _number(n, v, 0.5, 15.0)),
    },
    "GET_ORACLE_STATE": {"episode_id": (True, lambda n, v: _string(n, v))},
    "GET_EVALUATION": {
        "episode_id": (True, lambda n, v: _string(n, v)),
        "frame_id": (True, lambda n, v: _integer(n, v, 1, 2 ** 53)),
    },
}


def request_header(command: str, request_id: int,
                   args: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"command": command, "request_id": request_id, "args": dict(args or {})}


def ok_response(request: Request, **fields: Any) -> dict[str, Any]:
    return {"type": "response", "command": request.command, "request_id": request.request_id,
            "status": "ok", **fields}


def error_response(code: str, message: str, *, command: str | None = None,
                   request_id: int | None = None) -> dict[str, Any]:
    return {"type": "response", "command": command, "request_id": request_id, "status": "error",
            "error": {"code": code, "message": message}}
