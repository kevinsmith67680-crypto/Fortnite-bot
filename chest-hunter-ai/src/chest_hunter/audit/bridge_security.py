"""Phase 8 communication and security audit of the localhost research bridge.

Raw-socket checks against the Python reference server (``MockUnrealServer``). They bypass
the Python client's own validation on purpose: the server must defend itself. The Unreal
server implements the same checks in C++ (``ResearchProtocol.cpp``,
``ResearchBridgeServer.cpp``); that implementation was compared statically, not executed.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import Any


def _send(sock: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    from chest_hunter.unreal_bridge.protocol import encode_message

    sock.sendall(encode_message(header, payload))


def _recv(sock: socket.socket, timeout: float = 5.0) -> dict[str, Any] | None:
    from chest_hunter.unreal_bridge.protocol import read_message

    sock.settimeout(timeout)

    def exact(n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        return buf

    try:
        return read_message(exact).header
    except (ConnectionError, OSError, ValueError):
        return None


def _closed(sock: socket.socket, timeout: float = 3.0) -> bool:
    sock.settimeout(timeout)
    try:
        return sock.recv(1) == b""
    except (ConnectionResetError, BrokenPipeError):
        return True
    except OSError:
        return False


def evaluate() -> dict[str, Any]:
    from chest_hunter.unreal_bridge.client import BridgeClient
    from chest_hunter.unreal_bridge.mock_server import MockUnrealServer
    from chest_hunter.unreal_bridge.protocol import (
        MAGIC,
        PREFIX,
        PROTOCOL_NAME,
        PROTOCOL_VERSION,
        request_header,
    )

    def _req(command: str, rid: int, **args: Any) -> dict[str, Any]:
        return request_header(command, rid, args)

    checks: list[dict[str, Any]] = []

    def check(name: str, fn: Callable[[], tuple[bool, Any]]) -> None:
        try:
            ok, detail = fn()
        except Exception as exc:                              # a crashing check fails
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    def bind_public() -> tuple[bool, Any]:
        try:
            MockUnrealServer(host="0.0.0.0")
            return False, "server accepted 0.0.0.0"
        except ValueError as exc:
            return True, str(exc)

    def client_remote() -> tuple[bool, Any]:
        try:
            BridgeClient("192.0.2.10", 47820)
            return False, "client accepted a non-loopback host"
        except Exception as exc:
            return True, type(exc).__name__

    check("server refuses to bind a non-loopback address", bind_public)
    check("client refuses a non-loopback server", client_remote)

    server = MockUnrealServer(read_timeout=1.0, idle_timeout=30.0).start()
    port = server.port

    def connect(hello: bool = True) -> socket.socket:
        s = socket.create_connection(("127.0.0.1", port), timeout=5)
        if hello:
            _send(s, _req("HELLO", 1, protocol=PROTOCOL_NAME,
                                    version=PROTOCOL_VERSION, client="audit"))
            _recv(s)
        return s

    def episode(s: socket.socket) -> str:
        _send(s, _req("RESET_EPISODE", 2, map_id="research_arena_3d", seed=1,
                                episode=0, observation={"width": 320, "height": 180}))
        reply = _recv(s, 20)
        assert reply is not None and reply.get("status") == "ok", reply
        return str(reply["episode_id"])

    def step(s: socket.socket, eid: str, sid: int, rid: int, **action: Any) -> Any:
        act = {"move_forward": 0.0, "move_right": 0.0, "look_yaw": 0.0, "look_pitch": 0.0,
               "jump": False, "interact": False, "sprint": False, "crouch": False} | action
        _send(s, _req("STEP", rid, episode_id=eid, step_id=sid, action=act,
                                ticks=1, observe=False))
        return _recv(s, 10)

    try:
        def oversized_header() -> tuple[bool, Any]:
            s = connect(hello=False)
            s.sendall(PREFIX.pack(MAGIC, PROTOCOL_VERSION, 0, 10 * 1024 * 1024, 0))
            reply = _recv(s, 3)
            ok = _closed(s) or (reply or {}).get("status") == "error"
            s.close()
            return ok, reply

        def oversized_payload() -> tuple[bool, Any]:
            s = connect(hello=False)
            raw = json.dumps({"command": "PING", "request_id": 1}).encode()
            s.sendall(PREFIX.pack(MAGIC, PROTOCOL_VERSION, 0, len(raw), 512 * 1024 * 1024)
                      + raw)
            reply = _recv(s, 3)
            ok = _closed(s) or (reply or {}).get("status") == "error"
            s.close()
            return ok, reply

        def malformed_json() -> tuple[bool, Any]:
            s = connect(hello=False)
            raw = b"{not json"
            s.sendall(PREFIX.pack(MAGIC, PROTOCOL_VERSION, 0, len(raw), 0) + raw)
            reply = _recv(s, 3)
            ok = _closed(s) or (reply or {}).get("status") == "error"
            s.close()
            return ok, reply

        def bad_magic() -> tuple[bool, Any]:
            s = connect(hello=False)
            s.sendall(b"XXXX" + struct.pack("!HHII", PROTOCOL_VERSION, 0, 2, 0) + b"{}")
            reply = _recv(s, 3)                  # an error reply, then the connection closes
            ok = ((reply or {}).get("status") == "error" or reply is None) and _closed(s)
            s.close()
            return ok, (reply or {}).get("error")

        def bad_version() -> tuple[bool, Any]:
            s = connect(hello=False)
            _send(s, _req("HELLO", 1, protocol=PROTOCOL_NAME, version=999,
                                    client="audit"))
            reply = _recv(s, 3) or {}
            s.close()
            time.sleep(0.3)
            return reply.get("status") == "error", reply.get("error", reply)

        check("oversized header is refused", oversized_header)
        check("oversized payload is refused without reading it", oversized_payload)
        check("malformed JSON header is refused", malformed_json)
        check("wrong magic bytes close the connection", bad_magic)
        check("protocol version mismatch is refused", bad_version)

        bad_actions = {"NaN": {"move_forward": float("nan")},
                       "infinity": {"look_yaw": float("inf")},
                       "out of range": {"move_forward": 5.0},
                       "wrong type": {"interact": "yes"},
                       "unknown action key": {"teleport": [1, 2, 3]}}
        results: dict[str, Any] = {}
        idle = {"move_forward": 0.0, "move_right": 0.0, "look_yaw": 0.0, "look_pitch": 0.0,
                "jump": False, "interact": False, "sprint": False, "crouch": False}
        for k, (label, act) in enumerate(bad_actions.items()):
            # each on a fresh connection: a refusal may legitimately close it (NaN and
            # infinity are not JSON; the server refuses to parse them)
            c = connect()
            ceid = episode(c)
            raw = json.dumps(_req("STEP", 100 + k, episode_id=ceid, step_id=1,
                                  action=idle | act, ticks=1, observe=False),
                             allow_nan=True).encode()
            try:
                c.sendall(PREFIX.pack(MAGIC, PROTOCOL_VERSION, 0, len(raw), 0) + raw)
                reply = _recv(c, 5)
            except OSError:
                reply = None
            results[label] = "closed" if reply is None else (
                "error" if reply.get("status") == "error" else "ACCEPTED")
            c.close()
            time.sleep(0.2)
        c = connect()
        ceid = episode(c)
        _send(c, _req("STEP", 200, episode_id=ceid, step_id=1, action=idle, ticks=500,
                      observe=False))
        reply = _recv(c) or {}
        results["ticks above the per-step limit"] = "error" if reply.get("status") == "error" \
            else "ACCEPTED"
        c.close()
        time.sleep(0.2)
        checks.append({"check": "invalid actions are refused by the server",
                       "pass": "ACCEPTED" not in results.values(), "detail": results})
        s = connect()
        eid = episode(s)

        def duplicate() -> tuple[bool, Any]:
            first = step(s, eid, 1, 300, move_forward=1.0) or {}
            again = step(s, eid, 1, 301, move_forward=1.0) or {}
            t1 = first.get("telemetry", {}).get("sim_time")
            t2 = again.get("telemetry", {}).get("sim_time")
            return again.get("duplicate") is True and t1 == t2, {"first": t1, "replayed": t2}

        def stale() -> tuple[bool, Any]:
            step(s, eid, 2, 302)
            reply = step(s, eid, 1, 303) or {}
            skip = step(s, eid, 9, 304) or {}
            return reply.get("status") == "error" and skip.get("status") == "error", \
                [reply.get("error"), skip.get("error")]

        def privileged() -> tuple[bool, Any]:
            _send(s, _req("GET_ORACLE_STATE", 400, episode_id=eid))
            a = _recv(s) or {}
            _send(s, _req("GET_EVALUATION", 401, episode_id=eid, frame_id=1))
            b = _recv(s) or {}
            return a.get("status") == "error" and b.get("status") == "error", \
                [a.get("error"), b.get("error")]

        check("a replayed step is not applied twice", duplicate)
        check("stale and out-of-order steps are refused", stale)
        check("privileged requests are refused without the server flag", privileged)

        def busy() -> tuple[bool, Any]:
            s2 = connect(hello=False)
            _send(s2, _req("HELLO", 1, protocol=PROTOCOL_NAME,
                                     version=PROTOCOL_VERSION, client="second"))
            reply = _recv(s2, 3) or {}
            s2.close()
            return reply.get("status") == "error", reply.get("error")

        check("a second concurrent client is refused (busy)", busy)

        def abrupt_disconnect() -> tuple[bool, Any]:
            s.close()
            time.sleep(0.3)
            s3 = connect()
            eid3 = episode(s3)
            ok = (step(s3, eid3, 1, 2) or {}).get("status") == "ok"
            s3.close()
            return ok, None

        check("server recovers from an abrupt client disconnect", abrupt_disconnect)

        def slow_partial() -> tuple[bool, Any]:
            time.sleep(0.3)
            s4 = socket.create_connection(("127.0.0.1", port), timeout=5)
            s4.sendall(MAGIC + b"\x00")                  # start a message, never finish it
            started = time.monotonic()
            closed = _closed(s4, timeout=5.0)
            return closed and time.monotonic() - started < 4.0, \
                round(time.monotonic() - started, 2)

        check("a stalled partial message is dropped after the read timeout", slow_partial)
    finally:
        threads_before = threading.active_count()
        server.stop()
        time.sleep(0.5)
        checks.append({"check": "server stop releases its thread and listener",
                       "pass": not server.running and threading.active_count()
                       <= threads_before, "detail": {"running": server.running}})
    return {"checks": checks, "passed": sum(c["pass"] for c in checks), "total": len(checks),
            "server": "Python reference server (MockUnrealServer); the C++ server was checked "
                      "statically only"}
