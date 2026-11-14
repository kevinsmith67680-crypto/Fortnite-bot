"""Protocol-identical stand-in for the Unreal research server.

:class:`BridgeSession` implements the command semantics (shared by every
transport); :class:`MockUnrealServer` serves it over a loopback TCP socket
exactly like the Unreal plugin's ``FResearchBridgeServer``. The environment
behind it is the Python reference arena, not Unreal Engine: results obtained
with it say nothing about Unreal's rendering or physics, only about the
Python side of the system.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any

from chest_hunter.simulation3d.environment3d import EnvSettings, Environment3D
from chest_hunter.unreal_bridge.protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    SUPPORTED_SIZES,
    Message,
    ProtocolError,
    Request,
    encode_message,
    error_response,
    ok_response,
    read_message,
    validate_request,
)

logger = logging.getLogger(__name__)

SERVER_NAME = "chest-hunter-reference-server"
#: Errors after which the byte stream cannot be trusted: the connection is closed.
FATAL_CODES = frozenset({"bad_magic", "version_mismatch", "too_large", "bad_json", "bad_header"})


class BridgeSession:
    """Command handling for one research server (episode state survives reconnects)."""

    def __init__(self, *, oracle: bool = False, evaluation: bool = False,
                 base_settings: EnvSettings | None = None) -> None:
        self.oracle_enabled = oracle
        self.evaluation_enabled = evaluation
        self.base_settings = base_settings or EnvSettings()
        self.env: Environment3D | None = None
        self.episode_id: str | None = None
        self.navigation = "navmesh"
        self.last_step_id = 0
        self._last_step: tuple[dict[str, Any], bytes] | None = None
        self._episodes = 0
        self.steps_applied = 0
        self.observations_rendered = 0

    # -- dispatch -------------------------------------------------------------------------------

    def handle(self, request: Request) -> tuple[dict[str, Any], bytes]:
        handler = getattr(self, "_cmd_" + request.command.lower())
        result: tuple[dict[str, Any], bytes] = handler(request)
        return result

    def _require_episode(self, request: Request) -> Environment3D:
        if self.env is None or self.episode_id is None:
            raise ProtocolError("no_episode", "no episode has been started (RESET_EPISODE)")
        if request.args["episode_id"] != self.episode_id:
            raise ProtocolError("stale_episode",
                                f"episode {request.args['episode_id']!r} is not the current "
                                f"episode {self.episode_id!r}")
        return self.env

    # -- commands -------------------------------------------------------------------------------

    def _cmd_hello(self, request: Request) -> tuple[dict[str, Any], bytes]:
        if request.args["protocol"] != PROTOCOL_NAME:
            raise ProtocolError("version_mismatch", f"unknown protocol {request.args['protocol']!r}")
        if request.args["version"] != PROTOCOL_VERSION:
            raise ProtocolError("version_mismatch",
                                f"client protocol version {request.args['version']}, server "
                                f"{PROTOCOL_VERSION}")
        return ok_response(
            request, server=SERVER_NAME, engine="python-reference-arena",
            protocol=PROTOCOL_NAME, version=PROTOCOL_VERSION,
            maps=["research_arena_3d"],
            observation_sizes=[list(s) for s in SUPPORTED_SIZES],
            features={"navigation_service": True, "navigation_raycast": True,
                      "oracle": self.oracle_enabled,
                      "evaluation": self.evaluation_enabled},
        ), b""

    def _cmd_reset_episode(self, request: Request) -> tuple[dict[str, Any], bytes]:
        a = request.args
        width, height = a.get("observation", (self.base_settings.width, self.base_settings.height))
        settings = EnvSettings(**{
            **{f: getattr(self.base_settings, f) for f in self.base_settings.__dataclass_fields__},
            "width": width, "height": height,
            "hfov_deg": a.get("fov_deg", self.base_settings.hfov_deg),
            "graphics_quality": a.get("graphics_quality", self.base_settings.graphics_quality),
            "max_ticks": a.get("max_ticks", self.base_settings.max_ticks),
        })
        try:
            env = Environment3D(a["map_id"], seed=a["seed"], episode=a.get("episode", 0),
                                settings=settings)
        except ValueError as exc:
            raise ProtocolError("invalid_argument", str(exc)) from exc
        self._episodes += 1
        self.env = env
        self.episode_id = f"{a['seed']}-{a.get('episode', 0)}-{self._episodes}"
        self.navigation = a.get("navigation", "navmesh")
        self.last_step_id = 0
        self._last_step = None
        return ok_response(request, episode_id=self.episode_id, rules=env.rules(),
                           navigation=self.navigation, telemetry=env.telemetry()), b""

    def _cmd_get_observation(self, request: Request) -> tuple[dict[str, Any], bytes]:
        env = self._require_episode(request)
        frame, meta = env.observe(labels=self.evaluation_enabled)
        self.observations_rendered += 1
        return ok_response(request, episode_id=self.episode_id, step_id=self.last_step_id,
                           observation=meta), frame.tobytes()

    def _cmd_step(self, request: Request) -> tuple[dict[str, Any], bytes]:
        env = self._require_episode(request)
        step_id = request.args["step_id"]
        if step_id == self.last_step_id and self._last_step is not None:
            header, payload = self._last_step          # resend: never applied twice
            return {**header, "request_id": request.request_id, "duplicate": True}, payload
        if step_id <= self.last_step_id:
            raise ProtocolError("stale_step", f"step {step_id} is older than {self.last_step_id}")
        if step_id != self.last_step_id + 1:
            raise ProtocolError("out_of_order",
                                f"expected step {self.last_step_id + 1}, got {step_id}")
        if env.ended:
            raise ProtocolError("episode_ended", "the episode has ended")
        env.step(request.args["action"], request.args.get("ticks", 1))
        self.steps_applied += 1
        self.last_step_id = step_id
        fields: dict[str, Any] = {"episode_id": self.episode_id, "step_id": step_id,
                                  "telemetry": env.telemetry(), "ended": env.ended,
                                  "duplicate": False}
        payload = b""
        if request.args.get("observe", False):
            frame, meta = env.observe(labels=self.evaluation_enabled)
            self.observations_rendered += 1
            fields["observation"] = meta
            payload = frame.tobytes()
        header = ok_response(request, **fields)
        self._last_step = (header, payload)
        return header, payload

    def _cmd_get_episode_status(self, request: Request) -> tuple[dict[str, Any], bytes]:
        env = self._require_episode(request)
        state = "stopped" if env.stopped else ("ended" if env.ended else "running")
        return ok_response(request, episode_id=self.episode_id, state=state, tick=env.tick,
                           sim_time=round(env.time, 6), max_ticks=env.settings.max_ticks,
                           last_step_id=self.last_step_id), b""

    def _cmd_stop_episode(self, request: Request) -> tuple[dict[str, Any], bytes]:
        env = self._require_episode(request)
        env.stopped = True
        return ok_response(request, episode_id=self.episode_id, state="stopped",
                           summary=env.summary()), b""

    def _cmd_ping(self, request: Request) -> tuple[dict[str, Any], bytes]:
        return ok_response(request, pong=True, nonce=request.args.get("nonce"),
                           server_time=time.time()), b""

    def _cmd_nav_query(self, request: Request) -> tuple[dict[str, Any], bytes]:
        env = self._require_episode(request)
        if self.navigation != "navmesh":
            raise ProtocolError("navigation_disabled",
                                "the navigation service is disabled for this episode")
        a = request.args
        path = env.find_path(a["start"], a["goal"], a.get("goal_tolerance", 0.5))
        return ok_response(request, status_code=path.status,
                           waypoints=[[round(c, 4) for c in w] for w in path.waypoints],
                           length=round(path.length, 4)), b""

    def _cmd_nav_raycast(self, request: Request) -> tuple[dict[str, Any], bytes]:
        env = self._require_episode(request)
        if self.navigation == "none":
            raise ProtocolError("navigation_disabled",
                                "the navigation service is disabled for this episode")
        a = request.args
        reach = a.get("max_distance", 8.0)
        hits = [round(env.nav_raycast(a["origin"], b, reach), 3) for b in a["bearings_deg"]]
        return ok_response(request, distances=hits, max_distance=reach), b""

    def _cmd_get_oracle_state(self, request: Request) -> tuple[dict[str, Any], bytes]:
        if not (self.oracle_enabled or self.evaluation_enabled):
            raise ProtocolError("forbidden",
                                "the server was not started in oracle or evaluation mode")
        env = self._require_episode(request)
        return ok_response(request, state=env.oracle_state()), b""

    def _cmd_get_evaluation(self, request: Request) -> tuple[dict[str, Any], bytes]:
        if not self.evaluation_enabled:
            raise ProtocolError("forbidden", "the server was not started in evaluation mode")
        env = self._require_episode(request)
        labels = env.evaluation_labels()
        if labels is None or labels["frame_id"] != request.args["frame_id"]:
            raise ProtocolError("invalid_argument",
                                f"no labels for frame {request.args['frame_id']} (only the latest "
                                "rendered frame is labelled)")
        return ok_response(request, labels=labels), b""


def process(session: BridgeSession, message: Message) -> tuple[dict[str, Any], bytes, bool]:
    """Validate and handle one message: (response header, payload, close_connection)."""
    try:
        request = validate_request(message.header)
    except ProtocolError as exc:
        rid = message.header.get("request_id")
        return (error_response(exc.code, exc.message, command=message.header.get("command"),
                               request_id=rid if isinstance(rid, int) else None), b"",
                exc.code in FATAL_CODES)
    try:
        header, payload = session.handle(request)
        return header, payload, False
    except ProtocolError as exc:
        return error_response(exc.code, exc.message, command=request.command,
                              request_id=request.request_id), b"", False
    except Exception as exc:                     # never kill the server on a handler bug
        logger.exception("Internal error while handling %s", request.command)
        return error_response("internal", f"{type(exc).__name__}: {exc}", command=request.command,
                              request_id=request.request_id), b"", False


class MockUnrealServer:
    """Loopback TCP server speaking the bridge protocol (one client at a time)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, *, oracle: bool = False,
                 evaluation: bool = False, base_settings: EnvSettings | None = None,
                 idle_timeout: float = 120.0, read_timeout: float = 10.0) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the research server only listens on the loopback interface")
        self.host = "127.0.0.1" if host == "localhost" else host
        self.requested_port = port
        self.session = BridgeSession(oracle=oracle, evaluation=evaluation,
                                     base_settings=base_settings)
        self.idle_timeout = idle_timeout
        self.read_timeout = read_timeout
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._active: socket.socket | None = None
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self.connections = 0
        self.rejected = 0

    @property
    def port(self) -> int:
        assert self._listener is not None, "server not started"
        return int(self._listener.getsockname()[1])

    def start(self) -> MockUnrealServer:
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        listener = socket.socket(family, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.requested_port))
        listener.listen(2)
        listener.settimeout(0.2)
        self._listener = listener
        self._thread = threading.Thread(target=self._accept_loop, name="mock-unreal-server",
                                        daemon=True)
        self._thread.start()
        return self

    def __enter__(self) -> MockUnrealServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self) -> None:
        self._stopping.set()
        with self._lock:
            if self._active is not None:
                try:
                    self._active.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._stopping.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with self._lock:
                busy = self._active is not None
                if not busy:
                    self._active = conn
            if busy:
                self.rejected += 1
                try:
                    conn.sendall(encode_message(error_response(
                        "busy", "another client is connected")))
                finally:
                    conn.close()
                continue
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn,), name="mock-unreal-client",
                             daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def recv_exact(n: int) -> bytes:
            chunks = bytearray()
            while len(chunks) < n:
                chunk = conn.recv(min(n - len(chunks), 1 << 20))
                if not chunk:
                    raise ConnectionError("client closed the connection")
                chunks += chunk
                conn.settimeout(self.read_timeout)      # the rest must follow promptly
            return bytes(chunks)

        greeted = False
        try:
            while not self._stopping.is_set():
                conn.settimeout(self.idle_timeout)
                try:
                    message = read_message(recv_exact)
                except ProtocolError as exc:
                    conn.sendall(encode_message(error_response(exc.code, exc.message)))
                    break
                command = message.header.get("command")
                if not greeted and command not in ("HELLO", "PING"):
                    rid = message.header.get("request_id")
                    conn.sendall(encode_message(error_response(
                        "forbidden", "send HELLO first (protocol version negotiation)",
                        command=command if isinstance(command, str) else None,
                        request_id=rid if isinstance(rid, int) else None)))
                    continue
                header, payload, close = process(self.session, message)
                if command == "HELLO" and header.get("status") == "ok":
                    greeted = True
                conn.sendall(encode_message(header, payload))
                if close:
                    break
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.debug("Client connection ended: %s", exc)
        finally:
            with self._lock:
                self._active = None
            try:
                conn.close()
            except OSError:
                pass
