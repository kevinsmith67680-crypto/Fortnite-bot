"""Localhost-only client for the research bridge (Unreal plugin or mock server).

Reliability rules:

* Connections are only made to loopback addresses; anything else is refused
  before a socket is opened.
* ``connect`` retries a bounded number of times (the editor may still be
  loading), then fails with :class:`BridgeConnectionError`.
* Every request has a timeout. A ``STEP`` interrupted by a timeout or a
  dropped connection is re-sent once *with the same step id*: the server
  recognises the duplicate and returns the stored result instead of applying
  the action twice.
* Observations must arrive with strictly increasing frame ids; an older frame
  raises :class:`StaleFrameError`.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import time
from dataclasses import dataclass
from typing import Any

from chest_hunter.unreal_bridge.observation_adapter import (
    Observation3D,
    Telemetry,
    decode_observation,
    decode_telemetry,
)
from chest_hunter.unreal_bridge.protocol import (
    DEFAULT_PORT,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    Message,
    ProtocolError,
    encode_message,
    read_message,
    request_header,
    validate_action,
)

logger = logging.getLogger(__name__)


class BridgeError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class BridgeConnectionError(BridgeError):
    pass


class BridgeTimeout(BridgeError):
    pass


class StaleFrameError(BridgeError):
    pass


def require_loopback(host: str) -> None:
    """Refuse any host that does not resolve exclusively to loopback addresses."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve {host!r}: {exc}") from exc
    addresses = {info[4][0] for info in infos}
    if not addresses or not all(ipaddress.ip_address(a.split("%")[0]).is_loopback
                                for a in addresses):
        raise ValueError(f"the research bridge only connects to this machine (loopback); "
                         f"{host!r} resolves to {sorted(addresses)}")


@dataclass(frozen=True, slots=True)
class StepResult:
    step_id: int
    telemetry: Telemetry
    observation: Observation3D | None
    ended: bool
    duplicate: bool


@dataclass(frozen=True, slots=True)
class NavResult:
    status: str
    waypoints: tuple[tuple[float, float, float], ...]
    length: float

    @property
    def found(self) -> bool:
        return self.status == "ok"


class BridgeClient:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, *,
                 timeout: float = 10.0, connect_attempts: int = 5,
                 retry_delay: float = 0.5) -> None:
        require_loopback(host)
        self.host, self.port = host, port
        self.timeout = timeout
        self.connect_attempts = max(1, connect_attempts)
        self.retry_delay = retry_delay
        self._sock: socket.socket | None = None
        self._request_id = 0
        self.server_info: dict[str, Any] = {}
        self.episode_id: str | None = None
        self.rules: dict[str, Any] = {}
        self.step_id = 0
        self.last_frame_id = 0
        self.requests = 0
        self.resends = 0
        self.bytes_received = 0
        self.round_trip_seconds = 0.0

    # -- connection ----------------------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.connect_attempts):
            try:
                sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self.timeout)
                self._sock = sock
                break
            except OSError as exc:
                last = exc
                time.sleep(self.retry_delay * (2 ** attempt))
        else:
            raise BridgeConnectionError(
                "connect_failed", f"no research server on {self.host}:{self.port} after "
                                  f"{self.connect_attempts} attempt(s): {last}")
        reply = self._request("HELLO", {"protocol": PROTOCOL_NAME, "version": PROTOCOL_VERSION,
                                        "client": "chest_hunter"})
        self.server_info = reply.header
        return self.server_info

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def __enter__(self) -> BridgeClient:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                raise ConnectionError("server closed the connection")
            buf += chunk
        self.bytes_received += n
        return bytes(buf)

    def _request(self, command: str, args: dict[str, Any] | None = None) -> Message:
        if self._sock is None:
            raise BridgeConnectionError("not_connected", "call connect() first")
        self._request_id += 1
        rid = self._request_id
        try:
            data = encode_message(request_header(command, rid, args))
        except ProtocolError as exc:                   # bad input: the connection stays usable
            raise BridgeError(exc.code, exc.message) from exc
        started = time.perf_counter()
        try:
            self._sock.sendall(data)
            reply = read_message(self._recv_exact)
        except TimeoutError as exc:
            self.close()
            raise BridgeTimeout("timeout", f"{command} got no answer within {self.timeout} s") \
                from exc
        except (ConnectionError, OSError) as exc:
            self.close()
            raise BridgeConnectionError("disconnected", f"{command} failed: {exc}") from exc
        except ProtocolError as exc:
            self.close()
            raise BridgeError(exc.code, exc.message) from exc
        self.requests += 1
        self.round_trip_seconds += time.perf_counter() - started
        header = reply.header
        if header.get("status") != "ok":
            err = header.get("error") or {}
            raise BridgeError(str(err.get("code", "internal")), str(err.get("message", "")))
        if header.get("request_id") != rid:
            self.close()
            raise BridgeError("bad_header", f"response for request {header.get('request_id')}, "
                                            f"expected {rid}")
        return reply

    def _reconnect(self) -> None:
        self.close()
        self.connect()

    # -- episode API -------------------------------------------------------------------------

    def reset_episode(self, map_id: str, *, seed: int, episode: int = 0, width: int = 640,
                      height: int = 360, fov_deg: float = 90.0, graphics_quality: str = "low",
                      max_ticks: int | None = None, navigation: str = "navmesh") -> Telemetry:
        args: dict[str, Any] = {"map_id": map_id, "seed": seed, "episode": episode,
                                "observation": {"width": width, "height": height},
                                "fov_deg": fov_deg, "graphics_quality": graphics_quality,
                                "navigation": navigation}
        if max_ticks is not None:
            args["max_ticks"] = max_ticks
        reply = self._request("RESET_EPISODE", args)
        self.episode_id = str(reply.header["episode_id"])
        self.rules = dict(reply.header.get("rules", {}))
        self.step_id = 0
        self.last_frame_id = 0
        return decode_telemetry(reply.header["telemetry"])

    def _episode(self) -> str:
        if self.episode_id is None:
            raise BridgeError("no_episode", "call reset_episode() first")
        return self.episode_id

    def _observation(self, meta: dict[str, Any], payload: bytes) -> Observation3D:
        obs = decode_observation(meta, payload, self._episode())
        if obs.frame_id <= self.last_frame_id:
            raise StaleFrameError("stale_frame", f"frame {obs.frame_id} is not newer than "
                                                 f"{self.last_frame_id}")
        self.last_frame_id = obs.frame_id
        return obs

    def get_observation(self) -> Observation3D:
        reply = self._request("GET_OBSERVATION", {"episode_id": self._episode()})
        return self._observation(reply.header["observation"], reply.payload)

    def step(self, action: dict[str, Any], *, ticks: int = 1, observe: bool = False) -> StepResult:
        try:
            action = validate_action(action)            # same rules as the server
        except ProtocolError as exc:
            raise BridgeError(exc.code, exc.message) from exc
        step_id = self.step_id + 1
        args = {"episode_id": self._episode(), "step_id": step_id, "action": action,
                "ticks": ticks, "observe": observe}
        try:
            reply = self._request("STEP", args)
        except (BridgeTimeout, BridgeConnectionError) as exc:
            logger.warning("STEP %d interrupted (%s); reconnecting and re-sending", step_id, exc)
            self.resends += 1
            self._reconnect()
            reply = self._request("STEP", args)          # same step id: applied at most once
        self.step_id = step_id
        h = reply.header
        obs = self._observation(h["observation"], reply.payload) if "observation" in h else None
        return StepResult(step_id, decode_telemetry(h["telemetry"]), obs, bool(h.get("ended")),
                          bool(h.get("duplicate")))

    def episode_status(self) -> dict[str, Any]:
        return dict(self._request("GET_EPISODE_STATUS", {"episode_id": self._episode()}).header)

    def stop_episode(self, reason: str = "agent finished") -> dict[str, Any]:
        return dict(self._request("STOP_EPISODE", {"episode_id": self._episode(),
                                                   "reason": reason}).header)

    def ping(self, nonce: int = 0) -> float:
        started = time.perf_counter()
        self._request("PING", {"nonce": nonce})
        return time.perf_counter() - started

    def nav_query(self, start: tuple[float, float, float], goal: tuple[float, float, float], *,
                  goal_tolerance: float = 0.5) -> NavResult:
        reply = self._request("NAV_QUERY", {"episode_id": self._episode(), "start": list(start),
                                            "goal": list(goal), "goal_tolerance": goal_tolerance})
        h = reply.header
        return NavResult(str(h["status_code"]),
                         tuple((float(w[0]), float(w[1]), float(w[2])) for w in h["waypoints"]),
                         float(h["length"]))

    def nav_raycast(self, origin: tuple[float, float, float], bearings_deg: list[float], *,
                    max_distance: float = 8.0) -> list[float]:
        reply = self._request("NAV_RAYCAST", {"episode_id": self._episode(),
                                              "origin": list(origin),
                                              "bearings_deg": list(bearings_deg),
                                              "max_distance": max_distance})
        return [float(d) for d in reply.header["distances"]]

    # -- privileged (benchmark / evaluation only; refused by servers not started for it) ----

    def oracle_state(self) -> dict[str, Any]:
        return dict(self._request("GET_ORACLE_STATE", {"episode_id": self._episode()})
                    .header["state"])

    def evaluation_labels(self, frame_id: int) -> dict[str, Any]:
        return dict(self._request("GET_EVALUATION", {"episode_id": self._episode(),
                                                     "frame_id": frame_id}).header["labels"])


class AgentPort:
    """The only bridge surface the visual policy is given.

    It exposes observations, actions and the navigation service. It has no
    method that could request oracle state or evaluation labels.
    """

    __slots__ = ("__client",)

    def __init__(self, client: BridgeClient) -> None:
        self.__client = client

    @property
    def rules(self) -> dict[str, Any]:
        return dict(self.__client.rules)

    def observe(self) -> Observation3D:
        return self.__client.get_observation()

    def step(self, action: dict[str, Any], *, ticks: int = 1, observe: bool = False) -> StepResult:
        return self.__client.step(action, ticks=ticks, observe=observe)

    def nav_query(self, start: tuple[float, float, float], goal: tuple[float, float, float], *,
                  goal_tolerance: float = 0.5) -> NavResult:
        return self.__client.nav_query(start, goal, goal_tolerance=goal_tolerance)

    def nav_raycast(self, origin: tuple[float, float, float], bearings_deg: list[float], *,
                    max_distance: float = 8.0) -> list[float]:
        return self.__client.nav_raycast(origin, bearings_deg, max_distance=max_distance)
