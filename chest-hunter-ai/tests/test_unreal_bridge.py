"""Bridge client and server over a real loopback socket (mock server = Python reference arena).

These tests exercise the transport and the command semantics that the Unreal plugin mirrors
(ResearchBridgeSubsystem.cpp). They do not involve Unreal Engine.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import numpy as np
import pytest

from chest_hunter.simulation3d.environment3d import EnvSettings
from chest_hunter.unreal_bridge.client import (
    AgentPort,
    BridgeClient,
    BridgeConnectionError,
    BridgeError,
    StaleFrameError,
    require_loopback,
)
from chest_hunter.unreal_bridge.mock_server import MockUnrealServer
from chest_hunter.unreal_bridge.protocol import (
    IDLE_ACTION,
    MAGIC,
    PREFIX,
    PROTOCOL_VERSION,
    encode_message,
    read_message,
    request_header,
)

MAP = "research_arena_3d"
SMALL = EnvSettings(width=320, height=180)


@pytest.fixture
def server() -> Iterator[MockUnrealServer]:
    with MockUnrealServer(base_settings=SMALL) as srv:
        yield srv


@pytest.fixture
def client(server: MockUnrealServer) -> Iterator[BridgeClient]:
    c = BridgeClient(port=server.port, timeout=10.0, connect_attempts=1)
    c.connect()
    yield c
    c.close()


def _raw(server: MockUnrealServer) -> socket.socket:
    s = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    return s


def _recv_exact(s: socket.socket):
    def recv(n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError
            buf += chunk
        return buf
    return recv


def _closed_by_peer(s: socket.socket) -> bool:
    try:
        return s.recv(1) == b""
    except ConnectionResetError:                           # closed with unread input: RST
        return True


def _reset(c: BridgeClient, **kw: object) -> None:
    c.reset_episode(MAP, seed=3, width=320, height=180, **kw)  # type: ignore[arg-type]


class TestConnection:
    def test_hello_negotiates_version_and_features(self, client: BridgeClient) -> None:
        info = client.server_info
        assert info["version"] == PROTOCOL_VERSION
        assert info["features"]["oracle"] is False and info["features"]["evaluation"] is False

    def test_only_loopback_hosts_are_accepted(self) -> None:
        for host in ("127.0.0.1", "localhost", "::1"):
            require_loopback(host)
        for host in ("0.0.0.0", "192.168.1.10", "10.0.0.1", "8.8.8.8"):
            with pytest.raises(ValueError, match="loopback"):
                require_loopback(host)
        with pytest.raises(ValueError):
            BridgeClient(host="192.168.1.10")
        with pytest.raises(ValueError):
            MockUnrealServer(host="0.0.0.0")

    def test_connection_refused_is_reported_after_retries(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        c = BridgeClient(port=port, connect_attempts=2, retry_delay=0.01)
        with pytest.raises(BridgeConnectionError) as exc:
            c.connect()
        assert "--backend reference" in str(exc.value)

    def test_hello_is_required_first(self, server: MockUnrealServer) -> None:
        with _raw(server) as s:
            s.sendall(encode_message(request_header("GET_EPISODE_STATUS", 1, {"episode_id": "x"})))
            reply = read_message(_recv_exact(s))
        assert reply.header["error"]["code"] == "forbidden"

    def test_version_mismatch_in_hello(self, server: MockUnrealServer) -> None:
        with _raw(server) as s:
            s.sendall(encode_message(request_header(
                "HELLO", 1, {"protocol": "chest-hunter-bridge", "version": 99})))
            reply = read_message(_recv_exact(s))
        assert reply.header["error"]["code"] == "version_mismatch"

    def test_framing_version_mismatch_closes_the_connection(self, server: MockUnrealServer) -> None:
        with _raw(server) as s:
            body = b'{"command":"PING","request_id":1}'
            s.sendall(PREFIX.pack(MAGIC, PROTOCOL_VERSION + 1, 0, len(body), 0) + body)
            reply = read_message(_recv_exact(s))
            assert reply.header["error"]["code"] == "version_mismatch"
            assert _closed_by_peer(s)

    def test_garbage_bytes_close_the_connection(self, server: MockUnrealServer) -> None:
        with _raw(server) as s:
            s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            reply = read_message(_recv_exact(s))
            assert reply.header["error"]["code"] == "bad_magic"
            assert _closed_by_peer(s)

    def test_second_client_is_rejected_as_busy(self, server: MockUnrealServer,
                                               client: BridgeClient) -> None:
        with _raw(server) as s:
            reply = read_message(_recv_exact(s))
        assert reply.header["error"]["code"] == "busy"
        assert client.ping() >= 0                          # the first client is unaffected

    def test_unknown_command_keeps_the_connection(self, server: MockUnrealServer) -> None:
        with _raw(server) as s:
            recv = _recv_exact(s)
            s.sendall(encode_message(request_header(
                "HELLO", 1, {"protocol": "chest-hunter-bridge", "version": 1})))
            read_message(recv)
            s.sendall(encode_message(request_header("RUN_SCRIPT", 2, {"code": "import os"})))
            assert read_message(recv).header["error"]["code"] == "unknown_command"
            s.sendall(encode_message(request_header("PING", 3, {})))
            assert read_message(recv).header["status"] == "ok"


class TestEpisodes:
    def test_reset_returns_rules_and_telemetry(self, client: BridgeClient) -> None:
        tel = client.reset_episode(MAP, seed=3, width=320, height=180)
        assert tel.tick == 0 and tel.camera is not None
        assert client.rules["interaction_seconds"] == 2.0

    def test_unknown_map_and_invalid_ranges(self, client: BridgeClient) -> None:
        with pytest.raises(BridgeError) as exc:
            client.reset_episode("not_a_map", seed=1)
        assert exc.value.code == "invalid_argument"
        with pytest.raises(BridgeError) as exc:
            client.reset_episode(MAP, seed=1, width=123, height=45)
        assert exc.value.code == "invalid_argument"

    def test_commands_before_reset(self, client: BridgeClient) -> None:
        with pytest.raises(BridgeError) as exc:
            client.get_observation()
        assert exc.value.code == "no_episode"

    def test_observation_is_a_binary_bgr_frame(self, client: BridgeClient) -> None:
        _reset(client)
        obs = client.get_observation()
        assert obs.frame.shape == (180, 320, 3) and obs.frame.dtype == np.uint8
        assert obs.intrinsics.width == 320 and obs.frame_id == 1

    def test_step_advances_time_and_returns_observation(self, client: BridgeClient) -> None:
        _reset(client)
        res = client.step({**IDLE_ACTION, "move_forward": 1.0}, ticks=10, observe=True)
        assert res.telemetry.tick == 10 and res.observation is not None
        assert res.duplicate is False
        assert res.telemetry.odometry.position[0] > 50 * 0 + 30.0     # moved north from x=30

    def test_invalid_action_is_rejected_by_the_client_before_sending(self,
                                                                     client: BridgeClient) -> None:
        _reset(client)
        sent = client.requests
        with pytest.raises(BridgeError) as exc:
            client.step({**IDLE_ACTION, "move_forward": 5.0})
        assert exc.value.code == "invalid_action" and client.requests == sent

    def test_duplicate_step_is_not_applied_twice(self, server: MockUnrealServer,
                                                 client: BridgeClient) -> None:
        _reset(client)
        client.step({**IDLE_ACTION, "move_forward": 1.0}, ticks=5)
        tick = server.session.env.tick if server.session.env else -1
        args = {"episode_id": client.episode_id, "step_id": 1, "action": IDLE_ACTION,
                "ticks": 5, "observe": False}
        reply = client._request("STEP", args)                # resend of step 1
        assert reply.header["duplicate"] is True
        assert server.session.env is not None and server.session.env.tick == tick

    def test_stale_and_out_of_order_steps(self, client: BridgeClient) -> None:
        _reset(client)
        client.step(IDLE_ACTION)
        client.step(IDLE_ACTION)
        base = {"episode_id": client.episode_id, "action": IDLE_ACTION}
        with pytest.raises(BridgeError) as exc:
            client._request("STEP", {**base, "step_id": 1})
        assert exc.value.code == "stale_step"
        with pytest.raises(BridgeError) as exc:
            client._request("STEP", {**base, "step_id": 7})
        assert exc.value.code == "out_of_order"

    def test_stale_episode_id(self, client: BridgeClient) -> None:
        _reset(client)
        old = client.episode_id
        _reset(client)
        with pytest.raises(BridgeError) as exc:
            client._request("GET_OBSERVATION", {"episode_id": old})
        assert exc.value.code == "stale_episode"

    def test_stale_frames_are_detected(self, client: BridgeClient) -> None:
        _reset(client)
        client.get_observation()
        client.last_frame_id = 99                            # pretend a newer frame was seen
        with pytest.raises(StaleFrameError):
            client.get_observation()

    def test_reconnect_resends_step_and_episode_survives(self, server: MockUnrealServer,
                                                         client: BridgeClient) -> None:
        _reset(client)
        client.step(IDLE_ACTION, ticks=2)
        assert client._sock is not None
        client._sock.close()                                 # simulate a dropped connection
        client._sock = socket.socket()                       # unusable socket -> send fails
        res = client.step({**IDLE_ACTION, "move_forward": 1.0}, ticks=3)
        assert client.resends == 1 and res.step_id == 2 and res.telemetry.tick == 5

    def test_episode_end_and_stop(self, client: BridgeClient) -> None:
        _reset(client, max_ticks=4)
        res = client.step(IDLE_ACTION, ticks=10)
        assert res.ended and res.telemetry.tick == 4
        with pytest.raises(BridgeError) as exc:
            client.step(IDLE_ACTION)
        assert exc.value.code == "episode_ended"
        _reset(client)
        status = client.stop_episode("test")
        assert status["state"] == "stopped" and "summary" in status
        assert client.episode_status()["state"] == "stopped"

    def test_navigation_modes(self, client: BridgeClient) -> None:
        _reset(client)
        path = client.nav_query((30.0, 50.0, 0.0), (40.0, 50.0, 0.0))
        assert path.found and path.length > 0
        assert len(client.nav_raycast((30.0, 50.0, 0.0), [0.0, 90.0])) == 2
        _reset(client, navigation="local")
        with pytest.raises(BridgeError) as exc:
            client.nav_query((30.0, 50.0, 0.0), (40.0, 50.0, 0.0))
        assert exc.value.code == "navigation_disabled"
        assert client.nav_raycast((30.0, 50.0, 0.0), [0.0])
        _reset(client, navigation="none")
        with pytest.raises(BridgeError):
            client.nav_raycast((30.0, 50.0, 0.0), [0.0])


class TestInformationBoundary:
    def test_privileged_commands_are_refused_by_default(self, client: BridgeClient) -> None:
        _reset(client)
        with pytest.raises(BridgeError) as exc:
            client.oracle_state()
        assert exc.value.code == "forbidden"
        frame = client.get_observation().frame_id
        with pytest.raises(BridgeError) as exc:
            client.evaluation_labels(frame)
        assert exc.value.code == "forbidden"

    def test_oracle_mode_allows_truth_but_not_labels(self) -> None:
        with MockUnrealServer(oracle=True, base_settings=SMALL) as srv, \
                BridgeClient(port=srv.port, connect_attempts=1) as c:
            _reset(c)
            state = c.oracle_state()
            assert {"present", "opened", "position"} <= set(state["chests"][0])
            with pytest.raises(BridgeError):
                c.evaluation_labels(c.get_observation().frame_id)

    def test_evaluation_mode_labels_latest_frame_only(self) -> None:
        with MockUnrealServer(evaluation=True, base_settings=SMALL) as srv, \
                BridgeClient(port=srv.port, connect_attempts=1) as c:
            _reset(c)
            f1 = c.get_observation().frame_id
            labels = c.evaluation_labels(f1)
            assert labels["frame_id"] == f1
            c.get_observation()
            with pytest.raises(BridgeError):
                c.evaluation_labels(f1)

    def test_agent_port_has_no_privileged_surface(self, client: BridgeClient) -> None:
        port = AgentPort(client)
        public = {n for n in dir(port) if not n.startswith("_")}
        assert public == {"rules", "observe", "step", "nav_query", "nav_raycast"}
        with pytest.raises(AttributeError):
            port.client = client                            # type: ignore[attr-defined]

    def test_agent_facing_messages_carry_no_chest_truth(self, client: BridgeClient) -> None:
        _reset(client)
        obs = client.get_observation()
        res = client.step(IDLE_ACTION, observe=True)
        texts = [str(client.server_info), str(client.rules), str(res.telemetry), str(obs.camera),
                 str(obs.odometry)]
        for text in texts:
            for forbidden in ("present", "spawn_index", "chests", "oracle_state"):
                assert forbidden not in text

    def test_odometry_is_not_the_true_position(self) -> None:
        settings = EnvSettings(width=320, height=180, odometry_drift=0.05)
        with MockUnrealServer(oracle=True, base_settings=settings) as srv, \
                BridgeClient(port=srv.port, connect_attempts=1) as c:
            _reset(c)
            res = c.step({**IDLE_ACTION, "move_forward": 1.0, "look_yaw": 0.2}, ticks=30)
            for _ in range(5):
                res = c.step({**IDLE_ACTION, "move_forward": 1.0}, ticks=30)
            true = c.oracle_state()["agent"]["position"]
            est = res.telemetry.odometry.position
            assert est != pytest.approx(true, abs=1e-6)
            assert abs(est[0] - true[0]) < 2.0 and abs(est[1] - true[1]) < 2.0


def test_server_survives_a_handler_bug(server: MockUnrealServer, client: BridgeClient,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_request: object) -> None:
        raise RuntimeError("bug")
    monkeypatch.setattr(server.session, "_cmd_ping", boom)
    with pytest.raises(BridgeError) as exc:
        client.ping()
    assert exc.value.code == "internal"
    monkeypatch.undo()
    assert client.ping() >= 0


def test_concurrent_clients_are_serialised(server: MockUnrealServer) -> None:
    results: list[str] = []

    def run() -> None:
        c = BridgeClient(port=server.port, connect_attempts=1, retry_delay=0.01)
        try:
            c.connect()
            results.append("connected")
            c.close()
        except BridgeError as exc:
            results.append(exc.code)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert "connected" in results
    assert set(results) <= {"connected", "connect_failed"}     # "busy" after its one attempt


def test_connect_retries_while_the_server_is_busy(server: MockUnrealServer,
                                                  client: BridgeClient) -> None:
    timer = threading.Timer(0.15, client.close)                # the first client leaves soon
    timer.start()
    second = BridgeClient(port=server.port, connect_attempts=6, retry_delay=0.05)
    try:
        assert second.connect()["status"] == "ok"
    finally:
        timer.join()
        second.close()
