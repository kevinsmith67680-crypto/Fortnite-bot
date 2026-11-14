"""Bridge client and server over a real loopback socket (mock server = Python reference arena).

These tests exercise the transport and the command semantics that the Unreal plugin mirrors
(ResearchBridgeSubsystem.cpp). They do not involve Unreal Engine.
"""

from __future__ import annotations

import socket
import threading
import unittest
from unittest import mock

import numpy as np

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


class BridgeTestCase(unittest.TestCase):
    """A mock server and a connected client, started per test on request.

    A test starts only what it uses: some need a server with no other client connected (the
    server accepts one client at a time), some need neither. Everything started is stopped
    when the test ends, the client before its server.
    """

    def start_server(self) -> MockUnrealServer:
        return self.enterContext(MockUnrealServer(base_settings=SMALL))

    def connect_client(self, server: MockUnrealServer) -> BridgeClient:
        c = BridgeClient(port=server.port, timeout=10.0, connect_attempts=1)
        c.connect()
        self.addCleanup(c.close)
        return c


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


class TestConnection(BridgeTestCase):
    def test_hello_negotiates_version_and_features(self) -> None:
        client = self.connect_client(self.start_server())
        info = client.server_info
        self.assertEqual(info["version"], PROTOCOL_VERSION)
        self.assertIs(info["features"]["oracle"], False)
        self.assertIs(info["features"]["evaluation"], False)

    def test_only_loopback_hosts_are_accepted(self) -> None:
        for host in ("127.0.0.1", "localhost", "::1"):
            require_loopback(host)
        for host in ("0.0.0.0", "192.168.1.10", "10.0.0.1", "8.8.8.8"):
            with self.assertRaisesRegex(ValueError, "loopback"):
                require_loopback(host)
        with self.assertRaises(ValueError):
            BridgeClient(host="192.168.1.10")
        with self.assertRaises(ValueError):
            MockUnrealServer(host="0.0.0.0")

    def test_connection_refused_is_reported_after_retries(self) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        c = BridgeClient(port=port, connect_attempts=2, retry_delay=0.01)
        with self.assertRaises(BridgeConnectionError) as cm:
            c.connect()
        self.assertIn("--backend reference", str(cm.exception))

    def test_hello_is_required_first(self) -> None:
        server = self.start_server()
        with _raw(server) as s:
            s.sendall(encode_message(request_header("GET_EPISODE_STATUS", 1, {"episode_id": "x"})))
            reply = read_message(_recv_exact(s))
        self.assertEqual(reply.header["error"]["code"], "forbidden")

    def test_version_mismatch_in_hello(self) -> None:
        server = self.start_server()
        with _raw(server) as s:
            s.sendall(encode_message(request_header(
                "HELLO", 1, {"protocol": "chest-hunter-bridge", "version": 99})))
            reply = read_message(_recv_exact(s))
        self.assertEqual(reply.header["error"]["code"], "version_mismatch")

    def test_framing_version_mismatch_closes_the_connection(self) -> None:
        server = self.start_server()
        with _raw(server) as s:
            body = b'{"command":"PING","request_id":1}'
            s.sendall(PREFIX.pack(MAGIC, PROTOCOL_VERSION + 1, 0, len(body), 0) + body)
            reply = read_message(_recv_exact(s))
            self.assertEqual(reply.header["error"]["code"], "version_mismatch")
            self.assertTrue(_closed_by_peer(s))

    def test_garbage_bytes_close_the_connection(self) -> None:
        server = self.start_server()
        with _raw(server) as s:
            s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            reply = read_message(_recv_exact(s))
            self.assertEqual(reply.header["error"]["code"], "bad_magic")
            self.assertTrue(_closed_by_peer(s))

    def test_second_client_is_rejected_as_busy(self) -> None:
        server = self.start_server()
        client = self.connect_client(server)
        with _raw(server) as s:
            reply = read_message(_recv_exact(s))
        self.assertEqual(reply.header["error"]["code"], "busy")
        self.assertGreaterEqual(client.ping(), 0)          # the first client is unaffected

    def test_unknown_command_keeps_the_connection(self) -> None:
        server = self.start_server()
        with _raw(server) as s:
            recv = _recv_exact(s)
            s.sendall(encode_message(request_header(
                "HELLO", 1, {"protocol": "chest-hunter-bridge", "version": 1})))
            read_message(recv)
            s.sendall(encode_message(request_header("RUN_SCRIPT", 2, {"code": "import os"})))
            self.assertEqual(read_message(recv).header["error"]["code"], "unknown_command")
            s.sendall(encode_message(request_header("PING", 3, {})))
            self.assertEqual(read_message(recv).header["status"], "ok")


class TestEpisodes(BridgeTestCase):
    def setUp(self) -> None:
        self.server = self.start_server()
        self.client = self.connect_client(self.server)

    def test_reset_returns_rules_and_telemetry(self) -> None:
        client = self.client
        tel = client.reset_episode(MAP, seed=3, width=320, height=180)
        self.assertEqual(tel.tick, 0)
        self.assertIsNotNone(tel.camera)
        self.assertEqual(client.rules["interaction_seconds"], 2.0)

    def test_unknown_map_and_invalid_ranges(self) -> None:
        client = self.client
        with self.assertRaises(BridgeError) as cm:
            client.reset_episode("not_a_map", seed=1)
        self.assertEqual(cm.exception.code, "invalid_argument")
        with self.assertRaises(BridgeError) as cm:
            client.reset_episode(MAP, seed=1, width=123, height=45)
        self.assertEqual(cm.exception.code, "invalid_argument")

    def test_commands_before_reset(self) -> None:
        client = self.client
        with self.assertRaises(BridgeError) as cm:
            client.get_observation()
        self.assertEqual(cm.exception.code, "no_episode")

    def test_observation_is_a_binary_bgr_frame(self) -> None:
        client = self.client
        _reset(client)
        obs = client.get_observation()
        self.assertEqual(obs.frame.shape, (180, 320, 3))
        self.assertEqual(obs.frame.dtype, np.uint8)
        self.assertEqual(obs.intrinsics.width, 320)
        self.assertEqual(obs.frame_id, 1)

    def test_step_advances_time_and_returns_observation(self) -> None:
        client = self.client
        _reset(client)
        res = client.step({**IDLE_ACTION, "move_forward": 1.0}, ticks=10, observe=True)
        self.assertEqual(res.telemetry.tick, 10)
        self.assertIsNotNone(res.observation)
        self.assertIs(res.duplicate, False)
        # Moved north from x=30.
        self.assertGreater(res.telemetry.odometry.position[0], 50 * 0 + 30.0)

    def test_invalid_action_is_rejected_by_the_client_before_sending(self) -> None:
        client = self.client
        _reset(client)
        sent = client.requests
        with self.assertRaises(BridgeError) as cm:
            client.step({**IDLE_ACTION, "move_forward": 5.0})
        self.assertEqual(cm.exception.code, "invalid_action")
        self.assertEqual(client.requests, sent)

    def test_duplicate_step_is_not_applied_twice(self) -> None:
        server, client = self.server, self.client
        _reset(client)
        client.step({**IDLE_ACTION, "move_forward": 1.0}, ticks=5)
        tick = server.session.env.tick if server.session.env else -1
        args = {"episode_id": client.episode_id, "step_id": 1, "action": IDLE_ACTION,
                "ticks": 5, "observe": False}
        reply = client._request("STEP", args)                # resend of step 1
        self.assertIs(reply.header["duplicate"], True)
        self.assertIsNotNone(server.session.env)
        self.assertEqual(server.session.env.tick, tick)  # type: ignore[union-attr]

    def test_stale_and_out_of_order_steps(self) -> None:
        client = self.client
        _reset(client)
        client.step(IDLE_ACTION)
        client.step(IDLE_ACTION)
        base = {"episode_id": client.episode_id, "action": IDLE_ACTION}
        with self.assertRaises(BridgeError) as cm:
            client._request("STEP", {**base, "step_id": 1})
        self.assertEqual(cm.exception.code, "stale_step")
        with self.assertRaises(BridgeError) as cm:
            client._request("STEP", {**base, "step_id": 7})
        self.assertEqual(cm.exception.code, "out_of_order")

    def test_stale_episode_id(self) -> None:
        client = self.client
        _reset(client)
        old = client.episode_id
        _reset(client)
        with self.assertRaises(BridgeError) as cm:
            client._request("GET_OBSERVATION", {"episode_id": old})
        self.assertEqual(cm.exception.code, "stale_episode")

    def test_stale_frames_are_detected(self) -> None:
        client = self.client
        _reset(client)
        client.get_observation()
        client.last_frame_id = 99                            # pretend a newer frame was seen
        with self.assertRaises(StaleFrameError):
            client.get_observation()

    def test_reconnect_resends_step_and_episode_survives(self) -> None:
        client = self.client
        _reset(client)
        client.step(IDLE_ACTION, ticks=2)
        self.assertIsNotNone(client._sock)
        # Simulate a dropped connection.
        client._sock.close()  # type: ignore[union-attr]
        client._sock = socket.socket()                       # unusable socket -> send fails
        res = client.step({**IDLE_ACTION, "move_forward": 1.0}, ticks=3)
        self.assertEqual(client.resends, 1)
        self.assertEqual(res.step_id, 2)
        self.assertEqual(res.telemetry.tick, 5)

    def test_episode_end_and_stop(self) -> None:
        client = self.client
        _reset(client, max_ticks=4)
        res = client.step(IDLE_ACTION, ticks=10)
        self.assertTrue(res.ended)
        self.assertEqual(res.telemetry.tick, 4)
        with self.assertRaises(BridgeError) as cm:
            client.step(IDLE_ACTION)
        self.assertEqual(cm.exception.code, "episode_ended")
        _reset(client)
        status = client.stop_episode("test")
        self.assertEqual(status["state"], "stopped")
        self.assertIn("summary", status)
        self.assertEqual(client.episode_status()["state"], "stopped")

    def test_navigation_modes(self) -> None:
        client = self.client
        _reset(client)
        path = client.nav_query((30.0, 50.0, 0.0), (40.0, 50.0, 0.0))
        self.assertTrue(path.found)
        self.assertGreater(path.length, 0)
        self.assertEqual(len(client.nav_raycast((30.0, 50.0, 0.0), [0.0, 90.0])), 2)
        _reset(client, navigation="local")
        with self.assertRaises(BridgeError) as cm:
            client.nav_query((30.0, 50.0, 0.0), (40.0, 50.0, 0.0))
        self.assertEqual(cm.exception.code, "navigation_disabled")
        self.assertTrue(client.nav_raycast((30.0, 50.0, 0.0), [0.0]))
        _reset(client, navigation="none")
        with self.assertRaises(BridgeError):
            client.nav_raycast((30.0, 50.0, 0.0), [0.0])


class TestInformationBoundary(BridgeTestCase):
    def test_privileged_commands_are_refused_by_default(self) -> None:
        client = self.connect_client(self.start_server())
        _reset(client)
        with self.assertRaises(BridgeError) as cm:
            client.oracle_state()
        self.assertEqual(cm.exception.code, "forbidden")
        frame = client.get_observation().frame_id
        with self.assertRaises(BridgeError) as cm:
            client.evaluation_labels(frame)
        self.assertEqual(cm.exception.code, "forbidden")

    def test_oracle_mode_allows_truth_but_not_labels(self) -> None:
        with MockUnrealServer(oracle=True, base_settings=SMALL) as srv, \
                BridgeClient(port=srv.port, connect_attempts=1) as c:
            _reset(c)
            state = c.oracle_state()
            self.assertLessEqual({"present", "opened", "position"}, set(state["chests"][0]))
            with self.assertRaises(BridgeError):
                c.evaluation_labels(c.get_observation().frame_id)

    def test_evaluation_mode_labels_latest_frame_only(self) -> None:
        with MockUnrealServer(evaluation=True, base_settings=SMALL) as srv, \
                BridgeClient(port=srv.port, connect_attempts=1) as c:
            _reset(c)
            f1 = c.get_observation().frame_id
            labels = c.evaluation_labels(f1)
            self.assertEqual(labels["frame_id"], f1)
            c.get_observation()
            with self.assertRaises(BridgeError):
                c.evaluation_labels(f1)

    def test_agent_port_has_no_privileged_surface(self) -> None:
        client = self.connect_client(self.start_server())
        port = AgentPort(client)
        public = {n for n in dir(port) if not n.startswith("_")}
        self.assertEqual(public, {"rules", "observe", "step", "nav_query", "nav_raycast"})
        with self.assertRaises(AttributeError):
            port.client = client                            # type: ignore[attr-defined]

    def test_agent_facing_messages_carry_no_chest_truth(self) -> None:
        client = self.connect_client(self.start_server())
        _reset(client)
        obs = client.get_observation()
        res = client.step(IDLE_ACTION, observe=True)
        texts = [str(client.server_info), str(client.rules), str(res.telemetry), str(obs.camera),
                 str(obs.odometry)]
        for text in texts:
            for forbidden in ("present", "spawn_index", "chests", "oracle_state"):
                self.assertNotIn(forbidden, text)

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
            # The estimate is not the truth: some coordinate differs by more than 1e-6.
            self.assertFalse(len(est) == len(true) and all(
                abs(e - t) <= 1e-6 for e, t in zip(est, true, strict=True)))
            self.assertLess(abs(est[0] - true[0]), 2.0)
            self.assertLess(abs(est[1] - true[1]), 2.0)


class TestServerRobustness(BridgeTestCase):
    def test_server_survives_a_handler_bug(self) -> None:
        server = self.start_server()
        client = self.connect_client(server)

        def boom(_request: object) -> None:
            raise RuntimeError("bug")
        with mock.patch.object(server.session, "_cmd_ping", boom):
            with self.assertRaises(BridgeError) as cm:
                client.ping()
            self.assertEqual(cm.exception.code, "internal")
        self.assertGreaterEqual(client.ping(), 0)

    def test_concurrent_clients_are_serialised(self) -> None:
        server = self.start_server()
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
        self.assertIn("connected", results)
        # "busy" after its one attempt
        self.assertLessEqual(set(results), {"connected", "connect_failed"})

    def test_connect_retries_while_the_server_is_busy(self) -> None:
        server = self.start_server()
        client = self.connect_client(server)
        timer = threading.Timer(0.15, client.close)                # the first client leaves soon
        timer.start()
        second = BridgeClient(port=server.port, connect_attempts=6, retry_delay=0.05)
        try:
            self.assertEqual(second.connect()["status"], "ok")
        finally:
            timer.join()
            second.close()
