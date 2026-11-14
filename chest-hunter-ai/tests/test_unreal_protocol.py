"""Wire protocol: framing, size limits, versioning and request/action validation."""

from __future__ import annotations

import io
import struct
import unittest

from chest_hunter.unreal_bridge.protocol import (
    COMMANDS,
    ERROR_CODES,
    IDLE_ACTION,
    MAGIC,
    MAX_HEADER_BYTES,
    MAX_PAYLOAD_BYTES,
    PREFIX,
    PRIVILEGED_COMMANDS,
    PROTOCOL_VERSION,
    ProtocolError,
    encode_message,
    error_response,
    parse_header,
    parse_prefix,
    read_message,
    request_header,
    validate_action,
    validate_request,
)


def _reader(data: bytes):
    stream = io.BytesIO(data)

    def recv_exact(n: int) -> bytes:
        chunk = stream.read(n)
        if len(chunk) != n:
            raise ConnectionError("short read")
        return chunk
    return recv_exact


def _action(**overrides):
    return {**IDLE_ACTION, **overrides}


class TestFraming(unittest.TestCase):
    def test_round_trip_with_binary_payload(self) -> None:
        payload = bytes(range(256)) * 10
        data = encode_message({"command": "PING", "n": 1}, payload)
        message = read_message(_reader(data))
        self.assertEqual(message.header, {"command": "PING", "n": 1})
        self.assertEqual(message.payload, payload)

    def test_prefix_layout_matches_the_cpp_test_vector(self) -> None:
        data = encode_message({"command": "PING", "request_id": 1, "args": {}})
        magic, version, flags, hlen, plen = PREFIX.unpack(data[:16])
        self.assertEqual((magic, version, flags, plen), (MAGIC, PROTOCOL_VERSION, 0, 0))
        self.assertEqual(hlen, len(data) - 16)
        # Same bytes as asserted by ResearchProtocolTests.cpp (magic, version 1).
        self.assertEqual(data[:6].hex(), "434852420001")

    def test_images_are_raw_bytes_not_json(self) -> None:
        payload = b"\x00\x01\x02" * 1000
        data = encode_message({"encoding": "bgr8"}, payload)
        header_len = PREFIX.unpack(data[:16])[3]
        self.assertEqual(len(data), 16 + header_len + len(payload))
        self.assertNotIn(b"[", data[16:16 + header_len].replace(b"[]", b""))

    def test_bad_magic(self) -> None:
        data = b"XXXX" + encode_message({"a": 1})[4:]
        with self.assertRaises(ProtocolError) as cm:
            parse_prefix(data[:16])
        self.assertEqual(cm.exception.code, "bad_magic")

    def test_version_mismatch(self) -> None:
        prefix = PREFIX.pack(MAGIC, PROTOCOL_VERSION + 1, 0, 2, 0)
        with self.assertRaises(ProtocolError) as cm:
            parse_prefix(prefix)
        self.assertEqual(cm.exception.code, "version_mismatch")

    def test_size_limits(self) -> None:
        for hlen, plen, code in [
            (MAX_HEADER_BYTES + 1, 0, "too_large"),
            (1, 0, "bad_header"),
            (10, MAX_PAYLOAD_BYTES + 1, "too_large"),
        ]:
            with self.subTest(hlen=hlen, plen=plen, code=code):
                with self.assertRaises(ProtocolError) as cm:
                    parse_prefix(PREFIX.pack(MAGIC, PROTOCOL_VERSION, 0, hlen, plen))
                self.assertEqual(cm.exception.code, code)

    def test_encoder_refuses_oversized_payload_and_nan(self) -> None:
        with self.assertRaises(ProtocolError) as cm:
            encode_message({}, b"\0" * (MAX_PAYLOAD_BYTES + 1))
        self.assertEqual(cm.exception.code, "too_large")
        with self.assertRaises(ProtocolError) as cm:
            encode_message({"x": float("nan")})
        self.assertEqual(cm.exception.code, "bad_header")

    def test_truncated_prefix(self) -> None:
        with self.assertRaises(ProtocolError) as cm:
            parse_prefix(b"CHRB")
        self.assertEqual(cm.exception.code, "bad_header")

    def test_bad_headers(self) -> None:
        for raw in [b"not json", b"[1, 2]", b'{"x": NaN}', b"\xff\xfe"]:
            with self.subTest(raw=raw):
                with self.assertRaises(ProtocolError) as cm:
                    parse_header(raw)
                self.assertIn(cm.exception.code, {"bad_json", "bad_header"})

    def test_truncated_stream(self) -> None:
        data = encode_message({"a": 1}, b"abc")
        with self.assertRaises(ConnectionError):
            read_message(_reader(data[:-1]))


class TestActions(unittest.TestCase):
    def test_valid_action_is_normalised(self) -> None:
        out = validate_action(_action(move_forward=1, look_yaw=-0.5))
        self.assertEqual(out["move_forward"], 1.0)
        self.assertIsInstance(out["move_forward"], float)
        self.assertIs(out["sprint"], False)
        self.assertIs(out["crouch"], False)

    def test_invalid_actions(self) -> None:
        for bad in [
            _action(move_forward=1.5), _action(look_pitch=-1.01), _action(move_right=float("inf")),
            _action(jump=1), _action(interact="yes"), _action(move_forward=True),
            {**_action(), "teleport": True},
            {k: v for k, v in IDLE_ACTION.items() if k != "interact"},
            "forward", None,
        ]:
            with self.subTest(bad=bad):
                with self.assertRaises(ProtocolError) as cm:
                    validate_action(bad)
                self.assertEqual(cm.exception.code, "invalid_action")

    def test_no_action_key_can_address_hidden_state_or_scripts(self) -> None:
        with self.assertRaises(ProtocolError):
            validate_action({**_action(), "exec": "quit"})
        with self.assertRaises(ProtocolError):
            validate_action({**_action(), "set_position": [1, 2, 3]})


class TestRequests(unittest.TestCase):
    def test_every_command_has_an_argument_specification(self) -> None:
        for command in COMMANDS:
            args = {"episode_id": "e"} if command not in {"HELLO", "RESET_EPISODE", "PING"} else {}
            if command == "HELLO":
                args = {"protocol": "chest-hunter-bridge", "version": 1}
            if command == "RESET_EPISODE":
                args = {"map_id": "m", "seed": 1}
            if command == "STEP":
                args |= {"step_id": 1, "action": _action()}
            if command == "NAV_QUERY":
                args |= {"start": [0, 0, 0], "goal": [1, 1, 0]}
            if command == "NAV_RAYCAST":
                args |= {"origin": [0, 0, 0], "bearings_deg": [0, 90]}
            if command == "GET_EVALUATION":
                args |= {"frame_id": 1}
            self.assertEqual(validate_request(request_header(command, 1, args)).command, command)

    def test_unknown_command(self) -> None:
        with self.assertRaises(ProtocolError) as cm:
            validate_request(request_header("EXEC_CONSOLE", 1, {"cmd": "quit"}))
        self.assertEqual(cm.exception.code, "unknown_command")

    def test_privileged_commands_are_named(self) -> None:
        self.assertEqual({"GET_ORACLE_STATE", "GET_EVALUATION"}, set(PRIVILEGED_COMMANDS))
        self.assertTrue(set(PRIVILEGED_COMMANDS).issubset(COMMANDS))

    def test_out_of_range_or_unknown_arguments(self) -> None:
        cases: list[tuple[str, dict]] = [
            ("RESET_EPISODE", {"map_id": "m", "seed": -1}),
            ("RESET_EPISODE", {"map_id": "m", "seed": 1,
                               "observation": {"width": 123, "height": 45}}),
            ("RESET_EPISODE", {"map_id": "m", "seed": 1, "fov_deg": 170}),
            ("RESET_EPISODE", {"map_id": "m", "seed": 1, "navigation": "teleport"}),
            ("RESET_EPISODE", {"map_id": "", "seed": 1}),
            ("STEP", {"episode_id": "e", "step_id": 0, "action": IDLE_ACTION}),
            ("STEP", {"episode_id": "e", "step_id": 1, "action": IDLE_ACTION, "ticks": 31}),
            ("NAV_QUERY", {"episode_id": "e", "start": [0, 0], "goal": [1, 1, 0]}),
            ("NAV_QUERY", {"episode_id": "e", "start": [0, 0, 1e9], "goal": [1, 1, 0]}),
            ("NAV_RAYCAST", {"episode_id": "e", "origin": [0, 0, 0],
                             "bearings_deg": list(range(33))}),
            ("PING", {"nonce": "x"}),
            ("GET_OBSERVATION", {"episode_id": "e", "chests": True}),
        ]
        for command, args in cases:
            with self.subTest(command=command, args=args):
                with self.assertRaises(ProtocolError) as cm:
                    validate_request(request_header(command, 1, args))
                self.assertIn(cm.exception.code, {"invalid_argument", "invalid_action"})

    def test_request_id_must_be_a_positive_integer(self) -> None:
        for rid in [0, -1, 1.5, "1", True, None]:
            with self.subTest(rid=rid):
                with self.assertRaises(ProtocolError):
                    validate_request({"command": "PING", "request_id": rid, "args": {}})

    def test_error_response_shape(self) -> None:
        r = error_response("busy", "x", command="PING", request_id=3)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["error"], {"code": "busy", "message": "x"})
        with self.assertRaises(ValueError):
            ProtocolError("not_a_code", "x")
        self.assertIn("timeout", ERROR_CODES)

    def test_prefix_is_network_byte_order(self) -> None:
        data = encode_message({"k": "v"}, b"12345")
        self.assertEqual(struct.unpack("!I", data[12:16])[0], 5)
