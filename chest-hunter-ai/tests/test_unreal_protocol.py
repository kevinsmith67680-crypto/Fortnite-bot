"""Wire protocol: framing, size limits, versioning and request/action validation."""

from __future__ import annotations

import io
import struct

import pytest

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


class TestFraming:
    def test_round_trip_with_binary_payload(self) -> None:
        payload = bytes(range(256)) * 10
        data = encode_message({"command": "PING", "n": 1}, payload)
        message = read_message(_reader(data))
        assert message.header == {"command": "PING", "n": 1}
        assert message.payload == payload

    def test_prefix_layout_matches_the_cpp_test_vector(self) -> None:
        data = encode_message({"command": "PING", "request_id": 1, "args": {}})
        magic, version, flags, hlen, plen = PREFIX.unpack(data[:16])
        assert (magic, version, flags, plen) == (MAGIC, PROTOCOL_VERSION, 0, 0)
        assert hlen == len(data) - 16
        # Same bytes as asserted by ResearchProtocolTests.cpp (magic, version 1).
        assert data[:6].hex() == "434852420001"

    def test_images_are_raw_bytes_not_json(self) -> None:
        payload = b"\x00\x01\x02" * 1000
        data = encode_message({"encoding": "bgr8"}, payload)
        header_len = PREFIX.unpack(data[:16])[3]
        assert len(data) == 16 + header_len + len(payload)
        assert b"[" not in data[16:16 + header_len].replace(b"[]", b"")

    def test_bad_magic(self) -> None:
        data = b"XXXX" + encode_message({"a": 1})[4:]
        with pytest.raises(ProtocolError) as exc:
            parse_prefix(data[:16])
        assert exc.value.code == "bad_magic"

    def test_version_mismatch(self) -> None:
        prefix = PREFIX.pack(MAGIC, PROTOCOL_VERSION + 1, 0, 2, 0)
        with pytest.raises(ProtocolError) as exc:
            parse_prefix(prefix)
        assert exc.value.code == "version_mismatch"

    @pytest.mark.parametrize(("hlen", "plen", "code"), [
        (MAX_HEADER_BYTES + 1, 0, "too_large"),
        (1, 0, "bad_header"),
        (10, MAX_PAYLOAD_BYTES + 1, "too_large"),
    ])
    def test_size_limits(self, hlen: int, plen: int, code: str) -> None:
        with pytest.raises(ProtocolError) as exc:
            parse_prefix(PREFIX.pack(MAGIC, PROTOCOL_VERSION, 0, hlen, plen))
        assert exc.value.code == code

    def test_encoder_refuses_oversized_payload_and_nan(self) -> None:
        with pytest.raises(ProtocolError) as exc:
            encode_message({}, b"\0" * (MAX_PAYLOAD_BYTES + 1))
        assert exc.value.code == "too_large"
        with pytest.raises(ProtocolError) as exc:
            encode_message({"x": float("nan")})
        assert exc.value.code == "bad_header"

    def test_truncated_prefix(self) -> None:
        with pytest.raises(ProtocolError) as exc:
            parse_prefix(b"CHRB")
        assert exc.value.code == "bad_header"

    @pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b'{"x": NaN}', b"\xff\xfe"])
    def test_bad_headers(self, raw: bytes) -> None:
        with pytest.raises(ProtocolError) as exc:
            parse_header(raw)
        assert exc.value.code in {"bad_json", "bad_header"}

    def test_truncated_stream(self) -> None:
        data = encode_message({"a": 1}, b"abc")
        with pytest.raises(ConnectionError):
            read_message(_reader(data[:-1]))


class TestActions:
    def test_valid_action_is_normalised(self) -> None:
        out = validate_action(_action(move_forward=1, look_yaw=-0.5))
        assert out["move_forward"] == 1.0 and isinstance(out["move_forward"], float)
        assert out["sprint"] is False and out["crouch"] is False

    @pytest.mark.parametrize("bad", [
        _action(move_forward=1.5), _action(look_pitch=-1.01), _action(move_right=float("inf")),
        _action(jump=1), _action(interact="yes"), _action(move_forward=True),
        {**_action(), "teleport": True}, {k: v for k, v in IDLE_ACTION.items() if k != "interact"},
        "forward", None,
    ])
    def test_invalid_actions(self, bad: object) -> None:
        with pytest.raises(ProtocolError) as exc:
            validate_action(bad)
        assert exc.value.code == "invalid_action"

    def test_no_action_key_can_address_hidden_state_or_scripts(self) -> None:
        with pytest.raises(ProtocolError):
            validate_action({**_action(), "exec": "quit"})
        with pytest.raises(ProtocolError):
            validate_action({**_action(), "set_position": [1, 2, 3]})


class TestRequests:
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
            assert validate_request(request_header(command, 1, args)).command == command

    def test_unknown_command(self) -> None:
        with pytest.raises(ProtocolError) as exc:
            validate_request(request_header("EXEC_CONSOLE", 1, {"cmd": "quit"}))
        assert exc.value.code == "unknown_command"

    def test_privileged_commands_are_named(self) -> None:
        assert {"GET_ORACLE_STATE", "GET_EVALUATION"} == set(PRIVILEGED_COMMANDS)
        assert set(PRIVILEGED_COMMANDS).issubset(COMMANDS)

    @pytest.mark.parametrize(("command", "args"), [
        ("RESET_EPISODE", {"map_id": "m", "seed": -1}),
        ("RESET_EPISODE", {"map_id": "m", "seed": 1, "observation": {"width": 123, "height": 45}}),
        ("RESET_EPISODE", {"map_id": "m", "seed": 1, "fov_deg": 170}),
        ("RESET_EPISODE", {"map_id": "m", "seed": 1, "navigation": "teleport"}),
        ("RESET_EPISODE", {"map_id": "", "seed": 1}),
        ("STEP", {"episode_id": "e", "step_id": 0, "action": IDLE_ACTION}),
        ("STEP", {"episode_id": "e", "step_id": 1, "action": IDLE_ACTION, "ticks": 31}),
        ("NAV_QUERY", {"episode_id": "e", "start": [0, 0], "goal": [1, 1, 0]}),
        ("NAV_QUERY", {"episode_id": "e", "start": [0, 0, 1e9], "goal": [1, 1, 0]}),
        ("NAV_RAYCAST", {"episode_id": "e", "origin": [0, 0, 0], "bearings_deg": list(range(33))}),
        ("PING", {"nonce": "x"}),
        ("GET_OBSERVATION", {"episode_id": "e", "chests": True}),
    ])
    def test_out_of_range_or_unknown_arguments(self, command: str, args: dict) -> None:
        with pytest.raises(ProtocolError) as exc:
            validate_request(request_header(command, 1, args))
        assert exc.value.code in {"invalid_argument", "invalid_action"}

    @pytest.mark.parametrize("rid", [0, -1, 1.5, "1", True, None])
    def test_request_id_must_be_a_positive_integer(self, rid: object) -> None:
        with pytest.raises(ProtocolError):
            validate_request({"command": "PING", "request_id": rid, "args": {}})

    def test_error_response_shape(self) -> None:
        r = error_response("busy", "x", command="PING", request_id=3)
        assert r["status"] == "error" and r["error"] == {"code": "busy", "message": "x"}
        with pytest.raises(ValueError):
            ProtocolError("not_a_code", "x")
        assert "timeout" in ERROR_CODES

    def test_prefix_is_network_byte_order(self) -> None:
        data = encode_message({"k": "v"}, b"12345")
        assert struct.unpack("!I", data[12:16])[0] == 5
