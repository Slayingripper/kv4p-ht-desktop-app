from __future__ import annotations

import math
import struct

import pytest

from kv4p_ht.protocol import (
    DELIMITER,
    FEAT_HAS_ESP32_AFSK,
    FEAT_HAS_HL,
    FEAT_HAS_PHYS_PTT,
    PROTO_MTU,
    EspCmd,
    FrameParser,
    FrameSender,
    HostCmd,
    pack_config,
    pack_filters,
    pack_group,
    pack_hl,
    pack_rssi,
    rssi_to_s_meter,
    unpack_rssi,
    unpack_version,
    unpack_window_update,
)


class TestHostCmd:
    def test_values(self):
        assert HostCmd.PTT_DOWN == 0x01
        assert HostCmd.PTT_UP == 0x02
        assert HostCmd.GROUP == 0x03
        assert HostCmd.FILTERS == 0x04
        assert HostCmd.STOP == 0x05
        assert HostCmd.CONFIG == 0x06
        assert HostCmd.TX_AUDIO == 0x07
        assert HostCmd.HL == 0x08
        assert HostCmd.RSSI == 0x09
        assert HostCmd.TX_AX25 == 0x0A

    def test_unique(self):
        vals = [cmd.value for cmd in HostCmd]
        assert len(vals) == len(set(vals))


class TestEspCmd:
    def test_values(self):
        assert EspCmd.SMETER_REPORT == 0x53
        assert EspCmd.PHYS_PTT_DOWN == 0x44
        assert EspCmd.PHYS_PTT_UP == 0x55
        assert EspCmd.DEBUG_INFO == 0x01
        assert EspCmd.DEBUG_ERROR == 0x02
        assert EspCmd.DEBUG_WARN == 0x03
        assert EspCmd.DEBUG_DEBUG == 0x04
        assert EspCmd.DEBUG_TRACE == 0x05
        assert EspCmd.HELLO == 0x06
        assert EspCmd.RX_AUDIO == 0x07
        assert EspCmd.VERSION == 0x08
        assert EspCmd.WINDOW_UPDATE == 0x09
        assert EspCmd.RX_AX25_PACKET == 0x0A

    def test_unique(self):
        vals = [cmd.value for cmd in EspCmd]
        assert len(vals) == len(set(vals))


class TestFeatureFlags:
    def test_values(self):
        assert FEAT_HAS_HL == 1
        assert FEAT_HAS_PHYS_PTT == 2
        assert FEAT_HAS_ESP32_AFSK == 4

    def test_no_overlap(self):
        assert (FEAT_HAS_HL & FEAT_HAS_PHYS_PTT) == 0
        assert (FEAT_HAS_HL & FEAT_HAS_ESP32_AFSK) == 0
        assert (FEAT_HAS_PHYS_PTT & FEAT_HAS_ESP32_AFSK) == 0


class TestPackGroup:
    def test_packed_struct(self):
        result = pack_group(1, 144.390, 144.390, 67, 3, 71)
        expected = struct.pack('<BffBBB', 1, 144.390, 144.390, 67, 3, 71)
        assert result == expected

    def test_byte_order_and_format(self):
        result = pack_group(0, 123.456, 789.012, 0, 0, 0)
        assert len(result) == struct.calcsize('<BffBBB')
        assert len(result) == 1 + 4 + 4 + 1 + 1 + 1

    def test_values_round_trip(self):
        bw, ftx, frx, ct, sq, cr = 3, 146.520, 146.520, 100, 5, 88
        data = pack_group(bw, ftx, frx, ct, sq, cr)
        unpacked = struct.unpack('<BffBBB', data)
        assert unpacked[0] == bw
        assert abs(unpacked[1] - ftx) < 1e-4
        assert abs(unpacked[2] - frx) < 1e-4
        assert unpacked[3] == ct
        assert unpacked[4] == sq
        assert unpacked[5] == cr


class TestPackFilters:
    @pytest.mark.parametrize("pre,high,low,expected_flags", [
        (False, False, False, 0),
        (True,  False, False, 1),
        (False, True,  False, 2),
        (False, False, True,  4),
        (True,  True,  False, 3),
        (True,  False, True,  5),
        (False, True,  True,  6),
        (True,  True,  True,  7),
    ])
    def test_all_combinations(self, pre, high, low, expected_flags):
        result = pack_filters(pre, high, low)
        assert result == struct.pack('<B', expected_flags)

    def test_correct_length(self):
        assert len(pack_filters(True, True, True)) == 1


class TestPackConfig:
    @pytest.mark.parametrize("val", [True, False])
    def test_pack_config(self, val):
        result = pack_config(val)
        assert result == struct.pack('<?', val)
        assert len(result) == 1

    def test_default_true(self):
        result = pack_config(True)
        assert struct.unpack('<?', result)[0] is True


class TestPackHl:
    @pytest.mark.parametrize("val", [True, False])
    def test_pack_hl(self, val):
        result = pack_hl(val)
        assert result == struct.pack('<?', val)
        assert len(result) == 1


class TestPackRssi:
    @pytest.mark.parametrize("val", [True, False])
    def test_pack_rssi(self, val):
        result = pack_rssi(val)
        assert result == struct.pack('<?', val)
        assert len(result) == 1


class TestUnpackVersion:
    def test_unpack_full(self):
        ver, status, win, mod, features = 0x0102, ord('R'), 16, 0, 7
        data = struct.pack('<H B I I B', ver, status, win, mod, features)
        result = unpack_version(data)
        assert result['ver'] == ver
        assert result['radio_status'] == 'R'
        assert result['window_size'] == win
        assert result['module_type'] == mod
        assert result['has_hl'] is True
        assert result['has_phys_ptt'] is True
        assert result['has_esp32_afsk'] is True

    def test_no_features(self):
        data = struct.pack('<H B I I B', 0, ord('O'), 8, 1, 0)
        result = unpack_version(data)
        assert result['has_hl'] is False
        assert result['has_phys_ptt'] is False
        assert result['has_esp32_afsk'] is False
        assert result['module_type'] == 1
        assert result['window_size'] == 8

    def test_feature_hl_only(self):
        data = struct.pack('<H B I I B', 1, ord('X'), 4, 0, FEAT_HAS_HL)
        result = unpack_version(data)
        assert result['has_hl'] is True
        assert result['has_phys_ptt'] is False
        assert result['has_esp32_afsk'] is False

    def test_feature_phys_ptt_only(self):
        data = struct.pack('<H B I I B', 1, ord('X'), 4, 0, FEAT_HAS_PHYS_PTT)
        result = unpack_version(data)
        assert result['has_hl'] is False
        assert result['has_phys_ptt'] is True
        assert result['has_esp32_afsk'] is False

    def test_feature_esp32_afsk_only(self):
        data = struct.pack('<H B I I B', 1, ord('X'), 4, 0, FEAT_HAS_ESP32_AFSK)
        result = unpack_version(data)
        assert result['has_hl'] is False
        assert result['has_phys_ptt'] is False
        assert result['has_esp32_afsk'] is True

    def test_radio_status_char(self):
        for ch in 'ABCDEFG':
            data = struct.pack('<H B I I B', 0, ord(ch), 0, 0, 0)
            result = unpack_version(data)
            assert result['radio_status'] == ch

    def test_data_too_short(self):
        with pytest.raises(struct.error):
            unpack_version(b'\x01\x02\x03')


class TestUnpackWindowUpdate:
    def test_unpack(self):
        data = struct.pack('<I', 12345)
        assert unpack_window_update(data) == 12345

    def test_zero(self):
        assert unpack_window_update(b'\x00\x00\x00\x00') == 0

    def test_max(self):
        assert unpack_window_update(b'\xff\xff\xff\xff') == 4294967295


class TestUnpackRssi:
    def test_empty(self):
        assert unpack_rssi(b'') == 0

    def test_first_byte(self):
        assert unpack_rssi(b'\x2A') == 42

    def test_first_byte_only(self):
        assert unpack_rssi(b'\x01\x02\x03') == 1

    def test_zero(self):
        assert unpack_rssi(b'\x00') == 0

    def test_255(self):
        assert unpack_rssi(b'\xff') == 255


class TestRssiToSMeter:
    def test_zero_returns_one(self):
        assert rssi_to_s_meter(0) == 1

    def test_one(self):
        result = rssi_to_s_meter(1)
        assert 1 <= result <= 9

    def test_255(self):
        result = rssi_to_s_meter(255)
        assert 1 <= result <= 9

    def test_clamps_min(self):
        assert rssi_to_s_meter(0) == 1

    def test_clamps_max(self):
        result = rssi_to_s_meter(255)
        assert result <= 9

    def test_known_values(self):
        result = rssi_to_s_meter(100)
        val = 9.73 * math.log(0.0297 * 100) - 1.88
        expected = max(1, min(9, round(val)))
        assert result == expected


class TestFrameSender:
    def build_frame(self, cmd: int, payload: bytes) -> bytes:
        return DELIMITER + bytes([cmd]) + struct.pack('<H', len(payload)) + payload

    def test_send_empty_payload(self):
        collected = []
        sender = FrameSender(collected.append)
        sender._send(HostCmd.PTT_DOWN)
        assert collected == [self.build_frame(HostCmd.PTT_DOWN, b'')]

    def test_send_with_payload(self):
        collected = []
        sender = FrameSender(collected.append)
        payload = b'\x01\x02\x03'
        sender._send(HostCmd.GROUP, payload)
        assert collected == [self.build_frame(HostCmd.GROUP, payload)]

    def test_multiple_sends(self):
        collected = []
        sender = FrameSender(collected.append)
        sender._send(HostCmd.PTT_DOWN)
        sender._send(HostCmd.PTT_UP)
        sender._send(HostCmd.GROUP, b'\x00' * 10)
        assert len(collected) == 3
        assert collected[0] == self.build_frame(HostCmd.PTT_DOWN, b'')
        assert collected[1] == self.build_frame(HostCmd.PTT_UP, b'')
        assert collected[2] == self.build_frame(HostCmd.GROUP, b'\x00' * 10)

    def test_ptt_down(self):
        collected = []
        FrameSender(collected.append).ptt_down()
        assert collected == [self.build_frame(HostCmd.PTT_DOWN, b'')]

    def test_ptt_up(self):
        collected = []
        FrameSender(collected.append).ptt_up()
        assert collected == [self.build_frame(HostCmd.PTT_UP, b'')]

    def test_stop(self):
        collected = []
        FrameSender(collected.append).stop()
        assert collected == [self.build_frame(HostCmd.STOP, b'')]

    def test_set_group(self):
        collected = []
        payload = struct.pack('<BffBBB', 1, 144.390, 144.390, 67, 3, 71)
        FrameSender(collected.append).set_group(1, 144.390, 144.390, 67, 3, 71)
        assert collected == [self.build_frame(HostCmd.GROUP, payload)]

    def test_set_group_defaults(self):
        collected = []
        payload = struct.pack('<BffBBB', 0, 144.390, 144.390, 0, 3, 0)
        FrameSender(collected.append).set_group()
        assert collected == [self.build_frame(HostCmd.GROUP, payload)]

    def test_set_filters(self):
        collected = []
        payload = struct.pack('<B', 7)
        FrameSender(collected.append).set_filters(True, True, True)
        assert collected == [self.build_frame(HostCmd.FILTERS, payload)]

    def test_set_filters_defaults(self):
        collected = []
        FrameSender(collected.append).set_filters()
        assert collected == [self.build_frame(HostCmd.FILTERS, b'\x00')]

    def test_send_config(self):
        collected = []
        FrameSender(collected.append).send_config(True)
        assert collected == [self.build_frame(HostCmd.CONFIG, b'\x01')]

    def test_send_config_false(self):
        collected = []
        FrameSender(collected.append).send_config(False)
        assert collected == [self.build_frame(HostCmd.CONFIG, b'\x00')]

    def test_send_tx_audio(self):
        collected = []
        payload = b'\x00\x01\x02\x03'
        FrameSender(collected.append).send_tx_audio(payload)
        assert collected == [self.build_frame(HostCmd.TX_AUDIO, payload)]

    def test_send_tx_audio_empty(self):
        collected = []
        FrameSender(collected.append).send_tx_audio(b'')
        assert collected == [self.build_frame(HostCmd.TX_AUDIO, b'')]

    def test_set_hl_true(self):
        collected = []
        FrameSender(collected.append).set_hl(True)
        assert collected == [self.build_frame(HostCmd.HL, b'\x01')]

    def test_set_hl_false(self):
        collected = []
        FrameSender(collected.append).set_hl(False)
        assert collected == [self.build_frame(HostCmd.HL, b'\x00')]

    def test_set_rssi_true(self):
        collected = []
        FrameSender(collected.append).set_rssi(True)
        assert collected == [self.build_frame(HostCmd.RSSI, b'\x01')]

    def test_set_rssi_false(self):
        collected = []
        FrameSender(collected.append).set_rssi(False)
        assert collected == [self.build_frame(HostCmd.RSSI, b'\x00')]

    def test_set_rssi_default(self):
        collected = []
        FrameSender(collected.append).set_rssi()
        assert collected == [self.build_frame(HostCmd.RSSI, b'\x01')]

    def test_send_tx_ax25(self):
        collected = []
        payload = b'\x00' * 50
        FrameSender(collected.append).send_tx_ax25(payload)
        assert collected == [self.build_frame(HostCmd.TX_AX25, payload)]


class TestFrameParser:

    def test_single_complete_frame(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        payload = b'\x01\x02\x03'
        frame = DELIMITER + bytes([EspCmd.SMETER_REPORT]) + struct.pack('<H', len(payload)) + payload
        parser.feed(frame)
        assert len(frames) == 1
        assert frames[0] == (EspCmd.SMETER_REPORT, payload)

    def test_multiple_frames_in_sequence(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        f1 = DELIMITER + b'\x53' + struct.pack('<H', 2) + b'\xAB\xCD'
        f2 = DELIMITER + b'\x44' + struct.pack('<H', 1) + b'\xEF'
        parser.feed(f1 + f2)
        assert len(frames) == 2
        assert frames[0] == (0x53, b'\xAB\xCD')
        assert frames[1] == (0x44, b'\xEF')

    def test_frame_split_across_feed_calls(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        frame = DELIMITER + b'\x01' + struct.pack('<H', 4) + b'\xAA\xBB\xCC\xDD'
        parser.feed(frame[:5])
        parser.feed(frame[5:])
        assert len(frames) == 1
        assert frames[0] == (0x01, b'\xAA\xBB\xCC\xDD')

    def test_zero_length_payload(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        frame = DELIMITER + b'\x02' + struct.pack('<H', 0)
        parser.feed(frame)
        assert len(frames) == 1
        assert frames[0] == (0x02, b'')

    def test_payload_exceeds_mtu_rejected(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        frame = DELIMITER + b'\x03' + struct.pack('<H', PROTO_MTU + 1) + b'\xFF' * (PROTO_MTU + 1)
        parser.feed(frame)
        assert len(frames) == 0

    def test_payload_at_mtu_accepted(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        payload = b'\xBB' * PROTO_MTU
        frame = DELIMITER + b'\x04' + struct.pack('<H', PROTO_MTU) + payload
        parser.feed(frame)
        assert len(frames) == 1
        assert frames[0] == (0x04, payload)

    def test_partial_delimiter_then_full(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        partial = b'\xDE\xAD'
        full = DELIMITER + b'\x05' + struct.pack('<H', 1) + b'\xEE'
        parser.feed(partial + full)
        assert len(frames) == 1
        assert frames[0] == (0x05, b'\xEE')

    def test_delimiter_bytes_in_payload_no_false_frame(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        payload = DELIMITER
        frame = DELIMITER + b'\x06' + struct.pack('<H', len(payload)) + payload
        parser.feed(frame)
        assert len(frames) == 1
        assert frames[0] == (0x06, DELIMITER)

    def test_empty_feed(self):
        parser = FrameParser(lambda cmd, pl: None)
        parser.feed(b'')
        collector = []
        parser = FrameParser(lambda cmd, pl: collector.append(1))
        parser.feed(b'')
        assert collector == []

    def test_resync_after_garbage(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        garbage = b'\x00\x01\x02\xDE\xAD'
        frame = DELIMITER + b'\x07' + struct.pack('<H', 2) + b'\x11\x22'
        parser.feed(garbage + frame)
        assert len(frames) == 1
        assert frames[0] == (0x07, b'\x11\x22')

    def test_garbage_with_delimiter_start_then_resync(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        garbage = b'\xDE\x00\xAD\xBE\xEF\xFF'
        frame = DELIMITER + b'\x08' + struct.pack('<H', 1) + b'\x99'
        parser.feed(garbage + frame)
        assert len(frames) == 1
        assert frames[0] == (0x08, b'\x99')

    def test_many_partial_fragments(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        payload = b'\xFF' * 10
        frame = DELIMITER + b'\x09' + struct.pack('<H', len(payload)) + payload
        chunks = [frame[i:i+3] for i in range(0, len(frame), 3)]
        for chunk in chunks:
            parser.feed(chunk)
        assert len(frames) == 1
        assert frames[0] == (0x09, payload)

    def test_reset_after_mtu_rejection_syncs_next_frame(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        bad_frame = DELIMITER + b'\x0A' + struct.pack('<H', PROTO_MTU + 5) + b'\x00' * (PROTO_MTU + 5)
        good_frame = DELIMITER + b'\x0B' + struct.pack('<H', 2) + b'\x55\x66'
        parser.feed(bad_frame + good_frame)
        assert len(frames) == 1
        assert frames[0] == (0x0B, b'\x55\x66')

    def test_callback_receives_correct_cmd_and_payload(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        payload = b'\xF0\xF1\xF2\xF3'
        frame = DELIMITER + b'\x0C' + struct.pack('<H', len(payload)) + payload
        parser.feed(frame)
        assert frames[0][0] == 0x0C
        assert frames[0][1] == payload
        assert type(frames[0][0]) is int
        assert type(frames[0][1]) is bytes

    def test_no_callbacks_for_incomplete_frames(self):
        frames = []
        parser = FrameParser(lambda cmd, pl: frames.append((cmd, pl)))
        parser.feed(DELIMITER)
        parser.feed(b'\x0D')
        parser.feed(b'\x01')
        parser.feed(b'\x00')
        assert len(frames) == 0

    def test_multiple_writes_to_sender_correct_concat(self):
        collected = b''
        def write_fn(data):
            nonlocal collected
            collected += data
        sender = FrameSender(write_fn)
        sender.ptt_down()
        sender.ptt_up()
        sender.stop()
        expected = (
            DELIMITER + bytes([HostCmd.PTT_DOWN]) + struct.pack('<H', 0) +
            DELIMITER + bytes([HostCmd.PTT_UP]) + struct.pack('<H', 0) +
            DELIMITER + bytes([HostCmd.STOP]) + struct.pack('<H', 0)
        )
        assert collected == expected
