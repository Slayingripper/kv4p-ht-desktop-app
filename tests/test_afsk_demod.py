from __future__ import annotations

import math
import random
import struct

import numpy as np

from kv4p_ht.afsk import (
    PREAMBLE_FLAGS,
    SQUELCH_OPEN_MS,
    build_tx_waveform_from_body,
    crc_ccitt,
)
from kv4p_ht.afsk_demod import (
    _CRC_TABLE,
    AX25_CRC_OK,
    BAUD,
    MARK_HZ,
    SPACE_HZ,
    SPB,
    SR,
    AfskDemodulator,
)
from kv4p_ht.aprs import encode_ax25_ui


class TestConstants:
    def test_mark_hz(self):
        assert MARK_HZ == 1200.0

    def test_space_hz(self):
        assert SPACE_HZ == 2200.0

    def test_baud(self):
        assert BAUD == 1200

    def test_sr(self):
        assert SR == 48000

    def test_spb(self):
        assert SPB == SR // BAUD
        assert SPB == 40

    def test_ax25_crc_ok(self):
        assert AX25_CRC_OK == 0xF0B8


class TestCrcTable:
    def test_table_has_256_entries(self):
        assert len(_CRC_TABLE) == 256

    def test_table_entries_are_16bit(self):
        for v in _CRC_TABLE:
            assert 0 <= v <= 0xFFFF

    def test_known_entry_from_crc_ccitt(self):
        for b in [0x00, 0x01, 0x41, 0x7E, 0xFF, 0xA5, 0x5A]:
            actual = crc_ccitt(bytes([b])) ^ 0xFFFF
            crc = 0xFFFF
            crc = ((crc >> 8) ^ _CRC_TABLE[(crc ^ b) & 0xFF])
            assert crc == actual


class TestInitialization:
    def test_callback_defaults_to_none(self):
        d = AfskDemodulator()
        assert d.callback is None

    def test_callback_stored(self):
        cb = lambda pkt: None
        d = AfskDemodulator(callback=cb)
        assert d.callback is cb

    def test_quadrature_oscillators(self):
        d = AfskDemodulator()
        assert d._ph_mark == 0.0
        assert d._ph_space == 0.0
        assert d._inc_mark == 2.0 * math.pi * MARK_HZ / SR
        assert d._inc_space == 2.0 * math.pi * SPACE_HZ / SR

    def test_sliding_correlation_arrays(self):
        d = AfskDemodulator()
        assert len(d._c0r) == SPB
        assert len(d._c0i) == SPB
        assert len(d._c1r) == SPB
        assert len(d._c1i) == SPB
        assert all(v == 0.0 for v in d._c0r)
        assert all(v == 0.0 for v in d._c0i)
        assert all(v == 0.0 for v in d._c1r)
        assert all(v == 0.0 for v in d._c1i)
        assert d._jcorr == 0

    def test_fdiff_history(self):
        d = AfskDemodulator()
        assert d._fdiff_hist_len == 3
        assert d._fdiff_hist == []
        assert d._last_fdiff == 0.0

    def test_timing_recovery(self):
        d = AfskDemodulator()
        assert d._last_xing == -1
        assert d._nsamp == 0

    def test_bit_decoder_state(self):
        d = AfskDemodulator()
        assert d._data == 0
        assert d._bitcount == 0
        assert d._state == 'WAITING'
        assert d._flag_count == 0
        assert d._flag_sep is False

    def test_frame_buffer(self):
        d = AfskDemodulator()
        assert d._frame == bytearray()
        assert d._crc == 0xFFFF

    def test_carrier_detect(self):
        d = AfskDemodulator()
        assert d._carrier_energy == 0.0
        assert d._carrier_threshold == 0.001
        assert d._carrier_on is False
        assert d.diagnostics() == {
            'carrier': False,
            'carrier_starts': 0,
            'valid_frames': 0,
            'rejected_frames': 0,
        }


class TestReset:
    def test_reset_clears_state(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._flag_count = 5
        d._flag_sep = True
        d._data = 0xAB
        d._bitcount = 4
        d._frame = bytearray(b'\x01\x02\x03')
        d._crc = 0x1234
        d._carrier_energy = 0.5
        d._reset()
        assert d._state == 'WAITING'
        assert d._flag_count == 0
        assert d._flag_sep is False
        assert d._data == 0
        assert d._bitcount == 0
        assert d._frame == bytearray()
        assert d._crc == 0xFFFF
        assert d._carrier_energy == 0.0

    def test_reset_preserves_other_attrs(self):
        d = AfskDemodulator()
        d._ph_mark = 1.5
        d._ph_space = 2.0
        d._reset()
        assert d._ph_mark == 1.5
        assert d._ph_space == 2.0
        assert d._nsamp == 0
        assert d._last_xing == -1


class TestCrcUpd:
    def _internal_crc(self, data: bytes) -> int:
        crc = 0xFFFF
        for b in data:
            crc = ((crc >> 8) ^ _CRC_TABLE[(crc ^ b) & 0xFF])
        return crc

    def test_crc_upd_single_byte(self):
        d = AfskDemodulator()
        d._crc_upd(0x41)
        assert d._crc == self._internal_crc(b'A')

    def test_crc_upd_multiple_bytes(self):
        d = AfskDemodulator()
        for b in b'Hello':
            d._crc_upd(b)
        assert d._crc == self._internal_crc(b'Hello')

    def test_crc_upd_accumulates(self):
        d = AfskDemodulator()
        for b in b'\x01\x02\x03':
            d._crc_upd(b)
        assert d._crc == self._internal_crc(b'\x01\x02\x03')

    def test_crc_upd_empty_noop(self):
        d = AfskDemodulator()
        assert d._crc == 0xFFFF

    def test_crc_upd_matches_afsk_ccitt(self):
        data = b'\x41\x42\x43\x44'
        expected = crc_ccitt(data) ^ 0xFFFF
        d = AfskDemodulator()
        for b in data:
            d._crc_upd(b)
        assert d._crc == expected

    def test_crc_upd_vs_table_direct(self):
        d = AfskDemodulator()
        d._crc_upd(0x00)
        expected = ((0xFFFF >> 8) ^ _CRC_TABLE[(0xFFFF ^ 0x00) & 0xFF])
        assert d._crc == expected

    def test_crc_upd_byte_0x00(self):
        d = AfskDemodulator()
        d._crc_upd(0x00)
        assert d._crc == 0x0F87

    def test_crc_upd_byte_0x41(self):
        d = AfskDemodulator()
        d._crc_upd(0x41)
        assert d._crc == 0x5C0A

    def test_crc_upd_byte_0x03(self):
        d = AfskDemodulator()
        d._crc_upd(0x03)
        assert d._crc == 0x3D1C

    def test_crc_upd_byte_0xF0(self):
        d = AfskDemodulator()
        d._crc_upd(0xF0)
        assert d._crc == 0xF808

    def test_crc_upd_byte_0x7E(self):
        d = AfskDemodulator()
        d._crc_upd(0x7E)
        assert d._crc == self._internal_crc(b'\x7E')

    def test_crc_upd_multiple_known(self):
        d = AfskDemodulator()
        for b in b'\x00\x41\x03\xF0':
            d._crc_upd(b)
        assert d._crc == self._internal_crc(b'\x00\x41\x03\xF0')


class TestEmitByte:
    def test_appends_to_frame(self):
        d = AfskDemodulator()
        d._emit_byte(0x41)
        assert d._frame == bytearray(b'A')

    def test_multiple_bytes_appended(self):
        d = AfskDemodulator()
        for b in b'\x01\x02\x03':
            d._emit_byte(b)
        assert d._frame == bytearray(b'\x01\x02\x03')

    def test_updates_crc(self):
        d = AfskDemodulator()
        d._emit_byte(0x41)
        expected_crc = ((0xFFFF >> 8) ^ _CRC_TABLE[(0xFFFF ^ 0x41) & 0xFF])
        assert d._crc == expected_crc

    def test_frame_order_preserved(self):
        d = AfskDemodulator()
        for b in range(256):
            d._emit_byte(b)
        assert d._frame == bytearray(range(256))

    def test_crc_consistent_with_separate_crc_upd(self):
        d1 = AfskDemodulator()
        d2 = AfskDemodulator()
        for b in b'ABCDEF':
            d1._emit_byte(b)
            d2._crc_upd(b)
        assert d1._frame == bytearray(b'ABCDEF')
        assert d1._crc == d2._crc

    def test_emit_byte_zero(self):
        d = AfskDemodulator()
        d._emit_byte(0x00)
        assert d._frame == bytearray(b'\x00')


class TestFinalize:
    def test_short_frame_rejected(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d._frame = bytearray(16)
        d._crc = AX25_CRC_OK
        d._finalize()
        assert len(captured) == 0
        assert d._state == 'WAITING'

    def test_invalid_crc_rejected(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d._frame = bytearray(18)
        d._crc = 0xFFFF
        d._finalize()
        assert len(captured) == 0
        assert d._state == 'WAITING'

    def test_exact_18_bytes_no_callback_if_crc_wrong(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d._frame = bytearray(18)
        d._crc = 0x0000
        d._finalize()
        assert len(captured) == 0

    def test_resets_after_finalize(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d._frame = bytearray(18)
        d._crc = AX25_CRC_OK
        d._frame[:] = b'\x00' * 18
        d._finalize()
        assert d._state == 'WAITING'
        assert d._frame == bytearray()
        assert d._crc == 0xFFFF


class TestOnTransition:
    def test_bits_zero_ignored(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._on_transition(0)
        assert d._state == 'DECODING'

    def test_bits_greater_than_seven_ignored(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._on_transition(8)
        assert d._state == 'DECODING'
        d._on_transition(100)
        assert d._state == 'DECODING'

    def test_bits_seven_in_waiting_transitions_to_pre_flag(self):
        d = AfskDemodulator()
        assert d._state == 'WAITING'
        d._on_transition(7)
        assert d._state == 'PRE_FLAG'
        assert d._flag_count == 1
        assert d._flag_sep is False

    def test_bits_seven_in_pre_flag_increments_flag_count(self):
        d = AfskDemodulator()
        d._state = 'PRE_FLAG'
        d._flag_count = 1
        d._on_transition(7)
        assert d._flag_count == 2

    def test_bits_seven_in_decoding_triggers_finalize_if_crc_ok(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._frame = bytearray(18)
        d._crc = AX25_CRC_OK
        d._on_transition(7)
        assert d._state == 'PRE_FLAG'

    def test_bits_seven_in_decoding_resets_if_crc_bad(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._frame = bytearray(18)
        d._crc = 0xFFFF
        d._on_transition(7)
        assert d._state == 'PRE_FLAG'
        assert d._frame == bytearray()

    def test_bits_seven_in_decoding_resets_if_short_frame(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d._state = 'DECODING'
        d._frame = bytearray(10)
        d._crc = AX25_CRC_OK
        d._on_transition(7)
        assert d._state == 'PRE_FLAG'
        assert len(captured) == 0

    def test_bits_one_to_six_in_pre_flag_starts_decoding(self):
        d = AfskDemodulator()
        d._state = 'PRE_FLAG'
        d._on_transition(3)
        assert d._state == 'DECODING'
        assert d._frame == bytearray()
        assert d._crc == 0xFFFF

    def test_bits_one_to_six_in_decoding_adds_bits(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._data = 0
        d._bitcount = 0
        d._on_transition(3)
        assert d._bitcount == 3

    def test_bits_two_emits_correct_data(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._on_transition(2)
        assert d._data == 0x40
        assert d._bitcount == 2

    def test_bits_one_emits_single_zero_bit(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._on_transition(1)
        assert d._bitcount == 1
        assert d._data == 0x00

    def test_bits_six_emits_five_ones_then_no_zero(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._data = 0x00
        d._bitcount = 0
        d._on_transition(6)
        assert d._bitcount == 5
        assert d._data == 0xF8

    def test_data_accumulates_into_bytes(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        for _ in range(8):
            d._on_transition(1)
        assert d._bitcount == 0
        assert d._data == 0x00
        assert d._frame == bytearray(b'\x00')

    def test_byte_accumulation_multiple_bytes(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        for _ in range(16):
            d._on_transition(2)
        assert d._bitcount == 0
        assert len(d._frame) == 4

    def test_flag_resets_data_and_bitcount(self):
        d = AfskDemodulator()
        d._state = 'WAITING'
        d._data = 0xFF
        d._bitcount = 7
        d._on_transition(7)
        assert d._data == 0
        assert d._bitcount == 0


class TestAddBit:
    def test_add_bit_zero(self):
        d = AfskDemodulator()
        d._add_bit(0)
        assert d._data == 0x00
        assert d._bitcount == 1

    def test_add_bit_one(self):
        d = AfskDemodulator()
        d._add_bit(1)
        assert d._data == 0x80
        assert d._bitcount == 1

    def test_add_bit_accumulates_lsb_first(self):
        d = AfskDemodulator()
        d._add_bit(1)
        d._add_bit(0)
        d._add_bit(1)
        assert d._data >> 5 == 0b101
        assert d._bitcount == 3

    def test_add_bit_completes_byte(self):
        d = AfskDemodulator()
        bits = [1, 0, 1, 0, 1, 0, 1, 1]
        for b in bits:
            d._add_bit(b)
        assert d._bitcount == 0
        assert d._data == 0x00
        assert d._frame == bytearray([0b11010101])

    def test_add_bit_multiple_bytes(self):
        d = AfskDemodulator()
        for b in [1] * 16:
            d._add_bit(b)
        assert len(d._frame) == 2
        assert d._bitcount == 0

    def test_add_bit_crc_updated(self):
        d = AfskDemodulator()
        for b in [1, 0, 1, 0, 1, 0, 1, 0]:
            d._add_bit(b)
        assert d._crc != 0xFFFF


class TestProcessEdgeCases:
    def test_empty_input(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process([])
        assert len(captured) == 0

    def test_all_zeros(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process([0.0] * 10000)
        assert len(captured) == 0
        assert d._carrier_on is False

    def test_single_sample(self):
        d = AfskDemodulator()
        d.process([0.1])
        assert d._nsamp == 1

    def test_two_samples(self):
        d = AfskDemodulator()
        d.process([0.1, -0.1])
        assert d._nsamp == 2


class TestNoiseRejection:
    def test_random_noise_no_false_decode(self):
        random.seed(42)
        noise = [random.uniform(-0.5, 0.5) for _ in range(SR * 5)]
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(noise)
        assert len(captured) == 0

    def test_gaussian_noise_no_false_decode(self):
        rng = np.random.default_rng(1234)
        noise = rng.normal(0, 0.15, SR * 5).tolist()
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(noise)
        assert len(captured) == 0

    def test_impulse_noise_no_false_decode(self):
        samples = [0.0] * SR
        for i in range(100):
            idx = random.randint(0, len(samples) - 1)
            samples[idx] = random.uniform(-1.0, 1.0)
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(samples)
        assert len(captured) == 0


class TestCarrierDetect:
    def test_carrier_off_initially(self):
        d = AfskDemodulator()
        assert d._carrier_on is False

    def test_carrier_on_with_signal(self):
        d = AfskDemodulator()
        for _ in range(SPB):
            d.process([0.1])
        assert d._carrier_on is True

    def test_carrier_off_with_silence(self):
        d = AfskDemodulator()
        d.process([0.0] * SPB)
        assert d._carrier_on is False

    def test_carrier_energy_increases_with_amplitude(self):
        d1 = AfskDemodulator()
        d2 = AfskDemodulator()
        for _ in range(SPB):
            d1.process([0.1])
            d2.process([0.01])
        assert d1._carrier_energy > d2._carrier_energy

    def test_carrier_threshold_default(self):
        d = AfskDemodulator()
        assert d._carrier_threshold == 0.001

    def test_carrier_decays_without_signal(self):
        d = AfskDemodulator()
        for _ in range(SPB):
            d.process([0.1])
        assert d._carrier_on is True
        for _ in range(SPB * 200):
            d.process([0.0])
        assert d._carrier_on is False

    def test_carrier_turns_on_off_on(self):
        d = AfskDemodulator()
        d.process([0.0] * SPB)
        was_off = d._carrier_on is False
        d.process([0.2] * SPB)
        was_on = d._carrier_on is True
        d.process([0.0] * SPB * 200)
        was_off_again = d._carrier_on is False
        d.process([0.2] * SPB)
        was_on_again = d._carrier_on is True
        assert was_off and was_on and was_off_again and was_on_again


class TestRoundTrip:
    def build_packet(self, source='N0CALL', dest='APZ010', info_text='>Hello, world!', digis=None):
        info_bytes = info_text.encode('ascii')
        body = encode_ax25_ui(source, dest, digis or [], info_bytes)
        waveform = build_tx_waveform_from_body(body)
        return body, waveform

    def test_basic_roundtrip(self):
        body, waveform = self.build_packet()
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(waveform.tolist())
        assert len(captured) == 1
        pkt = captured[0]
        assert pkt.get('source') == 'N0CALL'
        assert pkt.get('raw_frame') == body

    def test_status_message_decode(self):
        _, waveform = self.build_packet(info_text='>Testing status')
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(waveform.tolist())
        assert len(captured) == 1
        pkt = captured[0]
        assert pkt.get('type') == 'status'
        assert pkt.get('text') == 'Testing status'

    def test_beacon_message_decode(self):
        _, waveform = self.build_packet(info_text='=4040.00N/07400.00W-Test')
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(waveform.tolist())
        assert len(captured) == 1
        pkt = captured[0]
        assert pkt.get('type') == 'position'

    def test_message_with_ssid(self):
        body, waveform = self.build_packet(source='N0CALL-7', dest='APZ010-3', info_text='>Hello')
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(waveform.tolist())
        assert len(captured) == 1
        pkt = captured[0]
        assert pkt.get('source') == 'N0CALL-7'
        assert pkt.get('raw_frame') == body

    def test_different_info_field(self):
        info_text = '>Another test message with more data for verification purposes'
        body, waveform = self.build_packet(info_text=info_text)
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(waveform.tolist())
        assert len(captured) == 1
        pkt = captured[0]
        assert pkt.get('raw') == info_text
        assert pkt.get('raw_frame') == body

    def test_source_and_dest_recovered(self):
        body, waveform = self.build_packet(source='MYCALL', dest='DST', info_text='>hello')
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(waveform.tolist())
        assert len(captured) == 1
        pkt = captured[0]
        assert pkt.get('source') == 'MYCALL'
        assert pkt.get('raw_frame') == body


class TestRoundTripChunked:
    def test_chunks_of_1_sample(self):
        body = encode_ax25_ui('N0CALL', 'APZ010', [], b'>hello')
        waveform = build_tx_waveform_from_body(body).tolist()
        captured = []
        d = AfskDemodulator(callback=captured.append)
        for s in waveform:
            d.process([s])
        assert len(captured) == 1

    def test_chunks_of_100_samples(self):
        body = encode_ax25_ui('N0CALL', 'APZ010', [], b'>hello')
        waveform = build_tx_waveform_from_body(body).tolist()
        captured = []
        d = AfskDemodulator(callback=captured.append)
        for i in range(0, len(waveform), 100):
            d.process(waveform[i:i + 100])
        assert len(captured) == 1

    def test_chunks_of_500_samples(self):
        body = encode_ax25_ui('N0CALL', 'APZ010', [], b'>chunked')
        waveform = build_tx_waveform_from_body(body).tolist()
        captured = []
        d = AfskDemodulator(callback=captured.append)
        for i in range(0, len(waveform), 500):
            d.process(waveform[i:i + 500])
        assert len(captured) == 1

    def test_chunks_of_varying_sizes(self):
        body = encode_ax25_ui('N0CALL', 'APZ010', [], b'>varying')
        waveform = build_tx_waveform_from_body(body).tolist()
        captured = []
        d = AfskDemodulator(callback=captured.append)
        pos = 0
        sizes = [50, 200, 30, 1000, 5, 300]
        for sz in sizes:
            d.process(waveform[pos:pos + sz])
            pos += sz
        if pos < len(waveform):
            d.process(waveform[pos:])
        assert len(captured) == 1

    def test_chunked_preserves_packet_content(self):
        body = encode_ax25_ui('TEST', 'DEST', [], b'>chunked content')
        waveform = build_tx_waveform_from_body(body).tolist()
        captured = []
        d = AfskDemodulator(callback=captured.append)
        for i in range(0, len(waveform), 73):
            d.process(waveform[i:i + 73])
        assert len(captured) == 1
        pkt = captured[0]
        assert pkt.get('source') == 'TEST'
        assert pkt.get('raw_frame') == body


class TestMultiplePackets:
    def test_two_packets_decoded(self):
        info_text = '>packet'
        body = encode_ax25_ui('N0CALL', 'APZ010', [], info_text.encode('ascii'))
        wav1 = build_tx_waveform_from_body(body)
        wav2 = build_tx_waveform_from_body(body)
        combined = np.concatenate([wav1, wav2])
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(combined.tolist())
        assert len(captured) == 2

    def test_two_different_packets(self):
        body1 = encode_ax25_ui('CALL1', 'DST1', [], b'>first')
        body2 = encode_ax25_ui('CALL2', 'DST2', [], b'>second')
        wav1 = build_tx_waveform_from_body(body1)
        wav2 = build_tx_waveform_from_body(body2)
        combined = np.concatenate([wav1, wav2])
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(combined.tolist())
        assert len(captured) == 2
        assert captured[0].get('source') == 'CALL1'
        assert captured[1].get('source') == 'CALL2'

    def test_three_packets(self):
        body = encode_ax25_ui('N0CALL', 'APZ010', [], b'>data')
        wav = build_tx_waveform_from_body(body)
        combined = np.concatenate([wav, wav, wav])
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(combined.tolist())
        assert len(captured) == 3

    def test_all_have_correct_raw_frames(self):
        body = encode_ax25_ui('TEST', 'DEST', [], b'>multi')
        wav = build_tx_waveform_from_body(body)
        combined = np.concatenate([wav, wav])
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(combined.tolist())
        assert len(captured) == 2
        assert captured[0].get('raw_frame') == body
        assert captured[1].get('raw_frame') == body


class TestInvalidCRC:
    def test_corrupted_waveform_does_not_decode(self):
        body = encode_ax25_ui('N0CALL', 'APZ010', [], b'>test data')
        waveform = build_tx_waveform_from_body(body)
        payload_start = int(SR * SQUELCH_OPEN_MS / 1000) + PREAMBLE_FLAGS * 8 * SPB
        waveform[payload_start:payload_start + int(SR * 0.2)] = 0.0
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(waveform.tolist())
        assert len(captured) == 0

    def test_corrupted_samples_does_not_decode(self):
        body = encode_ax25_ui('TEST', 'DEST', [], b'>test')
        waveform = build_tx_waveform_from_body(body).copy()
        audio_start = int(SR * SQUELCH_OPEN_MS / 1000) + PREAMBLE_FLAGS * 8 * SPB
        audio_data = waveform[audio_start:]
        audio_data[:len(audio_data) // 2] = np.random.uniform(-0.5, 0.5, len(audio_data) // 2)
        waveform[audio_start:] = audio_data
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(waveform.tolist())
        assert len(captured) == 0

    def test_waveform_with_wrong_crc(self):
        body = encode_ax25_ui('TEST', 'DEST', [], b'>test')
        crc = crc_ccitt(body)
        wrong_crc = crc ^ 0xFFFF
        bad_frame = body + struct.pack('<H', wrong_crc)
        from kv4p_ht.afsk import (
            _bit_stuff,
            _bytes_to_bits_lsb,
            _flag_bits,
            _nrzi_encode,
        )
        flag_bits = _flag_bits()
        data_bits = _bytes_to_bits_lsb(bad_frame)
        stuffed = _bit_stuff(data_bits)
        nrz = _nrzi_encode(flag_bits + stuffed + flag_bits)
        total = len(nrz) * SPB
        waveform = np.empty(total, dtype=np.float32)
        freq_table = np.where(np.array(nrz) == 1, 1200.0, 2200.0)
        freqs = np.repeat(freq_table, 40)
        dt = 1.0 / 48000
        phase = 2.0 * np.pi * np.cumsum(freqs * dt)
        np.sin(phase, out=waveform)
        audio = waveform * 0.3
        lead = np.zeros(int(48000 * 1100 / 1000), dtype=np.float32)
        tail = np.zeros(int(48000 * 700 / 1000), dtype=np.float32)
        bad_waveform = np.concatenate([lead, audio, tail])
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(bad_waveform.tolist())
        assert len(captured) == 0

    def test_valid_then_corrupted_only_decodes_valid(self):
        body = encode_ax25_ui('N0CALL', 'APZ010', [], b'>good')
        valid_wav = build_tx_waveform_from_body(body)
        crc = crc_ccitt(body)
        wrong_crc = crc ^ 0xFFFF
        bad_frame = body + struct.pack('<H', wrong_crc)
        from kv4p_ht.afsk import (
            _bit_stuff,
            _bytes_to_bits_lsb,
            _flag_bits,
            _nrzi_encode,
        )
        flag_bits = _flag_bits()
        data_bits = _bytes_to_bits_lsb(bad_frame)
        stuffed = _bit_stuff(data_bits)
        nrz = _nrzi_encode(flag_bits + stuffed + flag_bits)
        total = len(nrz) * SPB
        w = np.empty(total, dtype=np.float32)
        freq_table = np.where(np.array(nrz) == 1, 1200.0, 2200.0)
        freqs = np.repeat(freq_table, 40)
        dt = 1.0 / 48000
        phase = 2.0 * np.pi * np.cumsum(freqs * dt)
        np.sin(phase, out=w)
        audio = w * 0.3
        lead = np.zeros(int(48000 * 1100 / 1000), dtype=np.float32)
        tail = np.zeros(int(48000 * 700 / 1000), dtype=np.float32)
        bad_wav = np.concatenate([lead, audio, tail])
        combined = np.concatenate([valid_wav, bad_wav])
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d.process(combined.tolist())
        assert len(captured) == 1


class TestShortFrame:
    def test_frame_less_than_18_resets(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d._frame = bytearray(14)
        d._crc = AX25_CRC_OK
        d._finalize()
        assert len(captured) == 0
        assert d._state == 'WAITING'

    def test_exactly_18_passes_length_check(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d._frame = bytearray(18)
        d._crc = AX25_CRC_OK
        d._frame[:] = b'\x00' * 18
        d._finalize()
        assert d._state == 'WAITING'

    def test_17_bytes_rejected_by_length(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d._frame = bytearray(17)
        d._crc = AX25_CRC_OK
        d._finalize()
        assert len(captured) == 0

    def test_0_bytes_rejected(self):
        captured = []
        d = AfskDemodulator(callback=captured.append)
        d._frame = bytearray()
        d._crc = AX25_CRC_OK
        d._finalize()
        assert len(captured) == 0

    def test_callback_not_called_for_short_correct_crc(self):
        called = []
        def cb(pkt):
            called.append(True)
        d = AfskDemodulator(callback=cb)
        d._frame = bytearray(15)
        d._crc = AX25_CRC_OK
        d._finalize()
        assert len(called) == 0


class TestPhaseAccumulator:
    def test_phase_increments_correctly(self):
        d = AfskDemodulator()
        for _ in range(100):
            d.process([0.0])
        expected_mark = 100 * 2.0 * math.pi * MARK_HZ / SR
        expected_space = 100 * 2.0 * math.pi * SPACE_HZ / SR
        assert abs(d._ph_mark - expected_mark % (2 * math.pi)) < 1e-10
        assert abs(d._ph_space - expected_space % (2 * math.pi)) < 1e-10

    def test_phase_wraps(self):
        d = AfskDemodulator()
        n = int(2 * math.pi / d._inc_mark) + 1
        for _ in range(n):
            d.process([0.0])
        assert d._ph_mark < 2.0 * math.pi


class TestJcorrWraparound:
    def test_jcorr_increments_and_wraps(self):
        d = AfskDemodulator()
        for i in range(SPB + 5):
            d.process([0.1])
        assert d._jcorr == 5

    def test_jcorr_starts_at_zero(self):
        d = AfskDemodulator()
        assert d._jcorr == 0

    def test_jcorr_wraps_at_spb(self):
        d = AfskDemodulator()
        for _ in range(SPB):
            d.process([0.1])
        assert d._jcorr == 0


class TestFdiffHistory:
    def test_fdiff_history_append(self):
        d = AfskDemodulator()
        d.process([0.1])
        assert len(d._fdiff_hist) == 1

    def test_fdiff_history_max_len(self):
        d = AfskDemodulator()
        for _ in range(10):
            d.process([0.1])
        assert len(d._fdiff_hist) == 3

    def test_fdiff_history_fifo(self):
        d = AfskDemodulator()
        for _ in range(5):
            d.process([0.1])
        assert len(d._fdiff_hist) == 3


class TestOnTransitionFlagSep:
    def test_flag_sep_cleared_on_flag(self):
        d = AfskDemodulator()
        d._flag_sep = True
        d._on_transition(7)
        assert d._flag_sep is False

    def test_flag_count_incremented_on_each_flag(self):
        d = AfskDemodulator()
        d._state = 'PRE_FLAG'
        d._flag_count = 3
        d._on_transition(7)
        assert d._flag_count == 4

    def test_flag_reset_by_non_single_data(self):
        d = AfskDemodulator()
        d._state = 'DECODING'
        d._flag_sep = True
        d._on_transition(2)
        assert d._flag_count == 0

    def test_flag_count_persists_across_pre_flag(self):
        d = AfskDemodulator()
        d._on_transition(7)
        assert d._state == 'PRE_FLAG'
        assert d._flag_count == 1
        d._on_transition(7)
        assert d._flag_count == 2


class TestNsampIncrement:
    def test_nsamp_increments_each_sample(self):
        d = AfskDemodulator()
        for i in range(100):
            d.process([0.0])
            assert d._nsamp == i + 1

    def test_nsamp_after_process_batch(self):
        d = AfskDemodulator()
        d.process([0.0] * 500)
        assert d._nsamp == 500
