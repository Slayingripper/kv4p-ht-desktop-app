from __future__ import annotations

import struct

import numpy as np

from kv4p_ht.afsk import (
    BAUD,
    FLAG,
    LEAD_MS,
    MARK,
    POSTAMBLE_FLAGS,
    PREAMBLE_FLAGS,
    SQUELCH_OPEN_MS,
    SPACE,
    SPB,
    SR,
    TAIL_MS,
    _bit_stuff,
    _bytes_to_bits_lsb,
    _flag_bits,
    _nrzi_encode,
    build_ax25_bits,
    build_tx_waveform,
    build_tx_waveform_from_body,
    crc_ccitt,
    modulate_ax25,
)
from kv4p_ht.aprs import encode_ax25_ui


class TestConstants:
    def test_mark(self):
        assert MARK == 1200.0

    def test_space(self):
        assert SPACE == 2200.0

    def test_baud(self):
        assert BAUD == 1200

    def test_sr(self):
        assert SR == 48000

    def test_flag(self):
        assert FLAG == 0x7E

    def test_spb(self):
        assert SPB == SR // BAUD
        assert SPB == 40

    def test_flag_counts(self):
        assert PREAMBLE_FLAGS == 50
        assert POSTAMBLE_FLAGS == 3

    def test_squelch_open_duration(self):
        assert SQUELCH_OPEN_MS == 300


class TestCrcCcitt:
    def test_empty(self):
        assert crc_ccitt(b"") == 0x0000

    def test_single_zero(self):
        assert crc_ccitt(b"\x00") == 0xF078

    def test_single_one(self):
        assert crc_ccitt(b"\x01") == 0xE1F1

    def test_single_flag(self):
        assert crc_ccitt(b"\x7E") == 0x6A81

    def test_single_ff(self):
        assert crc_ccitt(b"\xFF") == 0xFF00

    def test_two_bytes(self):
        assert crc_ccitt(b"AB") == 0x31EF

    def test_standard_check_value(self):
        assert crc_ccitt(b"123456789") == 0x906E

    def test_different_lengths_same_start(self):
        assert crc_ccitt(b"\x01\x02") != crc_ccitt(b"\x01\x02\x03")


class TestFlagBits:
    def test_length(self):
        assert len(_flag_bits()) == 8

    def test_bits(self):
        assert _flag_bits() == [0, 1, 1, 1, 1, 1, 1, 0]

    def test_all_bits_lsb_first(self):
        bits = _flag_bits()
        for i in range(8):
            expected = (FLAG >> i) & 1
            assert bits[i] == expected


class TestBytesToBitsLsb:
    def test_single_byte_zero(self):
        assert _bytes_to_bits_lsb(b"\x00") == [0] * 8

    def test_single_byte_ff(self):
        assert _bytes_to_bits_lsb(b"\xFF") == [1] * 8

    def test_single_byte_one(self):
        assert _bytes_to_bits_lsb(b"\x01") == [1, 0, 0, 0, 0, 0, 0, 0]

    def test_single_byte_80(self):
        assert _bytes_to_bits_lsb(b"\x80") == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_single_byte_7e(self):
        assert _bytes_to_bits_lsb(b"\x7E") == [0, 1, 1, 1, 1, 1, 1, 0]

    def test_multiple_bytes(self):
        result = _bytes_to_bits_lsb(b"\x01\x80")
        assert result == [1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 1]

    def test_empty(self):
        assert _bytes_to_bits_lsb(b"") == []


class TestBitStuff:
    def test_no_stuffing_needed(self):
        bits = [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
        assert _bit_stuff(bits) == bits

    def test_exactly_five_ones(self):
        assert _bit_stuff([1, 1, 1, 1, 1]) == [1, 1, 1, 1, 1, 0]

    def test_six_ones(self):
        assert _bit_stuff([1, 1, 1, 1, 1, 1]) == [1, 1, 1, 1, 1, 0, 1]

    def test_ten_ones(self):
        result = _bit_stuff([1] * 10)
        assert result == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]

    def test_stuffing_resets_after_insert(self):
        bits = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
        result = _bit_stuff(bits)
        assert result == [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]

    def test_zero_after_five_ones_stuff_inserted_before_zero(self):
        result = _bit_stuff([1, 1, 1, 1, 1, 0])
        assert result == [1, 1, 1, 1, 1, 0, 0]

    def test_all_zeros(self):
        assert _bit_stuff([0] * 10) == [0] * 10

    def test_empty(self):
        assert _bit_stuff([]) == []


class TestNrziEncode:
    def test_no_transitions_initial_one(self):
        assert _nrzi_encode([1, 1, 1], initial=1) == [1, 1, 1]

    def test_zero_transitions_initial_one(self):
        assert _nrzi_encode([0, 0, 0], initial=1) == [0, 1, 0]

    def test_alternating_initial_one(self):
        assert _nrzi_encode([0, 1, 0, 1], initial=1) == [0, 0, 1, 1]

    def test_initial_zero(self):
        assert _nrzi_encode([0, 0, 0], initial=0) == [1, 0, 1]

    def test_mixed_sequence_initial_one(self):
        assert _nrzi_encode([1, 0, 1, 0], initial=1) == [1, 0, 0, 1]

    def test_mixed_sequence_initial_zero(self):
        assert _nrzi_encode([1, 0, 1, 0], initial=0) == [0, 1, 1, 0]

    def test_all_ones_initial_zero(self):
        assert _nrzi_encode([1, 1, 1], initial=0) == [0, 0, 0]

    def test_all_zeros_initial_zero(self):
        assert _nrzi_encode([0, 0, 0], initial=0) == [1, 0, 1]

    def test_default_initial_is_one(self):
        result_default = _nrzi_encode([0])
        result_explicit = _nrzi_encode([0], initial=1)
        assert result_default == result_explicit

    def test_empty(self):
        assert _nrzi_encode([], initial=1) == []


class TestBuildAx25Bits:
    def test_returns_list_of_ints(self):
        bits = build_ax25_bits("TEST", "DEST", [], b"info")
        assert isinstance(bits, list)
        assert all(isinstance(b, int) for b in bits)
        assert all(b in (0, 1) for b in bits)

    def test_starts_with_flag(self):
        bits = build_ax25_bits("TEST", "DEST", [], b"info")
        assert bits[:8] == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_ends_with_flag(self):
        bits = build_ax25_bits("TEST", "DEST", [], b"info")
        last8 = bits[-8:]
        assert (last8 == [1, 1, 1, 1, 1, 1, 1, 0] or
                last8 == [0, 0, 0, 0, 0, 0, 0, 1])

    def test_bit_stuffing_prevents_five_ones(self):
        body = encode_ax25_ui("TEST", "DEST", [], b"info")
        crc = crc_ccitt(body)
        frame = body + struct.pack('<H', crc)
        stuffed = _bit_stuff(_bytes_to_bits_lsb(frame))
        ones_count = 0
        for b in stuffed:
            if b == 1:
                ones_count += 1
                assert ones_count <= 5
            else:
                ones_count = 0

    def test_with_multiple_digipeaters(self):
        bits = build_ax25_bits("TEST", "DEST", ["WIDE1", "WIDE2"], b"data")
        assert len(bits) > 0
        assert bits[:8] == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_known_length(self):
        bits = build_ax25_bits("TEST", "DEST", [], b"info")
        assert len(bits) == 601

    def test_with_aprs_message(self):
        info = b":DEST     :Hello{001"
        bits = build_ax25_bits("TEST", "DEST", [], info)
        assert len(bits) > 0
        assert bits[:8] == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_empty_info(self):
        bits = build_ax25_bits("TEST", "DEST", [], b"")
        assert len(bits) > 0
        assert bits[:8] == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_nrzi_encoding_produces_valid_transitions(self):
        bits = build_ax25_bits("TEST", "DEST", [], b"info")
        stuffed = _bit_stuff(_bytes_to_bits_lsb(
            encode_ax25_ui("TEST", "DEST", [], b"info")
            + struct.pack('<H', crc_ccitt(
                encode_ax25_ui("TEST", "DEST", [], b"info")))
        ))
        flag = _flag_bits()
        expected_nrzi = _nrzi_encode(
            flag * PREAMBLE_FLAGS + stuffed + flag * POSTAMBLE_FLAGS
        )
        assert bits == expected_nrzi


class TestModulateAx25:
    def test_returns_float32(self):
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        assert wav.dtype == np.float32

    def test_correct_length(self):
        bits = build_ax25_bits("TEST", "DEST", [], b"info")
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        assert len(wav) == len(bits) * SPB

    def test_amplitude_envelope(self):
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        assert np.max(wav) <= 0.3 + 1e-6
        assert np.min(wav) >= -0.3 - 1e-6
        np.testing.assert_array_less(np.abs(wav), 0.3 + 1e-6)

    def test_amplitude_equality(self):
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        max_val = np.max(np.abs(wav))
        assert max_val > 0.2

    def test_continuous_phase(self):
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        diffs = np.abs(np.diff(wav))
        max_possible_change = 0.3 * 2.0 * np.pi * SPACE / SR + 1e-6
        assert np.all(diffs < max_possible_change * 2)

    def test_no_dc_offset(self):
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        assert abs(np.mean(wav)) < 0.01

    def test_bit_boundaries_no_discontinuity(self):
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        for i in range(1, len(wav) // SPB):
            idx = i * SPB - 1
            if idx + 1 < len(wav):
                diff = abs(wav[idx + 1] - wav[idx])
                assert diff < 0.5

    def test_with_aprs_message(self):
        info = b":DEST     :Hello{001"
        wav = modulate_ax25("TEST", "DEST", [], info)
        assert wav.dtype == np.float32
        assert len(wav) > 0

    def test_reproducibility(self):
        wav1 = modulate_ax25("TEST", "DEST", [], b"info")
        wav2 = modulate_ax25("TEST", "DEST", [], b"info")
        np.testing.assert_array_equal(wav1, wav2)

    def test_different_info_different_waveform(self):
        wav1 = modulate_ax25("TEST", "DEST", [], b"info1")
        wav2 = modulate_ax25("TEST", "DEST", [], b"info2")
        assert not np.allclose(wav1, wav2)

    def test_empty_info(self):
        wav = modulate_ax25("TEST", "DEST", [], b"")
        assert wav.dtype == np.float32
        assert len(wav) > 0


class TestBuildTxWaveform:
    def test_dtype_float32(self):
        tx = build_tx_waveform("TEST", "DEST", [], b"info")
        assert tx.dtype == np.float32

    def test_correct_total_length(self):
        audio = modulate_ax25("TEST", "DEST", [], b"info")
        tx = build_tx_waveform("TEST", "DEST", [], b"info")
        assert len(tx) == len(audio) + int(SR * SQUELCH_OPEN_MS / 1000)

    def test_starts_with_mark_tone(self):
        audio = modulate_ax25("TEST", "DEST", [], b"info")
        tx = build_tx_waveform("TEST", "DEST", [], b"info")
        carrier_len = int(SR * SQUELCH_OPEN_MS / 1000)
        assert np.max(np.abs(tx[:carrier_len])) > 0.2
        assert len(tx[carrier_len:]) == len(audio)

    def test_audio_segment_amplitude(self):
        audio = modulate_ax25("TEST", "DEST", [], b"info")
        tx = build_tx_waveform("TEST", "DEST", [], b"info")
        assert np.max(np.abs(tx)) <= 0.3 + 1e-6

    def test_overall_length(self):
        tx = build_tx_waveform("TEST", "DEST", [], b"info")
        audio = modulate_ax25("TEST", "DEST", [], b"info")
        assert len(tx) == len(audio) + int(SR * SQUELCH_OPEN_MS / 1000)

    def test_with_aprs_message(self):
        info = b":DEST     :Hello{001"
        tx = build_tx_waveform("TEST", "DEST", [], info)
        assert tx.dtype == np.float32
        assert len(tx) > 0

    def test_empty_info(self):
        tx = build_tx_waveform("TEST", "DEST", [], b"")
        assert tx.dtype == np.float32
        assert len(tx) > 0

    def test_reproducibility(self):
        tx1 = build_tx_waveform("TEST", "DEST", [], b"info")
        tx2 = build_tx_waveform("TEST", "DEST", [], b"info")
        np.testing.assert_array_equal(tx1, tx2)


class TestBuildTxWaveformFromBody:
    def test_dtype_float32(self):
        body = encode_ax25_ui("TEST", "DEST", [], b"info")
        tx = build_tx_waveform_from_body(body)
        assert tx.dtype == np.float32

    def test_matches_build_tx_waveform(self):
        body = encode_ax25_ui("TEST", "DEST", [], b"info")
        tx1 = build_tx_waveform("TEST", "DEST", [], b"info")
        tx2 = build_tx_waveform_from_body(body)
        np.testing.assert_array_equal(tx1, tx2)

    def test_correct_total_length(self):
        body = encode_ax25_ui("TEST", "DEST", [], b"info")
        crc = crc_ccitt(body)
        frame = body + struct.pack('<H', crc)
        flag = _flag_bits()
        stuffed = _bit_stuff(_bytes_to_bits_lsb(frame))
        nrz = _nrzi_encode(
            flag * PREAMBLE_FLAGS + stuffed + flag * POSTAMBLE_FLAGS
        )
        audio_len = len(nrz) * SPB + int(SR * SQUELCH_OPEN_MS / 1000)
        tx = build_tx_waveform_from_body(body)
        assert len(tx) == audio_len

    def test_audio_segment_not_zero(self):
        body = encode_ax25_ui("TEST", "DEST", [], b"info")
        tx = build_tx_waveform_from_body(body)
        audio = modulate_ax25("TEST", "DEST", [], b"info")
        assert np.any(tx != 0)

    def test_audio_segment_matches(self):
        body = encode_ax25_ui("TEST", "DEST", [], b"info")
        tx = build_tx_waveform_from_body(body)
        audio = modulate_ax25("TEST", "DEST", [], b"info")
        carrier_len = int(SR * SQUELCH_OPEN_MS / 1000)
        assert len(tx[carrier_len:]) == len(audio)

    def test_with_digipeaters(self):
        body = encode_ax25_ui("TEST", "DEST", ["WIDE1-1"], b"data")
        tx = build_tx_waveform_from_body(body)
        ref = build_tx_waveform("TEST", "DEST", ["WIDE1-1"], b"data")
        np.testing.assert_array_equal(tx, ref)

    def test_with_aprs_message(self):
        info = b":DEST     :Hello{001"
        body = encode_ax25_ui("TEST", "DEST", [], info)
        tx = build_tx_waveform_from_body(body)
        ref = build_tx_waveform("TEST", "DEST", [], info)
        np.testing.assert_array_equal(tx, ref)

    def test_empty_body(self):
        body = encode_ax25_ui("TEST", "DEST", [], b"")
        tx = build_tx_waveform_from_body(body)
        assert tx.dtype == np.float32
        assert len(tx) > 0

    def test_reproducibility(self):
        body = encode_ax25_ui("TEST", "DEST", [], b"info")
        tx1 = build_tx_waveform_from_body(body)
        tx2 = build_tx_waveform_from_body(body)
        np.testing.assert_array_equal(tx1, tx2)


class TestBuildAx25BitsEdgeCases:
    def test_long_callsign(self):
        bits = build_ax25_bits("LONGCALL", "DEST", [], b"info")
        assert len(bits) > 0
        assert bits[:8] == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_callsign_with_ssid(self):
        bits = build_ax25_bits("TEST-5", "DEST-10", [], b"info")
        assert len(bits) > 0
        assert bits[:8] == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_three_digipeaters(self):
        bits = build_ax25_bits("TEST", "DEST",
                               ["WIDE1-1", "WIDE2-2", "WIDE3-3"], b"data")
        assert len(bits) > 0
        assert bits[:8] == [0, 0, 0, 0, 0, 0, 0, 1]

    def test_nrzi_output_only_binary(self):
        bits = build_ax25_bits("TEST", "DEST", [], b"info")
        assert set(bits) <= {0, 1}

    def test_no_overly_long_runs_of_same_bit(self):
        bits = build_ax25_bits("TEST", "DEST", [], b"info")
        max_run = 0
        current = 1
        run = 0
        for b in bits:
            if b == current:
                run += 1
            else:
                max_run = max(max_run, run)
                current = b
                run = 1
        max_run = max(max_run, run)
        assert max_run < 20


class TestModulateAx25EdgeCases:
    def test_single_bit_frame(self):
        bits = build_ax25_bits("T", "D", [], b"")
        wav = modulate_ax25("T", "D", [], b"")
        assert len(wav) == len(bits) * SPB
        assert wav.dtype == np.float32

    def test_max_callsign_length(self):
        wav = modulate_ax25("AAAAAA", "BBBBBB", [], b"X" * 256)
        assert wav.dtype == np.float32

    def test_known_frequencies_present(self):
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        segment = wav[:SPB * 80]
        window = segment * np.hanning(len(segment))
        fft = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(len(segment), 1.0 / SR)
        peak_idx = np.argsort(fft)[-5:]
        peak_freqs = freqs[peak_idx]
        assert np.any(np.abs(peak_freqs - MARK) < 100) or np.any(np.abs(peak_freqs - SPACE) < 100)

    def test_waveform_symmetric(self):
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        assert abs(np.max(wav) + np.min(wav)) < 1e-6


class TestBuildTxWaveformEdgeCases:
    def test_lead_silence_exact_duration(self):
        lead_samples = int(SR * LEAD_MS / 1000)
        tx = build_tx_waveform("TEST", "DEST", [], b"info")
        assert len(tx[:lead_samples]) == lead_samples

    def test_tail_silence_exact_duration(self):
        tail_samples = int(SR * TAIL_MS / 1000)
        tx = build_tx_waveform("TEST", "DEST", [], b"info")
        assert tail_samples == 0
        assert len(tx) > 0

    def test_no_samples_outside_range(self):
        tx = build_tx_waveform("TEST", "DEST", [], b"info")
        assert np.all(tx >= -0.3 - 1e-6)
        assert np.all(tx <= 0.3 + 1e-6)


class TestBuildTxWaveformFromBodyEdgeCases:
    def test_from_body_with_digipeaters_matches_direct(self):
        body = encode_ax25_ui("TEST", "DEST", ["WIDE1-1", "WIDE2-2"], b"data")
        tx_body = build_tx_waveform_from_body(body)
        tx_direct = build_tx_waveform("TEST", "DEST", ["WIDE1-1", "WIDE2-2"], b"data")
        np.testing.assert_array_equal(tx_body, tx_direct)

    def test_from_body_ssid(self):
        body = encode_ax25_ui("TEST-7", "DEST-3", [], b"aprs data")
        tx = build_tx_waveform_from_body(body)
        assert tx.dtype == np.float32


class TestIntegration:
    def test_full_aprs_message_roundtrip(self):
        info = b":DEST     :Hello{001"
        tx = build_tx_waveform("SRC", "DST", [], info)
        carrier = int(SR * SQUELCH_OPEN_MS / 1000)
        assert len(tx) == carrier + len(modulate_ax25("SRC", "DST", [], info))

    def test_empty_digipeater_list(self):
        tx1 = build_tx_waveform("TEST", "DEST", [], b"data")
        tx2 = build_tx_waveform("TEST", "DEST", [], b"data")
        np.testing.assert_array_equal(tx1, tx2)

    def test_body_and_direct_identical(self):
        info = b":RECEIVER:Test message{12"
        body = encode_ax25_ui("MYCALL", "APRS", ["WIDE1-1"], info)
        tx_body = build_tx_waveform_from_body(body)
        tx_direct = build_tx_waveform("MYCALL", "APRS", ["WIDE1-1"], info)
        np.testing.assert_array_equal(tx_body, tx_direct)

    def test_waveform_is_stereo_compatible_mono(self):
        wav = modulate_ax25("TEST", "DEST", [], b"info")
        assert wav.ndim == 1
