from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

from kv4p_ht.aprs import (
    Digipeater,
    IGate,
    _encode_addr,
    decode_address,
    decode_ax25_frame,
    encode_ax25_ui,
    format_ack,
    format_beacon,
    format_message,
    parse_aprs,
)


class TestEncodeAddr:
    def test_basic_callsign_last(self):
        result = _encode_addr("KG4SHR", True)
        assert len(result) == 7
        assert result[0] == ord('K') << 1
        assert result[1] == ord('G') << 1
        assert result[2] == ord('4') << 1
        assert result[3] == ord('S') << 1
        assert result[4] == ord('H') << 1
        assert result[5] == ord('R') << 1
        assert result[6] == 0x61

    def test_basic_callsign_not_last(self):
        result = _encode_addr("KG4SHR", False)
        assert len(result) == 7
        assert result[6] == 0x60

    def test_callsign_with_ssid(self):
        result = _encode_addr("KG4SHR-7", True)
        ssid_field = result[6]
        assert (ssid_field >> 1) & 0x0F == 7
        assert ssid_field & 0x01 == 1

    def test_callsign_with_ssid_not_last(self):
        result = _encode_addr("KG4SHR-7", False)
        ssid_field = result[6]
        assert (ssid_field >> 1) & 0x0F == 7
        assert ssid_field & 0x01 == 0

    def test_six_char_callsign_padding(self):
        result = _encode_addr("A", False)
        assert result[0] == ord('A') << 1
        assert result[1] == ord(' ') << 1
        assert result[2] == ord(' ') << 1
        assert result[3] == ord(' ') << 1
        assert result[4] == ord(' ') << 1
        assert result[5] == ord(' ') << 1

    def test_ssid_bit_encoding(self):
        for ssid in range(16):
            result = _encode_addr(f"TEST-{ssid}", True)
            assert (result[6] >> 1) & 0x0F == ssid

    def test_last_flag_bit_zero(self):
        result = _encode_addr("TEST", False)
        assert result[6] & 0x01 == 0

    def test_last_flag_bit_one(self):
        result = _encode_addr("TEST", True)
        assert result[6] & 0x01 == 1

    def test_ssid_reserved_bits(self):
        result = _encode_addr("TEST", False)
        assert result[6] & 0x60 == 0x60

    def test_all_ssids_with_last_flag(self):
        for ssid in range(16):
            result = _encode_addr(f"TEST-{ssid}", True)
            expected = 0x60 | (ssid << 1) | 0x01
            assert result[6] == expected

    def test_all_ssids_without_last_flag(self):
        for ssid in range(16):
            result = _encode_addr(f"TEST-{ssid}", False)
            expected = 0x60 | (ssid << 1)
            assert result[6] == expected

    def test_callsign_shifted_left_by_one(self):
        result = _encode_addr("ABCDEF", False)
        for i, ch in enumerate("ABCDEF"):
            assert result[i] == ord(ch) << 1


class TestEncodeAx25Ui:
    def test_encode_no_digipeaters(self):
        frame = encode_ax25_ui("SRC", "DST", [], b"hello")
        dest = _encode_addr("DST", False)
        src = _encode_addr("SRC", True)
        expected = dest + src + b'\x03\xf0' + b'hello'
        assert frame == expected

    def test_encode_with_digipeaters(self):
        digis = ["WIDE1-1", "WIDE2-2"]
        frame = encode_ax25_ui("SRC", "DST", digis, b"data")
        dest = _encode_addr("DST", False)
        src = _encode_addr("SRC", False)
        d1 = _encode_addr("WIDE1-1", False)
        d2 = _encode_addr("WIDE2-2", True)
        expected = dest + src + d1 + d2 + b'\x03\xf0' + b'data'
        assert frame == expected

    def test_control_byte_ui(self):
        frame = encode_ax25_ui("SRC", "DST", [], b"")
        assert frame[-2] == 0x03

    def test_pid_byte_no_layer3(self):
        frame = encode_ax25_ui("SRC", "DST", [], b"")
        assert frame[-1] == 0xF0

    def test_info_bytes_appended_at_end(self):
        info = b"INFO DATA HERE"
        frame = encode_ax25_ui("SRC", "DST", [], info)
        assert frame[-len(info):] == info

    def test_address_ordering(self):
        frame = encode_ax25_ui("SRC", "DST", ["D1", "D2"], b"x")
        dest = _encode_addr("DST", False)
        src = _encode_addr("SRC", False)
        d1 = _encode_addr("D1", False)
        d2 = _encode_addr("D2", True)
        assert frame[:7] == dest
        assert frame[7:14] == src
        assert frame[14:21] == d1
        assert frame[21:28] == d2

    def test_single_digipeater_last_flag(self):
        frame = encode_ax25_ui("SRC", "DST", ["D1"], b"x")
        dp_byte = _encode_addr("D1", True)
        assert frame[14:21] == dp_byte

    def test_empty_info_field(self):
        frame = encode_ax25_ui("SRC", "DST", [], b"")
        assert len(frame) == 16

    def test_ssid_in_callsigns(self):
        frame = encode_ax25_ui("SRC-5", "DST-10", [], b"hi")
        dest = _encode_addr("DST-10", False)
        src = _encode_addr("SRC-5", True)
        assert frame[:7] == dest
        assert frame[7:14] == src


class TestDecodeAddress:
    def test_basic_callsign(self):
        addr = _encode_addr("KG4SHR", True)
        assert decode_address(addr) == "KG4SHR"

    def test_with_ssid(self):
        addr = _encode_addr("KG4SHR-7", True)
        assert decode_address(addr) == "KG4SHR-7"

    def test_strip_padding(self):
        addr = _encode_addr("A", True)
        assert decode_address(addr) == "A"

    def test_all_ssids_round_trip(self):
        for ssid in range(16):
            addr = _encode_addr(f"TEST-{ssid}", True)
            expected = f"TEST-{ssid}" if ssid else "TEST"
            assert decode_address(addr) == expected

    def test_six_char_callsign(self):
        addr = _encode_addr("ABCDEF", True)
        assert decode_address(addr) == "ABCDEF"

    def test_zero_ssid_omitted(self):
        addr = _encode_addr("TEST-0", True)
        assert decode_address(addr) == "TEST"


class TestDecodeAx25Frame:
    def test_round_trip_no_digis(self):
        info = b":DEST     :Hello{001"
        original = encode_ax25_ui("SRC", "DST", [], info)
        dec = decode_ax25_frame(original)
        assert dec['source'] == "SRC"
        assert dec['destination'] == "DST"
        assert dec['digipeaters'] == []
        assert dec['control'] == 0x03
        assert dec['pid'] == 0xF0
        assert dec['info'] == info

    def test_round_trip_with_digis(self):
        info = b"position data"
        original = encode_ax25_ui("SRC", "DST", ["WIDE1-1", "WIDE2-2"], info)
        dec = decode_ax25_frame(original)
        assert dec['source'] == "SRC"
        assert dec['destination'] == "DST"
        assert dec['digipeaters'] == ["WIDE1-1", "WIDE2-2"]
        assert dec['control'] == 0x03
        assert dec['pid'] == 0xF0
        assert dec['info'] == info

    def test_short_frame_returns_raw(self):
        result = decode_ax25_frame(b'\x00' * 13)
        assert result == {'raw': b'\x00' * 13}

    def test_empty_frame_returns_raw(self):
        result = decode_ax25_frame(b'')
        assert result == {'raw': b''}

    def test_frame_with_ssid_callsigns(self):
        info = b"test"
        original = encode_ax25_ui("SRC-7", "DST-3", ["WIDE1-5"], info)
        dec = decode_ax25_frame(original)
        assert dec['source'] == "SRC-7"
        assert dec['destination'] == "DST-3"
        assert dec['digipeaters'] == ["WIDE1-5"]

    def test_control_and_pid_extraction(self):
        frame = encode_ax25_ui("A", "B", [], b"x")
        dec = decode_ax25_frame(frame)
        assert dec['control'] == 0x03
        assert dec['pid'] == 0xF0

    def test_info_field_extraction(self):
        info = b"The quick brown fox"
        frame = encode_ax25_ui("A", "B", [], info)
        dec = decode_ax25_frame(frame)
        assert dec['info'] == info

    def test_info_field_empty(self):
        frame = encode_ax25_ui("A", "B", [], b"")
        frame = frame[:-2] + b'\x03\xf0'
        dec = decode_ax25_frame(frame)
        assert dec['info'] == b''

    def test_boundary_len_14(self):
        frame = _encode_addr("DST", False) + _encode_addr("SRC", True) + b'\x03'
        frame = frame[:14]
        result = decode_ax25_frame(frame)
        assert 'raw' not in result or 'source' in result


class TestParseAprs:
    def test_message_type(self):
        result = parse_aprs(":DEST     :Hello world", "MYCALL")
        assert result['type'] == 'message'
        assert result['addressee'] == 'DEST'
        assert result['text'] == 'Hello world'
        assert result['msg_id'] == ''
        assert result['source'] == 'MYCALL'

    def test_message_with_msg_id(self):
        result = parse_aprs(":DEST     :Hello{001", "SRC")
        assert result['type'] == 'message'
        assert result['text'] == 'Hello'
        assert result['msg_id'] == '001'

    def test_message_with_padding(self):
        result = parse_aprs(":DEST     :Hello world")
        assert result['addressee'] == 'DEST'

    def test_position_eq(self):
        result = parse_aprs("=4237.50N/07102.30W-Comment")
        assert result['type'] == 'position'

    def test_position_bang(self):
        result = parse_aprs("!4237.50N/07102.30W-")
        assert result['type'] == 'position'

    def test_position_at(self):
        result = parse_aprs("@4237.50N/07102.30W-Comment")
        assert result['type'] == 'position'

    def test_position_slash(self):
        result = parse_aprs("/4237.50N/07102.30W-Comment")
        assert result['type'] == 'position'

    def test_position_type_and_comment(self):
        result = parse_aprs("=4237.50N/07102.30W-Some comment here")
        assert result['type'] == 'position'
        assert result['raw_position'] == '4237.50N/07102.30W-'
        assert result['comment'] == 'ome comment here'

    def test_position_no_comment(self):
        result = parse_aprs("=4237.50N/07102.30W-")
        assert result['comment'] == ''

    def test_position_short_comment(self):
        result = parse_aprs("=4237.50N/07102.30W-")
        assert result['raw_position'] == '4237.50N/07102.30W-'
        assert result['comment'] == ''

    def test_status(self):
        result = parse_aprs(">Status text here")
        assert result['type'] == 'status'
        assert result['text'] == 'Status text here'

    def test_weather(self):
        result = parse_aprs("_4237.50N/07102.30W...data")
        assert result['type'] == 'weather'

    def test_object(self):
        result = parse_aprs(";OBJECT  *111111z4237.50N/07102.30W...")
        assert result['type'] == 'object'

    def test_telemetry(self):
        result = parse_aprs("T#123,1,2,3,4,5,6,7,8")
        assert result['type'] == 'telemetry'

    def test_beacon(self):
        result = parse_aprs("Some beacon text")
        assert result['type'] == 'beacon'
        assert result['text'] == 'Some beacon text'

    def test_empty_input(self):
        result = parse_aprs("", "MYCALL")
        assert result['source'] == 'MYCALL'
        assert result['raw'] == ''

    def test_source_propagation(self):
        result = parse_aprs(">status", "SOURCE")
        assert result['source'] == 'SOURCE'

    def test_message_addressee_strip(self):
        result = parse_aprs(":DEST    :Hello")
        assert result['addressee'] == 'DEST'


class TestFormatBeacon:
    def test_positive_lat_lon(self):
        result = format_beacon("TEST", 42.3750, -71.0230, "comment")
        assert "N" in result
        assert "W" in result
        assert result.startswith("=")

    def test_negative_lat_positive_lon(self):
        result = format_beacon("TEST", -33.86, 151.21, "down under")
        assert "S" in result
        assert "E" in result
        assert "down under" in result

    def test_negative_lat_negative_lon(self):
        result = format_beacon("TEST", -45.0, -90.0, "test")
        assert "S" in result
        assert "W" in result

    def test_custom_comment(self):
        result = format_beacon("TEST", 0.0, 0.0, "my comment")
        assert result.endswith("my comment")

    def test_custom_symbol(self):
        result = format_beacon("TEST", 0.0, 0.0, "", "/\\")
        assert "/" in result
        assert "\\" in result

    def test_lat_format_two_decimal_minutes(self):
        result = format_beacon("TEST", 42.3750, -71.0230)
        lat_min_part = result[3:8]
        assert lat_min_part == '22.50'

    def test_lon_format_two_decimal_minutes(self):
        result = format_beacon("TEST", 42.3750, -71.0230)
        lon_min_part = result[13:18]
        assert lon_min_part == '01.38'

    def test_known_position(self):
        result = format_beacon("TEST", 42.375, -71.023)
        assert result.startswith("=4222.50N/07101.38W")

    def test_zero_coordinates(self):
        result = format_beacon("TEST", 0.0, 0.0, "origin")
        assert "N" in result
        assert "E" in result
        assert "origin" in result


class TestFormatMessage:
    def test_without_msg_id(self):
        result = format_message("DEST", "Hello")
        assert result == ":DEST     :Hello"

    def test_with_msg_id(self):
        result = format_message("DEST", "Hello", "001")
        assert result == ":DEST     :Hello{001"

    def test_max_length_truncation(self):
        long_text = "X" * 300
        result = format_message("DEST", long_text, "001")
        assert len(result) == 256

    def test_dest_callsign_padding(self):
        result = format_message("ABC", "test")
        assert result[0] == ':'
        assert result[1:4] == 'ABC'
        assert result[4:10] == '      '
        assert result[10] == ':'

    def test_dest_callsign_exactly_9(self):
        result = format_message("123456789", "test")
        assert result[1:10] == "123456789"
        assert result[10] == ':'

    def test_truncation_without_msg_id(self):
        long_text = "Y" * 300
        result = format_message("DEST", long_text)
        assert len(result) == 256


class TestFormatAck:
    def test_correct_format(self):
        result = format_ack("DEST", "001")
        assert result == ":DEST     :ack001"

    def test_with_addressee_padding(self):
        result = format_ack("A", "5")
        assert result[1:4] == "A  "
        assert result == ":A        :ack5"

    def test_with_long_addressee(self):
        result = format_ack("ABCDEFGHI", "999")
        assert result == ":ABCDEFGHI:ack999"


class TestDigipeater:
    def test_process_returns_true_on_new_packet(self):
        tx_cb = Mock()
        dp = Digipeater("MYCALL", tx_cb)
        frame = encode_ax25_ui("SRC", "DST", [], b"info")
        assert dp.process(frame) is True
        tx_cb.assert_called_once()

    def test_process_dedup_within_30_seconds(self):
        tx_cb = Mock()
        dp = Digipeater("MYCALL", tx_cb)
        frame = encode_ax25_ui("SRC", "DST", [], b"info")
        assert dp.process(frame) is True
        assert dp.process(frame) is False
        assert tx_cb.call_count == 1

    def test_dedup_window_expiry(self):
        fake_time = 1000.0

        def mock_time():
            return fake_time

        with patch('time.time', mock_time):
            tx_cb = Mock()
            dp = Digipeater("MYCALL", tx_cb)
            frame = encode_ax25_ui("SRC", "DST", [], b"info")
            assert dp.process(frame) is True
            fake_time += 31.0
            assert dp.process(frame) is True
            assert tx_cb.call_count == 2

    def test_skip_if_our_callsign_in_path(self):
        tx_cb = Mock()
        dp = Digipeater("MYCALL", tx_cb)
        frame = encode_ax25_ui("SRC", "DST", ["MYCALL*"], b"info")
        assert dp.process(frame) is False
        tx_cb.assert_not_called()

    def test_skip_if_our_callsign_without_star_in_path(self):
        tx_cb = Mock()
        dp = Digipeater("MYCALL", tx_cb)
        frame = encode_ax25_ui("SRC", "DST", ["MYCALL"], b"info")
        assert dp.process(frame) is False
        tx_cb.assert_not_called()

    def test_skip_on_decode_errors(self):
        tx_cb = Mock()
        dp = Digipeater("MYCALL", tx_cb)
        result = dp.process(b'\x00')
        assert result is False
        tx_cb.assert_not_called()

    def test_skip_on_empty_frame(self):
        tx_cb = Mock()
        dp = Digipeater("MYCALL", tx_cb)
        result = dp.process(b'')
        assert result is False
        tx_cb.assert_not_called()

    def test_tx_callback_called_with_reencoded_frame(self):
        tx_cb = Mock()
        dp = Digipeater("MYCALL", tx_cb)
        info = b"test info"
        frame = encode_ax25_ui("SRC", "DST", [], info)
        dp.process(frame)
        expected = encode_ax25_ui("SRC", "DST", ["MYCALL*"], info)
        tx_cb.assert_called_once_with(expected)

    def test_tx_callback_with_existing_digis(self):
        tx_cb = Mock()
        dp = Digipeater("MYCALL", tx_cb)
        info = b"data"
        frame = encode_ax25_ui("SRC", "DST", ["WIDE1-1"], info)
        dp.process(frame)
        expected = encode_ax25_ui("SRC", "DST", ["WIDE1-1", "MYCALL*"], info)
        tx_cb.assert_called_once_with(expected)

    def test_tx_callback_exception_returns_false(self):
        tx_cb = Mock(side_effect=Exception("TX failed"))
        dp = Digipeater("MYCALL", tx_cb)
        frame = encode_ax25_ui("SRC", "DST", [], b"info")
        assert dp.process(frame) is False

    def test_heard_dict_maintenance(self):
        fake_time = 1000.0

        def mock_time():
            return fake_time

        with patch('time.time', mock_time):
            tx_cb = Mock()
            dp = Digipeater("MYCALL", tx_cb)
            f1 = encode_ax25_ui("SRC1", "DST1", [], b"a")
            f2 = encode_ax25_ui("SRC2", "DST2", [], b"b")
            dp.process(f1)
            fake_time += 31.0
            dp.process(f2)
            assert dp._heard == {("SRC2", "DST2"): fake_time}

    def test_callsign_case_insensitive_in_path(self):
        tx_cb = Mock()
        dp = Digipeater("mycall", tx_cb)
        frame = encode_ax25_ui("SRC", "DST", ["MYCALL*"], b"info")
        assert dp.process(frame) is False
        tx_cb.assert_not_called()


class TestIGate:
    @patch('time.sleep', return_value=None)
    @patch('socket.create_connection')
    def test_send_to_is_queue(self, mock_create_conn, mock_sleep):
        mock_sock = MagicMock()
        mock_file = MagicMock()
        mock_sock.makefile.return_value = mock_file
        mock_create_conn.return_value = mock_sock

        callback = MagicMock()
        igate = IGate("MYCALL", lambda x, **kw: None, callback, tx_enabled=False)
        igate.send_to_is("SRC", ":DEST     :Hello")

        read_calls = [0]
        def on_readline():
            read_calls[0] += 1
            igate._running = False
            return ''

        mock_file.readline.side_effect = on_readline
        igate.run()

        expected_line = "SRC>APZ010,TCPIP*::DEST     :Hello\r\n"
        mock_sock.sendall.assert_any_call(expected_line.encode())

    def test_stop_signals_thread(self):
        igate = IGate("MYCALL", lambda x, **kw: None, lambda x: None, tx_enabled=False)
        assert igate._running is True
        igate.stop()
        assert igate._running is False

    @patch('time.sleep', return_value=None)
    @patch('socket.create_connection')
    def test_is_queue_send_multiple(self, mock_create_conn, mock_sleep):
        mock_sock = MagicMock()
        mock_file = MagicMock()
        mock_sock.makefile.return_value = mock_file
        mock_create_conn.return_value = mock_sock

        callback = MagicMock()
        igate = IGate("MYCALL", lambda x, **kw: None, callback, tx_enabled=False)

        igate.send_to_is("SRC1", ">status1")
        igate.send_to_is("SRC2", ">status2")

        read_calls = [0]
        def on_readline():
            read_calls[0] += 1
            if read_calls[0] >= 2:
                igate._running = False
            return ''

        mock_file.readline.side_effect = on_readline
        igate.run()

        expected_1 = "SRC1>APZ010,TCPIP*:>status1\r\n"
        expected_2 = "SRC2>APZ010,TCPIP*:>status2\r\n"
        mock_sock.sendall.assert_any_call(expected_1.encode())
        mock_sock.sendall.assert_any_call(expected_2.encode())

    def test_queue_directly(self):
        igate = IGate("MYCALL", lambda x, **kw: None, lambda x: None, tx_enabled=False)
        igate.send_to_is("SRC", ":DEST     :Hello")
        item = igate._to_is_queue.get_nowait()
        assert item == ("SRC", ":DEST     :Hello")
        assert igate._to_is_queue.empty()

    def test_stop_puts_none_on_queue(self):
        igate = IGate("MYCALL", lambda x, **kw: None, lambda x: None, tx_enabled=False)
        igate.stop()
        assert igate._to_is_queue.get_nowait() is None
