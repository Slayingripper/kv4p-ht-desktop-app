"""
AX.25 file transfer protocol.

Sends and receives files over AX.25 connected-mode or UI frames with
packet sequencing, acknowledgements, and retransmission.

Protocol:
  - Header: FILE_START<filename><size><checksum> or FILE_DATA<seq><len>
  - Data packets with 16-bit sequence numbers
  - ACK/NAK for each packet
  - CRC-16 per packet
  - Automatic retry with configurable timeout
"""
from __future__ import annotations

import hashlib
import os
import queue
import struct
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .aprs import encode_ax25_ui, decode_ax25_frame

# ── Protocol constants ────────────────────────────────────────────

PROTO_ID = b'KF'  # KV4P File transfer
CMD_FILE_START = 0x01
CMD_FILE_DATA = 0x02
CMD_FILE_END = 0x03
CMD_ACK = 0x10
CMD_NAK = 0x11
CMD_ABORT = 0x1F
CMD_REQ_FILE = 0x20

MAX_PACKET_SIZE = 200  # bytes of data per packet (to fit AX.25 MTU)
HEADER_SIZE = 6  # proto_id(2) + cmd(1) + seq(2) + flags(1)
PACKET_TIMEOUT = 5.0
MAX_RETRIES = 10
ACK_TIMEOUT = 10.0


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc ^ 0xFFFF


class FileTransferPacket:
    """Single file transfer packet."""

    def __init__(self, cmd: int, seq: int, data: bytes = b''):
        self.cmd = cmd
        self.seq = seq
        self.data = data

    def pack(self) -> bytes:
        header = PROTO_ID + struct.pack('<BHB', self.cmd, self.seq, 0)
        payload = header + self.data
        crc = _crc16(payload)
        return payload + struct.pack('<H', crc)

    @classmethod
    def unpack(cls, raw: bytes) -> FileTransferPacket | None:
        if len(raw) < HEADER_SIZE + 2:
            return None
        if raw[:2] != PROTO_ID:
            return None
        cmd, seq, flags = struct.unpack_from('<BHB', raw, 2)
        payload = raw[HEADER_SIZE:-2]
        crc_received = struct.unpack_from('<H', raw, -2)[0]
        crc_computed = _crc16(raw[:-2])
        if crc_received != crc_computed:
            return None
        return cls(cmd, seq, payload)


class FileTransferSender:
    """Send a file over AX.25."""

    def __init__(self, source_call: str, dest_call: str,
                 tx_callback: Callable[[bytes], None],
                 log_fn: Callable[[str], None] | None = None,
                 progress_callback: Callable[[int, int] | None] = None,
                 on_complete: Callable[[bool, str] | None] = None):
        self.source_call = source_call
        self.dest_call = dest_call
        self._tx_callback = tx_callback
        self._log_fn = log_fn
        self._progress_callback = progress_callback
        self._on_complete = on_complete

        self._ack_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._abort = False
        self._total_packets = 0
        self._sent_packets = 0

    def send_file(self, filepath: str, path: list[str] | None = None):
        self._thread = threading.Thread(
            target=self._send_loop, args=(filepath, path or []), daemon=True
        )
        self._running = True
        self._abort = False
        self._thread.start()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def abort(self):
        self._abort = True
        self._running = False

    def wait(self, timeout: float = 600.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def progress(self) -> tuple[int, int]:
        return self._sent_packets, self._total_packets

    def _send_loop(self, filepath: str, path: list[str]):
        p = Path(filepath)
        if not p.exists():
            self._log(f"File not found: {filepath}")
            if self._on_complete:
                self._on_complete(False, "File not found")
            return

        file_size = p.stat().st_size
        filename = p.name.encode('utf-8')[:32]
        md5 = hashlib.md5(p.read_bytes()).hexdigest().encode()
        file_data = p.read_bytes()

        self._log(f"Starting file transfer: {filename.decode()} ({file_size} bytes)")

        start_pkt = FileTransferPacket(
            CMD_FILE_START, 0,
            struct.pack('<I', file_size) + struct.pack('<H', len(filename)) +
            filename + md5
        )
        self._send_packet(start_pkt, path)

        ack = self._wait_ack(0, path)
        if not ack:
            self._log("No ACK for FILE_START")
            if self._on_complete:
                self._on_complete(False, "No ACK for start")
            return

        num_packets = (file_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
        self._total_packets = num_packets
        self._sent_packets = 0

        for seq in range(num_packets):
            if self._abort:
                self._send_abort(path)
                self._log("Transfer aborted")
                if self._on_complete:
                    self._on_complete(False, "Aborted")
                return

            while self._paused and not self._abort:
                time.sleep(0.1)

            offset = seq * MAX_PACKET_SIZE
            chunk = file_data[offset:offset + MAX_PACKET_SIZE]

            pkt = FileTransferPacket(CMD_FILE_DATA, seq + 1, chunk)
            success = False
            for attempt in range(MAX_RETRIES):
                if self._abort:
                    break
                self._send_packet(pkt, path)
                ack = self._wait_ack(seq + 1, path)
                if ack:
                    success = True
                    break
                self._log(f"Retry {attempt + 1} for packet {seq + 1}")

            if not success:
                self._log(f"Failed to send packet {seq + 1}")
                if self._on_complete:
                    self._on_complete(False, f"Failed at packet {seq + 1}")
                return

            self._sent_packets = seq + 1
            if self._progress_callback:
                self._progress_callback(self._sent_packets, self._total_packets)

        end_pkt = FileTransferPacket(
            CMD_FILE_END, num_packets + 1,
            struct.pack('<I', num_packets) + md5
        )
        self._send_packet(end_pkt, path)
        ack = self._wait_ack(num_packets + 1, path)

        if ack:
            self._log(f"File transfer complete: {filename.decode()}")
            if self._on_complete:
                self._on_complete(True, "Complete")
        else:
            self._log("No ACK for FILE_END")
            if self._on_complete:
                self._on_complete(False, "No ACK for end")

    def _send_packet(self, pkt: FileTransferPacket, path: list[str]):
        body = pkt.pack()
        ax25_frame = encode_ax25_ui(
            self.source_call, self.dest_call, path, body
        )
        self._tx_callback(ax25_frame)

    def _wait_ack(self, seq: int, path: list[str]) -> bool:
        deadline = time.monotonic() + ACK_TIMEOUT
        while time.monotonic() < deadline:
            try:
                cmd, ack_seq = self._ack_queue.get_nowait()
                if cmd == CMD_ACK and ack_seq == seq:
                    return True
                elif cmd == CMD_NAK:
                    return False
            except queue.Empty:
                time.sleep(0.05)
        return False

    def _send_abort(self, path: list[str]):
        pkt = FileTransferPacket(CMD_ABORT, 0)
        self._send_packet(pkt, path)

    def receive_ack(self, cmd: int, seq: int):
        self._ack_queue.put((cmd, seq))

    def _log(self, msg: str):
        if self._log_fn:
            self._log_fn(msg)


class FileTransferReceiver:
    """Receive a file over AX.25."""

    def __init__(self, source_call: str, dest_call: str,
                 tx_callback: Callable[[bytes], None],
                 log_fn: Callable[[str], None] | None = None,
                 progress_callback: Callable[[int, int] | None] = None,
                 on_complete: Callable[[bool, str] | None] = None,
                 output_dir: str = '.'):
        self.source_call = source_call
        self.dest_call = dest_call
        self._tx_callback = tx_callback
        self._log_fn = log_fn
        self._progress_callback = progress_callback
        self._on_complete = on_complete
        self.output_dir = output_dir

        self._state = 'idle'
        self._filename = ''
        self._file_size = 0
        self._expected_packets = 0
        self._received_packets = 0
        self._file_data = bytearray()
        self._md5_expected = b''
        self._packet_buffer: dict[int, bytes] = {}
        self._running = False

    def start(self):
        self._state = 'wait_start'
        self._running = True
        self._file_data = bytearray()
        self._packet_buffer = {}

    def stop(self):
        self._running = False
        self._state = 'idle'

    def process_packet(self, frame_body: bytes):
        pkt = FileTransferPacket.unpack(frame_body)
        if pkt is None:
            return

        if pkt.cmd == CMD_FILE_START:
            self._handle_start(pkt)
        elif pkt.cmd == CMD_FILE_DATA:
            self._handle_data(pkt)
        elif pkt.cmd == CMD_FILE_END:
            self._handle_end(pkt)
        elif pkt.cmd == CMD_ABORT:
            self._log("Transfer aborted by sender")
            self._state = 'idle'
            if self._on_complete:
                self._on_complete(False, "Aborted by sender")

    def _handle_start(self, pkt: FileTransferPacket):
        if len(pkt.data) < 8:
            return
        self._file_size = struct.unpack_from('<I', pkt.data, 0)[0]
        fname_len = struct.unpack_from('<H', pkt.data, 4)[0]
        self._filename = pkt.data[6:6 + fname_len].decode('utf-8', errors='replace')
        self._md5_expected = pkt.data[6 + fname_len:6 + fname_len + 32]
        self._expected_packets = (self._file_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
        self._received_packets = 0
        self._file_data = bytearray()
        self._packet_buffer = {}
        self._state = 'receiving'
        self._log(f"Receiving: {self._filename} ({self._file_size} bytes, {self._expected_packets} packets)")

        self._send_ack(pkt.seq)

    def _handle_data(self, pkt: FileTransferPacket):
        if self._state != 'receiving':
            return
        if pkt.seq < 1 or pkt.seq > self._expected_packets:
            return

        self._packet_buffer[pkt.seq] = pkt.data
        self._received_packets = len(self._packet_buffer)
        self._send_ack(pkt.seq)

        if self._progress_callback:
            self._progress_callback(self._received_packets, self._expected_packets)

        self._log(f"Packet {pkt.seq}/{self._expected_packets} received")

    def _handle_end(self, pkt: FileTransferPacket):
        if self._state != 'receiving':
            return

        # Check for missing packets
        missing = []
        for seq in range(1, self._expected_packets + 1):
            if seq not in self._packet_buffer:
                missing.append(seq)

        if missing:
            self._log(f"Missing packets: {missing[:20]}{'...' if len(missing) > 20 else ''}")
            self._send_nak(pkt.seq)
            return

        for seq in range(1, self._expected_packets + 1):
            self._file_data.extend(self._packet_buffer[seq])

        md5 = hashlib.md5(self._file_data).hexdigest().encode()
        if md5 != self._md5_expected:
            self._log(f"CRC MISMATCH: expected {self._md5_expected.decode()}, got {md5.decode()}")
            self._log("File data discarded — not saved to disk")
            self._send_nak(pkt.seq)
            self._state = 'idle'
            if self._on_complete:
                self._on_complete(False, f"CRC mismatch for {self._filename}")
            return

        out_path = os.path.join(self.output_dir, self._filename)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(self._file_data)
        self._log(f"File saved: {out_path} ({len(self._file_data)} bytes)")

        self._send_ack(pkt.seq)
        self._state = 'idle'

        if self._on_complete:
            self._on_complete(True, out_path)

    def _send_ack(self, seq: int):
        pkt = FileTransferPacket(CMD_ACK, seq)
        body = pkt.pack()
        ax25_frame = encode_ax25_ui(
            self.source_call, self.dest_call, [], body
        )
        self._tx_callback(ax25_frame)

    def _send_nak(self, seq: int):
        pkt = FileTransferPacket(CMD_NAK, seq)
        body = pkt.pack()
        ax25_frame = encode_ax25_ui(
            self.source_call, self.dest_call, [], body
        )
        self._tx_callback(ax25_frame)

    def _log(self, msg: str):
        if self._log_fn:
            self._log_fn(msg)
