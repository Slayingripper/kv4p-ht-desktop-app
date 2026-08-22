# KV4P-Desktop — Agent Guide

## Architecture
- PyQt6 GUI with QThread workers for serial and audio I/O
- Binary frame protocol over USB serial to ESP32+SA818 board
- Opus audio codec (48kHz mono, 40ms frames)
- AFSK1200 for APRS modulation/demodulation
- FFT-based spectrum analyzer with waterfall display

## Key Files
- `kv4p_ht/app.py` — MainWindow: UI, state, signal wiring, all tab panels
- `kv4p_ht/radio.py` — SerialWorker (USB frame I/O), AudioWorker (mic/speaker + Opus + AFSK demod)
- `kv4p_ht/protocol.py` — FrameParser (state machine), FrameSender, command enums
- `kv4p_ht/aprs.py` — AX.25 encode/decode, APRS info parser, IGate (APRS-IS thread), Digipeater
- `kv4p_ht/afsk.py` — AFSK1200 modulation (Bell 202) for APRS TX
- `kv4p_ht/afsk_demod.py` — AFSK1200 Goertzel-based demodulator for APRS RX
- `kv4p_ht/hamlib.py` — Hamlib rigctld TCP integration, mode switching (FM/USB/LSB/AM/CW)
- `kv4p_ht/kiss.py` — KISS TNC protocol for Dire Wolf / BPQ
- `kv4p_ht/udp_broadcast.py` — UDP listener for FLDigi/WSJT-X/Dire Wolf
- `kv4p_ht/scanner.py` — Frequency scanning engine
- `kv4p_ht/spectrum.py` — FFT spectrum analyzer, waterfall buffer
- `kv4p_ht/spectrum_widget.py` — PyQt6 spectrum display + waterfall visualization
- `kv4p_ht/morse.py` — CW keyer (iambic/straight), Morse decoder, practice generator
- `kv4p_ht/sstv.py` — SSTV encoder/decoder (Martin M1/M2, Scottie S1, PD-120, Robot)
- `kv4p_ht/ax25_file_transfer.py` — File transfer over AX.25 with sequencing, ACK, CRC
- `kv4p_ht/main.py` — Entry point, QApplication setup

## Tabs (app.py)
1. **Radio** — Frequency, PTT, S-meter, mode buttons, audio controls
2. **Spectrum** — Real-time FFT spectrum + waterfall waterfall
3. **APRS** — iGate, Digipeater, Beacon, messaging, position
4. **Digital Modes** — WSJT-X/FLDigi/JS8Call/Dire Wolf integration via UDP/KISS/Hamlib
5. **SSTV** — Encode images to SSTV audio, decode incoming SSTV
6. **Morse / CW** — CW keyer, text-to-morse, decoder, practice mode
7. **File Transfer** — Send/receive files over AX.25 protocol
8. **Scanner** — Frequency scanning with dwell and signal detection
9. **Settings** — Audio devices, application config

## Testing
```bash
python -m pytest tests/ -v
```
Tests use pytest with mocks. No physical hardware needed.

## Protocol
- Delimiter: `DE AD BE EF`
- Frame: Delimiter + 1B cmd + 2B LE length + payload (max 2048B)
- Host commands: PTT_DOWN(1), PTT_UP(2), GROUP(3), FILTERS(4), STOP(5), CONFIG(6), TX_AUDIO(7), HL(8), RSSI(9), TX_AX25(10)
- ESP commands: SMETER(0x53), DEBUG(1-5), HELLO(6), RX_AUDIO(7), VERSION(8), WINDOW_UPDATE(9), RX_AX25(10)

## Spectrum Analysis
- FFT size: 2048 (configurable)
- Window: Hann (configurable)
- Averaging: Exponential moving average (configurable)
- Waterfall: Circular buffer of 200 rows

## SSTV Modes
- Martin M1 (256 lines), M2 (128 lines)
- Scottie S1 (256 lines), S2 (128 lines)
- PD-50, PD-90, PD-120
- Robot 8/12/24/32

## File Transfer Protocol
- Header: KF + CMD + SEQ + FLAGS
- Commands: FILE_START(0x01), FILE_DATA(0x02), FILE_END(0x03), ACK(0x10), NAK(0x11), ABORT(0x1F)
- Max 200 bytes data per packet
- CRC-16 per packet
- Automatic retry with configurable timeout

## Dev Notes
- AudioQueue bypasses Qt signal path (SerialWorker -> SimpleQueue -> AudioWorker) to prevent audio dropouts
- RSSI -> S-meter: 9.73 * ln(0.0297 * rssi) - 1.88, clamped to 1-9
- AFSK demod: sliding-window Goertzel + zero-crossing timing recovery + CRC-CCITT validation
- AGENTS.md must be updated when new modules are added
