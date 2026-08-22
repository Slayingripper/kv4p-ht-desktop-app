# KV4P-Desktop

Desktop companion app for the [kv4p-ht](https://kv4p.com) open-source ham radio board (ESP32 + SA818).

KV4P-Desktop turns a kv4p-ht board into a full-featured desktop FM transceiver with APRS, spectrum analysis, SSTV, Morse/CW, digital modes, and file transfer — all over a single USB cable.

## Features

### Radio
- **FM Voice Transceiver** — Full-duplex audio via Opus codec over USB serial
- **PTT Control** — Software PTT button, physical PTT sync, and rigctld PTT from external apps
- **CTCSS Tones** — Transmit/receive CTCSS (PL tones) with full tone table
- **Squelch** — Adjustable squelch level (0–8)
- **Channel Memory** — Save/recall frequencies with name, offset, CTCSS, and notes. Import/export CSV
- **Frequency Scanner** — Scan 2m/70cm bands with dwell time and signal detection
- **Repeater Offset** — Configurable TX offset (+/– MHz)

### APRS
- **iGate** — Bidirectional gateway between RF and APRS-IS (internet), with packet filtering
- **Digipeater** — Automatic packet forwarding with WIDEn-N path handling
- **Messaging** — Send/receive APRS text messages with automatic ACK
- **Beacon** — Periodic position beaconing with configurable interval and path
- **Position Display** — Decode and display position reports on the map

### Spectrum Analysis
- **Audio FFT** — Real-time FFT of audio path (configurable FFT size, window, averaging)
- **RF Sweep** — Swept spectrum analyzer: hops through a frequency range, reads RSSI at each step, plots power vs frequency. Default: 2m band (144–148 MHz), 25 kHz steps, ~4 sec sweep
- **Waterfall Display** — Scrolling waterfall with color intensity mapping
- **Click-to-Tune** — Click the spectrum to jump to a frequency

### SSTV
- **Encode** — Convert images to SSTV audio (Martin M1/M2, Scottie S1/S2, PD-90/120, Robot 36, and more)
- **Decode** — Decode incoming SSTV transmissions with automatic mode detection and image preview

### Morse / CW
- **CW Keyer** — Iambic and straight keyer with adjustable WPM
- **Morse Decoder** — Real-time decode of incoming Morse code
- **Practice Mode** — Random callsign/text generation for Morse practice

### Digital Modes Integration
- **KISS TNC** — Dire Wolf / BPQ AX.25 interface over TCP (port 8001)
- **UDP Broadcast** — Receive spots from WSJT-X, FLDigi, JS8Call, Dire Wolf
- **Hamlib/RigCtlD** — Sync frequency and PTT with external radio apps via rigctld (TCP/4532)

### File Transfer
- **AX.25 File Transfer** — Send and receive files over AX.25 with packet sequencing, ACK/NAK, CRC-16 verification, and automatic retry

### Radio Faceplate
- **Analog S-Meter** — Needle-style S-meter with color-coded segments
- **LCD Frequency Display** — Large green-on-black frequency readout
- **Channel Knob** — Quick channel selection from the faceplate view
- Toggle between desktop and radio-face modes

### General
- **Persistent Settings** — Callsign, frequencies, and preferences saved across sessions
- **Audio Controls** — Mic gain and speaker volume sliders
- **Debug Console** — Full TX/RX frame hex dump and protocol logging
- **Self-Test** — Built-in AFSK demodulator test

## Requirements

- Python 3.10+
- A kv4p-ht board connected via USB
- Linux, macOS, or Windows with audio drivers
- System packages: `portaudio19-dev`, `libopus0` (Linux)

## Install

```bash
# System deps (Ubuntu/Debian)
sudo apt install python3-pip python3-pyqt6 portaudio19-dev libopus0

# Python dependencies
pip install -r requirements.txt
```

## Run

```bash
# Quick start
python run.py

# With options
python run.py --callsign KG4SHR --freq 146.520 --igate --beacon

# Full help
python run.py --help
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--callsign CALL` | Default callsign |
| `--freq MHZ` | Initial RX frequency |
| `--offset MHZ` | Repeater offset |
| `--rigctl-host HOST` | Hamlib rigctld host |
| `--rigctl-port PORT` | Hamlib rigctld port (default 4532) |
| `--igate` | Start iGate on launch |
| `--beacon` | Start beacon on launch |
| `--scanner` | Start frequency scanner on launch |
| `--log-level LEVEL` | Log level (debug/info/warn/error) |

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              PyQt6 GUI (MainWindow)                       │
│  Radio | Channels | Spectrum | APRS | SSTV | CW | FT     │
└──────┬─────────────────────────────────────────┬─────────┘
       │ QThread signals                         │ queues
       ▼                                         ▼
┌──────────────┐                        ┌──────────────┐
│ SerialWorker │◄──── Opus ───────────►│ AudioWorker  │
│ USB serial   │                        │ mic/speaker  │
│ FrameParser  │                        │ Opus + AFSK  │
└──────┬───────┘                        └──────────────┘
       │ USB (115200 baud)
       ▼
┌──────────────┐
│ kv4p-ht      │
│ ESP32+SA818  │
└──────────────┘

External Integrations:
  hamlib/rigctld ── TCP/4532 ──> RigCtlD
  Dire Wolf/BPQ  ── TCP/8001 ──> KissTnc
  WSJT-X/FLD/DW  ── UDP ──────> UdpBroadcastRx
```

## Protocol

Communication between host and ESP32 uses a binary frame protocol:

```
Delimiter (4B): DE AD BE EF
Command (1B):   Host or ESP command code
Length (2B):    Payload length (little-endian)
Payload (0-2048B)
```

Host commands: PTT_DOWN, PTT_UP, GROUP (freq/squelch/tone), FILTERS, STOP, CONFIG, TX_AUDIO, RSSI, TX_AX25
ESP commands: SMETER, DEBUG, HELLO, RX_AUDIO, VERSION, WINDOW_UPDATE, RX_AX25

## Testing

```bash
python -m pytest tests/ -v
```

539 tests — no hardware needed, all mocked.

## Building Standalone Binary

### Linux
```bash
./build.sh
# Output: dist/kv4p-desktop
```

### Windows
```batch
build_windows.bat
:: Output: dist\kv4p-desktop.exe
```

### macOS
```bash
./build_macos.sh
# Output: dist/KV4P-Desktop.app
```

### Notes
- PyInstaller bundles Python interpreter + all deps into a single file (~110 MB)
- USB serial (pyserial) works out of the box on all platforms
- Audio (sounddevice) requires platform-native audio: ALSA/PulseAudio (Linux), WASAPI (Windows), CoreAudio (macOS)
- Opus codec (opuslib) links against system `libopus`

## License

GPL v3 — same as the parent kv4p-ht project.

Built by 5B4ANU.
