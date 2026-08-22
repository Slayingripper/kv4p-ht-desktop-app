"""
CW Morse code keyer, decoder, and practice mode.

Features:
  - Iambic keyer with adjustable WPM
  - Straight key mode
  - Practice/random QSO generator
  - Timing-based character decoder
"""
from __future__ import annotations

import queue
import random
import threading
import time
from collections.abc import Callable

# ── Morse code tables ─────────────────────────────────────────────

CHAR_TO_MORSE: dict[str, str] = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.',
    '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-',
    '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-',
    '+': '.-.-.', '-': '-....-', '_': '..--.-', '"': '.-..-.',
    '$': '...-..-', '@': '.--.-.', ' ': ' ',
}

MORSE_TO_CHAR: dict[str, str] = {v: k for k, v in CHAR_TO_MORSE.items()}

# ── Common abbreviations ──────────────────────────────────────────

ABBREVIATIONS: dict[str, str] = {
    'CQ': 'CQ CQ CQ DE',
    'DE': 'FROM',
    'K': 'OVER',
    'R': 'ROGER',
    'TU': 'THANK YOU',
    '73': 'BEST REGARDS',
    'OM': 'OLD MAN',
    'YL': 'YOUNG LADY',
    'HR': 'HERE',
    'FB': 'FINE BUSINESS',
    'QTH': 'LOCATION',
    'QSL': 'CONFIRMED',
    'QRM': 'INTERFERENCE',
    'QRN': 'ATMOSPHERIC NOISE',
    'QRS': 'SLOWER',
    'QSY': 'CHANGE FREQUENCY',
    'PSE': 'PLEASE',
    'agn': 'AGAIN',
}

# ── Practice phrases ──────────────────────────────────────────────

PRACTICE_PHRASES = [
    "CQ CQ CQ DE N0CALL N0CALL K",
    "N0CALL DE W1AW TU 73",
    "UR RST IS 599 5NN",
    "NAME IS JOHN JOHN",
    'QTH IS DENVER DENVER CO',
    "CQ POTA CQ POTA DE N0CALL",
    "K9FAA DE N0CALL AR",
    "FB OM TU 73 SK",
    "WX IS GOOD HERE QRZ",
    "PSE AGN UR RST 449",
]


class MorseKeyer:
    """CW keyer with timing-based output."""

    def __init__(self, wpm: int = 20, mode: str = "iambic",
                 tone_hz: int = 700, sample_rate: int = 48000):
        self.wpm = wpm
        self.mode = mode  # "iambic", "straight", "Ultimatic"
        self.tone_hz = tone_hz
        self.sample_rate = sample_rate

        self._element_time = 1.2 / wpm
        self._dot_time = self._element_time
        self._dash_time = self._element_time * 3
        self._char_gap = self._element_time * 3
        self._word_gap = self._element_time * 7

        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._tone_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._key_state = False

    def set_wpm(self, wpm: int):
        self.wpm = max(5, min(60, wpm))
        self._element_time = 1.2 / self.wpm
        self._dot_time = self._element_time
        self._dash_time = self._element_time * 3
        self._char_gap = self._element_time * 3
        self._word_gap = self._element_time * 7

    @property
    def dot_time_ms(self) -> float:
        return self._dot_time * 1000

    @property
    def dash_time_ms(self) -> float:
        return self._dash_time * 1000

    def text_to_timing(self, text: str) -> list[tuple[bool, float]]:
        """Convert text to list of (key_down, duration_sec) tuples."""
        timing: list[tuple[bool, float]] = []
        for ch in text.upper():
            if ch == ' ':
                timing.append((False, self._word_gap))
                continue
            morse = CHAR_TO_MORSE.get(ch, '')
            if not morse:
                continue
            for i, elem in enumerate(morse):
                dur = self._dot_time if elem == '.' else self._dash_time
                timing.append((True, dur))
                if i < len(morse) - 1:
                    timing.append((False, self._element_time))
            timing.append((False, self._char_gap))
        return timing

    def generate_tone(self, text: str) -> list[int]:
        """Generate PCM int16 samples for a text string."""
        import numpy as np
        timing = self.text_to_timing(text)
        samples = []
        sr = self.sample_rate
        phase = 0.0
        increment = 2.0 * np.pi * self.tone_hz / sr

        for key_down, duration_sec in timing:
            n_samples = int(sr * duration_sec)
            if key_down:
                for _ in range(n_samples):
                    val = int(16000 * np.sin(phase))
                    samples.append(val)
                    phase += increment
                    if phase > 2 * np.pi:
                        phase -= 2 * np.pi
            else:
                samples.extend([0] * n_samples)
        return samples

    def generate_tone_array(self, text: str):
        """Generate float32 numpy waveform for a text string."""
        import numpy as np
        timing = self.text_to_timing(text)
        chunks: list[np.ndarray] = []
        sr = self.sample_rate
        phase = 0.0
        increment = 2.0 * np.pi * self.tone_hz / sr

        for key_down, duration_sec in timing:
            n_samples = int(sr * duration_sec)
            if key_down:
                t = np.arange(n_samples, dtype=np.float32)
                wave = 0.5 * np.sin(phase + increment * t)
                chunks.append(wave)
                phase += increment * n_samples
                phase %= 2 * np.pi
            else:
                chunks.append(np.zeros(n_samples, dtype=np.float32))

        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


class MorseDecoder:
    """Timing-based Morse decoder for real-time key input."""

    def __init__(self, wpm: int = 20):
        self.set_wpm(wpm)
        self._buffer = ""
        self._decoded_text = ""
        self._last_key_up = 0.0
        self._last_key_down = 0.0
        self._key_down_time = 0.0
        self._key_up_time = 0.0
        self._is_down = False

    def set_wpm(self, wpm: int):
        self.wpm = max(5, min(60, wpm))
        self._element_time = 1.2 / self.wpm
        self._dot_thresh = self._element_time * 1.5
        self._dash_thresh = self._element_time * 2.5
        self._char_thresh = self._element_time * 2.0
        self._word_thresh = self._element_time * 5.0

    def process_key(self, down: bool, timestamp: float | None = None):
        """Feed key state changes.  Call with timestamp for timing-based decode."""
        now = timestamp or time.monotonic()
        if down and not self._is_down:
            self._is_down = True
            self._key_down_time = now
            if self._last_key_up > 0:
                gap = now - self._last_key_up
                if gap > self._word_thresh and self._buffer:
                    self._finish_char()
                    self._decoded_text += ' '
                elif gap > self._char_thresh and self._buffer:
                    self._finish_char()
        elif not down and self._is_down:
            self._is_down = False
            self._last_key_up = now
            element_time = now - self._key_down_time
            if element_time < self._dot_thresh:
                self._buffer += '.'
            elif element_time < self._dash_thresh:
                self._buffer += '-'
            else:
                self._buffer += '-'

        self._last_key_down = now

    def _finish_char(self):
        if self._buffer:
            ch = MORSE_TO_CHAR.get(self._buffer, '?')
            self._decoded_text += ch
            self._buffer = ""

    def get_text(self) -> str:
        return self._decoded_text

    def get_current_element(self) -> str:
        return self._buffer

    def reset(self):
        self._buffer = ""
        self._decoded_text = ""


class PracticeGenerator:
    """Generate random practice QSO messages."""

    def __init__(self):
        self._calls = [
            'W1AW', 'N0CALL', 'K9FAA', 'VE3ABC', 'G3XYZ',
            'JA1ABC', 'DL5ABC', 'VK2ABC', 'ZL1ABC', 'LU1ABC',
        ]

    def random_callsign(self) -> str:
        prefix = random.choice(['W', 'K', 'N', 'VE', 'G', 'JA', 'DL', 'VK'])
        num = random.randint(1, 9)
        suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
        return f"{prefix}{num}{suffix}"

    def generate_exchange(self) -> str:
        r = random.random()
        if r < 0.3:
            call = self.random_callsign()
            return f"CQ CQ CQ DE {call} {call} K"
        elif r < 0.5:
            call1 = self.random_callsign()
            call2 = self.random_callsign()
            rst = f"5{random.randint(3,9)}{random.randint(1,9)}"
            return f"{call1} DE {call2} TU RST {rst}"
        elif r < 0.7:
            call = self.random_callsign()
            name = random.choice(['BOB', 'AL', 'FRED', 'JOE', 'TOM', 'SAM'])
            qth = random.choice(['DENVER', 'NEW YORK', 'CHICAGO', 'SEATTLE', 'BOSTON'])
            return f"{call} DE N0CALL NAME {name} QTH {qth} TU"
        else:
            return random.choice(PRACTICE_PHRASES)

    def generate_session(self, count: int = 10) -> list[str]:
        return [self.generate_exchange() for _ in range(count)]
