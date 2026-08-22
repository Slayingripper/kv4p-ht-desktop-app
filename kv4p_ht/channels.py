"""
Channel memory system — save, load, import, export frequency channels.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

DEFAULT_FILE = Path.home() / ".config" / "kv4p" / "channels.json"


@dataclass
class Channel:
    name: str = "CH-1"
    freq_rx: float = 144.390
    offset: float = 0.0
    mode: str = "FM"
    ctcss_tx: int = 0
    ctcss_rx: int = 0
    squelch: int = 3
    bandwidth: int = 0
    high_power: bool = True
    notes: str = ""

    @property
    def freq_tx(self) -> float:
        return self.freq_rx + self.offset

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Channel:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


class ChannelBank:
    """Ordered list of channels persisted as JSON."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else DEFAULT_FILE
        self.channels: list[Channel] = []
        if self._path.exists():
            self.load()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.channels = [Channel.from_dict(c) for c in data]
        except Exception:
            self.channels = []

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([c.to_dict() for c in self.channels], indent=2),
            encoding="utf-8",
        )

    def add(self, ch: Channel, index: int | None = None) -> None:
        if index is None:
            self.channels.append(ch)
        else:
            self.channels.insert(index, ch)
        self.save()

    def remove(self, index: int) -> None:
        if 0 <= index < len(self.channels):
            self.channels.pop(index)
            self.save()

    def move(self, from_idx: int, to_idx: int) -> None:
        if 0 <= from_idx < len(self.channels) and 0 <= to_idx < len(self.channels):
            ch = self.channels.pop(from_idx)
            self.channels.insert(to_idx, ch)
            self.save()

    def update(self, index: int, ch: Channel) -> None:
        if 0 <= index < len(self.channels):
            self.channels[index] = ch
            self.save()

    # ── CSV import / export (CHIRP-compatible) ──────────────────

    def export_csv(self, path: str | Path) -> None:
        p = Path(path)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Name", "Frequency", "Offset", "Mode", "CTCSS TX", "CTCSS RX",
                "Squelch", "Bandwidth", "Power", "Notes",
            ])
            for ch in self.channels:
                writer.writerow([
                    ch.name, f"{ch.freq_rx:.4f}", f"{ch.offset:.4f}",
                    ch.mode, ch.ctcss_tx, ch.ctcss_rx, ch.squelch,
                    ch.bandwidth, "High" if ch.high_power else "Low", ch.notes,
                ])

    @staticmethod
    def _parse_offset(row: dict) -> float:
        """Parse a CSV offset value into a signed MHz offset.

        Detects the direction (+/-) from either a signed offset value or a
        separate duplex/direction column, and normalizes CHIRP-style kHz
        values into MHz.
        """
        raw = row.get("Offset", "0.0")
        try:
            magnitude = abs(float(raw))
        except (ValueError, TypeError):
            magnitude = 0.0

        # Direction: prefer an explicit duplex column, else the value's sign.
        sign = 1.0
        duplex = (row.get("Duplex") or row.get("Offset Direction")
                  or row.get("Offset Direction (MHz)") or "").strip().lower()
        if duplex:
            if duplex.startswith("-") or duplex in ("down", "minus"):
                sign = -1.0
            elif duplex.startswith("+") or duplex in ("up", "plus"):
                sign = 1.0
            elif duplex in ("simplex", "off", "none", ""):
                return 0.0
        else:
            try:
                if float(raw) < 0:
                    sign = -1.0
            except (ValueError, TypeError):
                pass

        # CHIRP exports offset in kHz; our internal unit is MHz.
        if magnitude >= 10.0:
            magnitude /= 1000.0

        return sign * magnitude

    def import_csv(self, path: str | Path) -> int:
        count = 0
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ch = Channel(
                        name=row.get("Name", f"CH-{len(self.channels)+1}"),
                        freq_rx=float(row.get("Frequency", "144.390")),
                        offset=self._parse_offset(row),
                        mode=row.get("Mode", "FM"),
                        ctcss_tx=int(float(row.get("CTCSS TX", "0"))),
                        ctcss_rx=int(float(row.get("CTCSS RX", "0"))),
                        squelch=int(row.get("Squelch", "3")),
                        bandwidth=int(row.get("Bandwidth", "0")),
                        high_power=row.get("Power", "High").upper().startswith("H"),
                        notes=row.get("Notes", ""),
                    )
                    self.channels.append(ch)
                    count += 1
                except Exception:
                    continue
        self.save()
        return count
