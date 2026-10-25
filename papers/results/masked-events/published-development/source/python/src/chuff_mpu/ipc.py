"""Serial IPC bridge between the MPU (this process) and the MCU side.

Protocol (see shared/comms_protocol.md and arduino/chuff_mcu/chuff_mcu.ino):
  MPU -> MCU, one JSON object per line: {"mood": "curious", "energy": 0.8}
  MCU -> MPU, one JSON object per line: {"event": "picked_up"}
Both sides are dumb about the other's internals — this module only knows
how to open the port and shuttle JSON lines across it.

This module is a real, working implementation (not a stub) since it's
small, has no audio/ML dependencies, and is exercised directly by
mvp_demo.py.
"""

from __future__ import annotations

import json
from typing import Any

import serial


class McuLink:
    """A line-oriented JSON serial connection to the MCU side."""

    def __init__(self, port: str, baud: int, timeout_s: float = 1.0) -> None:
        self._ser = serial.Serial(port, baud, timeout=timeout_s)

    def send_mood(self, mood: str, energy: float) -> None:
        """Send one `{"mood": ..., "energy": ...}` line to the MCU."""
        self._write_json({"mood": mood, "energy": round(float(energy), 3)})

    def poll_event(self) -> dict[str, Any] | None:
        """Read one pending `{"event": ...}` line from the MCU, if any.

        Returns:
            The parsed JSON object, or None if no complete line was waiting
            or it failed to parse (malformed lines are dropped, not raised).
        """
        if self._ser.in_waiting == 0:
            return None
        raw = self._ser.readline()
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8", errors="ignore").strip())
        except json.JSONDecodeError:
            return None

    def _write_json(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, separators=(",", ":")) + "\n"
        self._ser.write(line.encode("utf-8"))

    def close(self) -> None:
        self._ser.close()

    def __enter__(self) -> "McuLink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
