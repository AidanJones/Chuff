# arduino/ — Track B, MCU side

Firmware for the Arduino UNO Q's STM32U585 (MCU) side. The Qualcomm MPU
side (Debian Linux, Python 3) lives in `../python/`.

## Status

The UNO Q 4GB + Modulino kit are **on order, not yet in hand** (see the
hardware table in the top-level spec). `chuff_mcu/chuff_mcu.ino` is
written to not need them: it only uses `Serial` and `LED_BUILTIN`, so it
can be compiled and bench-tested on any Arduino board today, proving out
the serial IPC protocol ahead of the real hardware arriving.

## `chuff_mcu/chuff_mcu.ino`

Reads `{"mood": "...", "energy": 0.0-1.0}` JSON lines from
`python/src/chuff_mpu/mvp_demo.py` (or any sender following the same
protocol — see `../shared/comms_protocol.md`) and blinks `LED_BUILTIN` at
a rate set by `energy`, as a placeholder for the real per-mood LED
pattern.

Flash it with the Arduino IDE or `arduino-cli` the normal way; no extra
libraries required.

## When the UNO Q + Modulino kit arrive

See `TASKS.md` (CH-30 through CH-36) for the follow-up work: swapping the
`LED_BUILTIN` placeholder for the real 8x13 LED matrix and Modulino
Pixels, and wiring up Modulino Movement/Distance sensor events back to
the MPU side. The exact onboard-matrix API hasn't been verified against
real hardware yet — confirm it against Arduino's UNO Q documentation once
the board is in hand, rather than trusting any guess made before that.
