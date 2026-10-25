"""Chuff — Arduino UNO Q MPU-side package.

Mirrors the esp32/ MicroPython module layout (audio, memory, dream, mood,
comms, ipc, config) but running as full Python 3 on the UNO Q's Debian
Linux side. See the top-level Chuff spec for the overall architecture.

The runnable MVP entry point is `chuff_mpu.mvp_demo:main` (installed as
the `chuff-mvp-demo` console script) — mic loudness -> mood -> serial
line to the MCU, standing in for the full audio+ML pipeline below until
it's implemented.
"""
