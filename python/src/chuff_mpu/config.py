"""Tunable constants for the Chuff UNO Q MPU-side pipeline.

Mirrors `esp32/main/pet_config.h` and the Chuff spec's rule that every
tunable threshold lives in one config module, never hardcoded elsewhere.
Values are starting points copied/adapted from the spec and will need
calibration against the real acoustic environment and hardware.
"""

# --- Serial IPC to the MCU (STM32) side, see ipc.py and arduino/chuff_mcu ---
SERIAL_PORT = "/dev/ttyACM0"  # UNO Q bridge port; override for bench testing
SERIAL_BAUD = 115200

# --- Audio capture (full MFCC pipeline, see audio.py) ---
SAMPLE_RATE = 16000  # Hz, USB mic via sounddevice
CLIP_DURATION_S = 1.0  # seconds per audio window
N_MFCC = 13

# --- KNN memory (mirrors esp32/config.py MAX_EMBEDDINGS, see memory.py) ---
MAX_EMBEDDINGS = 2048  # MPU side has far more RAM than the ESP32 PSRAM ceiling
NOVELTY_THRESHOLD_LOW = 0.4
NOVELTY_THRESHOLD_HIGH = 0.7
RECOGNITION_THRESHOLD = 0.2

# --- Dream loop (see dream.py) ---
IDLE_TIMEOUT_S = 300  # 5 minutes before dream mode
DREAM_SURPRISE_THRESHOLD = 0.6

# --- Two-unit emergent comms, phase 2 (see comms.py) ---
CALL_DURATION_MS = 500
CALL_FREQ_MIN_HZ = 400
CALL_FREQ_MAX_HZ = 3000
REPERTOIRE_SIZE = 12
ACK_FREQ_HZ = 800
ACK_DURATION_MS = 200
REINFORCEMENT_RATE = 0.1
DECAY_RATE = 0.05

# --- Timing ---
BORED_TIMEOUT_S = 1800  # 30 minutes no new sounds

# --- MVP energy->mood demo (see mvp_demo.py) ---
# Simple RMS-energy bands standing in for real novelty scoring until the
# MFCC + KNN pipeline (audio.py / memory.py / mood.py) is implemented.
MVP_POLL_INTERVAL_S = 0.5
MVP_CALM_ENERGY_MAX = 0.02
MVP_CURIOUS_ENERGY_MAX = 0.08
