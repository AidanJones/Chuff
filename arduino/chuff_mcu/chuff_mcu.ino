/*
 * Chuff — Arduino UNO Q, MCU (STM32U585) side. v0 / MVP.
 *
 * STATUS: written but UNTESTED — the UNO Q + Modulino kit is still on
 * order (see Chuff spec, Track B hardware table). This sketch is the
 * smallest useful placeholder: it proves out the serial IPC protocol
 * between the MPU (Debian Python side, see python/) and the MCU, using
 * only onboard LED_BUILTIN so it can be bench-tested on *any* Arduino
 * before the real board arrives.
 *
 * Protocol (see shared/comms_protocol.md and the Chuff spec):
 *   MPU -> MCU, one JSON object per line:   {"mood": "curious", "energy": 0.8}
 *   MCU -> MPU, one JSON object per line:   {"event": "picked_up"}
 * Both sides are dumb about the other's internals — this sketch only
 * ever needs to pull the "mood" and "energy" fields back out.
 *
 * Deliberately no ArduinoJson dependency yet: the line format is fixed
 * and simple enough to hand-parse. Swap in a real JSON library once the
 * protocol grows (e.g. nested fields) rather than before.
 *
 * TODO once hardware arrives:
 *   - Replace the LED_BUILTIN blink with the UNO Q's built-in 8x13 LED
 *     matrix (face expression) and Modulino Pixels (Culture colour ring).
 *   - Replace blink-rate-as-mood-proxy with the real per-mood LED pattern
 *     from shared/mood_grammar.md.
 *   - Add Modulino Movement / Distance polling and emit {"event": ...}
 *     lines back to the MPU on pickup/proximity, per the spec.
 */

#include <Arduino.h>

// ---- Tunable constants (mirrors the "all thresholds in config" rule) ----
static const unsigned long SERIAL_BAUD = 115200;
static const unsigned long LINE_TIMEOUT_MS = 50;   // max time to wait mid-line
static const size_t LINE_BUF_SIZE = 128;

// Blink period range used as a stand-in for real mood LEDs: higher
// "energy" => faster blink. Replaced by LED matrix/Pixels once wired up.
static const unsigned long BLINK_PERIOD_MIN_MS = 100;
static const unsigned long BLINK_PERIOD_MAX_MS = 1000;

char line_buf[LINE_BUF_SIZE];
size_t line_len = 0;

unsigned long blink_period_ms = BLINK_PERIOD_MAX_MS;
unsigned long last_blink_toggle = 0;
bool led_state = false;

// Very small hand-rolled extractor for a top-level numeric/string field
// in a flat JSON object, e.g. find_field(line, "energy") on
// {"mood":"curious","energy":0.8} returns a pointer to "0.8".
// Not a general JSON parser — only safe for this fixed, flat protocol.
const char *find_field(const char *json, const char *key) {
  char needle[32];
  snprintf(needle, sizeof(needle), "\"%s\"", key);
  const char *pos = strstr(json, needle);
  if (!pos) return nullptr;
  pos = strchr(pos + strlen(needle), ':');
  if (!pos) return nullptr;
  pos++;
  while (*pos == ' ') pos++;
  return pos;
}

void handle_mood_line(const char *line) {
  const char *mood_val = find_field(line, "mood");
  const char *energy_val = find_field(line, "energy");

  if (energy_val) {
    float energy = atof(energy_val);
    energy = constrain(energy, 0.0f, 1.0f);
    // Map energy [0,1] -> blink period [MAX,MIN] (more energy = faster blink)
    blink_period_ms = BLINK_PERIOD_MAX_MS -
        (unsigned long)(energy * (BLINK_PERIOD_MAX_MS - BLINK_PERIOD_MIN_MS));
  }

  if (mood_val) {
    // v0: just log it. Real behaviour (LED matrix face + Pixels colour
    // per shared/mood_grammar.md) lands once the hardware is in hand.
    Serial.print("{\"event\":\"mood_received\",\"mood\":\"");
    for (const char *p = mood_val; *p && *p != '"' && *p != '}' && *p != ','; p++) {
      if (*p == '"') continue;
      Serial.print(*p);
    }
    Serial.println("\"}");
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(SERIAL_BAUD);
}

void loop() {
  // Non-blocking line reader off Serial.
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      line_buf[line_len] = '\0';
      if (line_len > 0) handle_mood_line(line_buf);
      line_len = 0;
    } else if (c != '\r' && line_len < LINE_BUF_SIZE - 1) {
      line_buf[line_len++] = c;
    }
  }

  // Blink LED at a rate set by the last "energy" value received.
  unsigned long now = millis();
  if (now - last_blink_toggle >= blink_period_ms) {
    led_state = !led_state;
    digitalWrite(LED_BUILTIN, led_state ? HIGH : LOW);
    last_blink_toggle = now;
  }
}
