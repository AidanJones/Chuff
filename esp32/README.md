# ESP32 Robotic Pet

> **This is Chuff's Track A (ESP32-S3).** Copied in directly from the
> standalone `esp32_digital_pet` project as the starting point, since it
> already had display/touch/audio working on this exact board — see the
> top-level repo `README.md` for why that's a deliberate deviation from
> the original Chuff spec (which specified MicroPython), and `TASKS.md`
> for what's Chuff-specific so far (`pet_config.h`'s idle/"dreaming"
> behaviour) vs. still to build (audio capture, novelty/mood, Culture
> colour expression).

A Tamagotchi-style animated pet running on a Waveshare ESP32-S3-Touch-AMOLED-1.8.
The character is **Flowey the flower** from Undertale — an animated face with blinking eyes,
a roaming gaze, a wide rectangular smile, and touch-triggered reactions (wiggle + blink + mouth
chomp + chiptune jingle).

---

## Project aims

- **Phase 1 (done):** Smoke-test board; confirm build + flash pipeline works.
- **Phase 2 (done):** Drive the AMOLED display with LVGL; render an animated Flowey face with idle animations (blink, look-around) and touch reactions.
- **Phase 3 (planned):** IMU reactions — tilt the board to make Flowey look in that direction; shake to trigger a startled expression.
- **Phase 4 (planned):** More expressions and a mood system (happy / sad / hungry / sleepy) driven by time-of-day or interaction frequency.
- **Phase 5 (planned):** BLE or Wi-Fi integration — send notifications to the pet; pet reacts to events from the outside world.
- **Phase 6 (stretch):** Physical servo output for a body that moves in sync with on-screen animations.

---

## Hardware

| Component | Details |
|---|---|
| Board | Waveshare ESP32-S3-Touch-AMOLED-1.8 |
| MCU | ESP32-S3-WROOM-1, dual-core LX7 @ 240 MHz |
| Display | 1.8" AMOLED, **368 × 448 px**, SH8601 driver via QSPI (4-wire SPI) |
| Touch | FT3168 capacitive (FT5x06-compatible), I2C |
| IO expander | TCAL9534 at I2C addr `0x20` — controls LCD reset, touch reset, DSI power, SD CS |
| IMU | QMI8658 6-axis (accelerometer + gyroscope), I2C |
| Power / PMIC | AXP2101, supports 3.7 V LiPo via MX1.25 2-pin connector |
| RTC | PCF85063 |
| Storage | 16 MB Flash (QIO), 8 MB PSRAM (octal) |
| Wireless | Wi-Fi 802.11 b/g/n, Bluetooth 5 LE |

### Key hardware references

| Resource | URL |
|---|---|
| Board overview & pinout | https://www.waveshare.com/wiki/ESP32-S3-Touch-AMOLED-1.8 |
| Board schematic (PDF) | https://www.waveshare.com/w/upload/2/2c/ESP32-S3-Touch-AMOLED-1.8-Schematic.pdf |
| Waveshare demo code (Arduino + IDF) | https://www.waveshare.com/wiki/ESP32-S3-Touch-AMOLED-1.8#Resources |
| SH8601 QSPI AMOLED datasheet | (bundled in Waveshare package; driver adapted from their GitHub) |
| FT5x06 touch driver (Espressif) | https://components.espressif.com/components/espressif/esp_lcd_touch_ft5x06 |
| TCAL9534 IO expander datasheet | https://www.ti.com/product/TCAL9534 |
| QMI8658 IMU datasheet | https://www.qstcorp.com/upload/pdf/202202/QMI8658A%20datasheet%20rev%200.9.pdf |
| AXP2101 PMIC datasheet | https://www.x-powers.com/en.php/Info/product_detail/article_id/112 |
| ESP32-S3 technical reference | https://www.espressif.com/sites/default/files/documentation/esp32-s3_technical_reference_manual_en.pdf |

---

## Pin assignments

### LCD (QSPI / SPI2_HOST)

| Signal | GPIO |
|---|---|
| CLK | 11 |
| D0 (MOSI) | 4 |
| D1 | 5 |
| D2 | 6 |
| D3 | 7 |
| CS | 12 |

### I2C (shared bus, I2C_NUM_0)

| Device | I2C addr | SCL | SDA | Notes |
|---|---|---|---|---|
| TCAL9534 IO expander | `0x20` | 14 | 15 | Must be init'd first — holds LCD & touch in reset |
| FT3168 touch | `0x38` | 14 | 15 | INT on GPIO 21 (active-low) |
| ES8311 audio codec | `0x18` | 14 | 15 | CE pin low = 0x18; shared bus, handle passed to pet_sound_init() |
| QMI8658 IMU | `0x6B` | 14 | 15 | Not yet used in firmware |
| PCF85063 RTC | `0x51` | 14 | 15 | Not yet used in firmware |
| AXP2101 PMIC | `0x34` | 14 | 15 | Not yet used in firmware |

### IO expander outputs (TCAL9534, via I2C)

| Bit | Function |
|---|---|
| 0 | LCD reset (active-low) |
| 1 | DSI / display power enable |
| 2 | Touch reset (active-low) |
| 7 | SD card CS |

The expander **must** be configured and its reset lines pulsed before the SPI display is
initialised. See `io_exp_reset()` in `main.c`.

### Audio (ES8311 codec + built-in speaker)

The board has a built-in speaker driven by an ES8311 mono audio codec over I2S.

| Signal | GPIO | Notes |
|---|---|---|
| I2S MCLK | **16** | Master clock to ES8311 |
| I2S BCLK | **9** | Bit clock |
| I2S WS / LRCK | **45** | Word select |
| I2S DOUT | **8** | Digital audio data → ES8311 DAC |
| I2S DIN | **10** | Digital audio data ← ES8311 ADC (not used) |
| PA enable | **46** | HIGH = speaker amplifier on |

ES8311 is at I2C address `0x18` on the shared `I2C_NUM_0` bus (SCL=14, SDA=15).
MCLK is configured as `sample_rate × 384` (6,144,000 Hz at 16 kHz).

---

## Software stack

| Component | Version | Notes |
|---|---|---|
| ESP-IDF | 6.0.1 | Target: `esp32s3` |
| LVGL | ~8.4.x | **Not v9** — API differs significantly; see gotchas below |
| esp_lcd_sh8601 | local copy | `components/esp_lcd_sh8601/` (adapted from Waveshare GitHub) |
| esp_lcd_touch_ft5x06 | `*` (latest) | Via idf_component_manager |
| es8311 | local copy | `components/es8311/` (from Waveshare IDF example) |
| FreeRTOS | bundled with IDF | LVGL runs in its own task, mutex-protected |

### LVGL 8 vs 9 — critical API differences for future agents

If LVGL is ever upgraded to v9, these APIs will break:

| LVGL 8 (current) | LVGL 9 equivalent |
|---|---|
| `lv_disp_drv_t` / `lv_disp_draw_buf_t` | `lv_display_t` |
| `lv_disp_drv_register()` | `lv_display_create()` |
| `lv_indev_drv_t` / `lv_indev_drv_register()` | `lv_indev_create()` |
| `LV_PART_MAIN` (arc background) | same |
| No `lv_obj_set_style_transform_scale_y` | `lv_obj_set_style_transform_scale_y` added in v9 |
| `lv_arc_set_bg_start_angle` / `lv_arc_set_bg_end_angle` | same |

---

## Architecture / code map

```
(repo root)
├── README.md
├── .gitignore
├── CMakeLists.txt          ← ESP-IDF top-level project file
├── sdkconfig               ← Full menuconfig output (committed for reproducibility)
├── sdkconfig.defaults      ← Project-specific defaults (flash size, PSRAM, LVGL colour)
├── dependencies.lock       ← Component manager lock file (like package-lock.json)
├── main/
│   ├── main.c              ← Hardware init, LVGL setup, touch callback, app_main
│   ├── pet_face.c          ← All LVGL drawing + animation logic for Flowey
│   ├── pet_face.h          ← pet_face_init(), pet_face_on_touch()
│   ├── pet_config.h        ← Chuff mood/animation tunables (idle timeout, cadences)
│   ├── pet_sound.c         ← ES8311 + I2S chiptune jingle player (built-in speaker)
│   ├── pet_sound.h         ← pet_sound_init(i2c_bus), pet_sound_play()
│   ├── idf_component.yml   ← Component manager deps (LVGL, touch driver)
│   └── CMakeLists.txt      ← REQUIRES es8311 esp_driver_i2s
├── components/
│   ├── esp_lcd_sh8601/     ← Local copy of SH8601 QSPI panel driver
│   └── es8311/             ← Local copy of ES8311 codec driver (from Waveshare IDF example)
└── tools/
    ├── animation-editor.html  ← Pixel art frame editor (open directly in browser, no server needed)
    └── png_to_c.py            ← Converts sprite sheet PNG → C header with RGB565 arrays
```

### LVGL task model

LVGL is not thread-safe. The project uses the standard Espressif pattern:

1. A dedicated `lvgl_task` FreeRTOS task calls `lv_timer_handler()` in a loop and holds
   `lvgl_mux` (a `SemaphoreHandle_t` mutex) while doing so.
2. Any code that touches LVGL objects from outside that task must take the mutex first.
3. The touch callback (`touch_cb`) is called *from within* `lv_timer_handler`, so it already
   holds the mutex — do not re-acquire it there.
4. `pet_sound_play()` spawns its own FreeRTOS task for blocking `vTaskDelay` calls; it does
   not touch LVGL and therefore needs no mutex.

### Display init sequence (critical)

The SH8601 AMOLED will not respond unless the IO expander (TCAL9534) has first been
initialised and its reset outputs pulsed. The mandatory order in `app_main` is:

```
1. i2c_new_master_bus()
2. io_exp_reset()          ← pulse LCD_RST and TCH_RST low, then high
3. spi_bus_initialize()
4. esp_lcd_new_panel_io_spi()
5. esp_lcd_new_panel_sh8601()
6. esp_lcd_panel_init() + disp_on_off()
7. touch init (over shared I2C bus)
8. LVGL init + register disp + indev drivers
```

Skipping step 2 leaves the display in reset and produces a blank screen with no errors.

### Known LVGL 8 rendering gotcha — transform_angle on lv_obj children

`lv_obj_set_style_transform_angle()` silently produces nothing when applied to children of
a regular `lv_obj` container. It **only** works on direct children of the screen object
(`lv_scr_act()`). The workaround used here: **eliminate all `transform_angle` calls**.

- Petals are now circles (`radius = PETAL_DIAM/2`) — no rotation required, and this matches
  the pixel-art sprite more closely anyway.
- The stem is a plain vertical rectangle — simpler and guaranteed to render.
- Eyebrows are horizontal bars.

If a future design requires rotated shapes, use `lv_canvas` with software rotation, or add
the object directly to `scr` (not a child container).

### Display colour depth

The SH8601 requires 16-bit colour with **byte-swap** enabled:

```
# sdkconfig.defaults
CONFIG_LV_COLOR_DEPTH_16=y
CONFIG_LV_COLOR_16_SWAP=y
```

Without `COLOR_16_SWAP`, colours appear as red/blue swapped (everything looks wrong).

### Audio init sequence

```
1. i2c_new_master_bus()         ← creates i2c_bus (shared by all I2C devices)
2. pet_sound_init(i2c_bus)      ← I2S channel, ES8311 init; PA pin set HIGH
   (GPIO 46 HIGH before i2s_channel_enable so speaker is active when audio starts)
3. LVGL init / face init
```

`pet_sound_init()` takes the existing `i2c_master_bus_handle_t` — it **must not** create its
own bus on the same port and pins (will fail with `ESP_ERR_INVALID_STATE`). All five I2C
devices on this board share the single `I2C_NUM_0` bus.

### ES8311 component — why it is copied locally

The `espressif/es8311` component on the IDF Component Registry carries the constraint
`idf: '>=4.4,<6.0'`. This project targets **IDF 6.0.1**, so the registry version would be
rejected by the component manager. The driver itself works fine on IDF 6 — the constraint is
just stale. The copy in `components/es8311/` is taken verbatim from the Waveshare IDF v5.5.1
example and skips the version check.

Do **not** add `espressif/es8311` to `idf_component.yml` — use the local component only.

### ES8311 I2C address — 7-bit vs 8-bit confusion

The `esp_codec_dev` abstraction layer defines `ES8311_CODEC_DEFAULT_ADDR (0x30)`, which is
the 8-bit write address (0x18 << 1). The lower-level `es8311.h` used here defines
`ES8311_ADDRESS_0 (0x18)`, which is the correct 7-bit address for the new `i2c_master` driver.
Always use `ES8311_ADDRESS_0` (or the numeric `0x18`) with this codebase.

### I2S driver — explicit REQUIRES needed

`driver/i2s_std.h` lives in the `esp_driver_i2s` component. Unlike most core drivers it is
**not** transitively pulled in, so `main/CMakeLists.txt` must list it explicitly:

```cmake
REQUIRES es8311 esp_driver_i2s
```

Omitting it produces `fatal error: driver/i2s_std.h: No such file or directory` even though
the header exists in the IDF tree.

### IntelliSense stale-cache warning

VS Code IntelliSense is driven by `build/compile_commands.json`. Until a successful build
runs, newly added components and their include paths are invisible to the IDE, producing
false "no such file" and "too many arguments" errors. Run **Build** once to regenerate
`compile_commands.json` and the errors will clear.

---

## Feature status

| Feature | Status |
|---|---|
| Display on, LVGL running | Done |
| Flowey face — idle blink | Done (every ~3.5 s) |
| Flowey face — look-around | Done (pupil drifts to 9 positions, ~4.2 s cycle) |
| Touch → wiggle + blink + mouth chomp | Done |
| Touch → chiptune jingle (built-in speaker) | Done |
| Mood: "dreaming" after 5 min idle (slow cadence, no blink) | Done — unverified, not build-tested in this environment |
| IMU (QMI8658) reactions | Not started |
| Mood / expression system (full novelty → Culture colour) | Not started — see TASKS.md CH-11 through CH-13 |
| Wi-Fi / BLE | Not started |
| PMIC / battery monitoring | Not started |

---

## Build & flash

Use the **ESP-IDF VS Code extension** buttons in the status bar (Build ▶ → Flash ⚡ → Monitor 🖥).

CLI equivalent:
```bash
cd hello_world
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/cu.usbmodem* flash monitor
```

The `sdkconfig.defaults` file pre-configures 16 MB flash QIO, octal PSRAM, 240 MHz CPU,
and LVGL colour settings so a full `menuconfig` pass is rarely needed.
