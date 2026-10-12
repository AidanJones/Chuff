#include "esp_heap_caps.h"
#include "lvgl.h"
#include "pet_face.h"
#include "pet_sound.h"
#include "pet_config.h"
#include "pet_neutral.h"
#include "pet_idle.h"
#include "pet_blink.h"
#include "pet_surprise.h"
#include "pet_anger.h"
#include "pet_happy.h"

#define LCD_W    368
#define LCD_H    448
#define SCALE    12
#define SCALED_W (PET_ANIM_COLS * SCALE)   /* 360 */
#define SCALED_H (PET_ANIM_ROWS * SCALE)   /* 444 */

#define FRAME_MS       125    /* 8 fps */

/* ── Animation table ──────────────────────────────────────────────────────── */

typedef struct { const uint16_t *const *frames; int n; } anim_t;

#define ANIM_IDLE     0
#define ANIM_BLINK    1
#define ANIM_SURPRISE 2
#define ANIM_ANGER    3
#define ANIM_HAPPY    4
#define ANIM_COUNT    5

static const anim_t ANIMS[ANIM_COUNT] = {
    [ANIM_IDLE]     = { pet_idle_frames,     PET_IDLE_N_FRAMES     },
    [ANIM_BLINK]    = { pet_blink_frames,    PET_BLINK_N_FRAMES    },
    [ANIM_SURPRISE] = { pet_surprise_frames, PET_SURPRISE_N_FRAMES },
    [ANIM_ANGER]    = { pet_anger_frames,    PET_ANGER_N_FRAMES    },
    [ANIM_HAPPY]    = { pet_happy_frames,    PET_HAPPY_N_FRAMES    },
};

/* ── State ────────────────────────────────────────────────────────────────── */

static lv_obj_t    *g_img;
static uint16_t    *g_scale_buf;   /* SCALED_W × SCALED_H × 2 bytes in PSRAM */
static lv_img_dsc_t g_dsc = {
    .header    = { .cf = LV_IMG_CF_TRUE_COLOR,
                   .w  = SCALED_W,
                   .h  = SCALED_H },
    .data_size = SCALED_W * SCALED_H * sizeof(uint16_t),
    .data      = NULL,
};

static int  g_cur_anim = -1;   /* -1 = resting, showing neutral */
static int  g_fi       = 0;
static bool g_busy     = false;

/* ── Mood: dreaming after idle ───────────────────────────────────────────────
 * Minimal first slice of the Culture colour grammar: no mic/novelty input
 * yet (that's the Phase 1 audio pipeline, still to come), but Chuff already
 * knows the difference between "awake" and "dreaming" based on how long it's
 * been since the last touch. Dreaming slows the idle animation cadence to a
 * calm "breathing" pace and skips blinking; a touch always wakes it back up.
 */
static lv_timer_t *g_idle_timer;
static uint32_t    g_last_touch_tick;
static bool        g_dreaming = false;

/* ── Helpers ──────────────────────────────────────────────────────────────── */

/* Nearest-neighbour scale from 30×37 source into the PSRAM scale buffer. */
static void scale_frame(const uint16_t *src)
{
    for (int r = 0; r < PET_ANIM_ROWS; r++) {
        for (int c = 0; c < PET_ANIM_COLS; c++) {
            uint16_t pix = src[r * PET_ANIM_COLS + c];
            for (int dr = 0; dr < SCALE; dr++) {
                uint16_t *row = g_scale_buf + (r * SCALE + dr) * SCALED_W + c * SCALE;
                for (int dc = 0; dc < SCALE; dc++) row[dc] = pix;
            }
        }
    }
    lv_obj_invalidate(g_img);
}

static void play(int anim_id)
{
    g_cur_anim = anim_id;
    g_fi       = 0;
}

/* ── Timers ───────────────────────────────────────────────────────────────── */

static void frame_tick(lv_timer_t *t)
{
    (void)t;
    if (g_cur_anim < 0) return;  /* neutral already in buffer */

    const anim_t *a = &ANIMS[g_cur_anim];
    if (g_fi < a->n) {
        scale_frame(a->frames[g_fi++]);
    } else {
        scale_frame(pet_neutral_frames[0]);
        g_cur_anim = -1;
        g_fi       = 0;
        g_busy     = false;
    }
}

static void idle_tick(lv_timer_t *t)
{
    (void)t;

    uint32_t idle_for = lv_tick_get() - g_last_touch_tick;
    bool should_dream = idle_for >= CHUFF_IDLE_TIMEOUT_MS;
    if (should_dream != g_dreaming) {
        g_dreaming = should_dream;
        lv_timer_set_period(g_idle_timer,
            g_dreaming ? CHUFF_DREAM_IDLE_PERIOD_MS : CHUFF_NORMAL_IDLE_PERIOD_MS);
    }

    if (g_busy) return;

    if (g_dreaming) {
        play(ANIM_IDLE);   /* slow look-around only; no blinking while dreaming */
        return;
    }

    static int seq = 0;
    play(seq++ % 2 == 0 ? ANIM_BLINK : ANIM_IDLE);
}

/* ── Public API ───────────────────────────────────────────────────────────── */

void pet_face_on_touch(void)
{
    g_last_touch_tick = lv_tick_get();
    if (g_dreaming) {
        g_dreaming = false;
        lv_timer_set_period(g_idle_timer, CHUFF_NORMAL_IDLE_PERIOD_MS);
    }

    if (g_busy) return;
    g_busy = true;
    pet_sound_play();
    play(lv_tick_get() % 2 ? ANIM_HAPPY : ANIM_SURPRISE);
}

void pet_face_init(void)
{
    /* ~312 KB — allocate from PSRAM, fall back to internal RAM */
    g_scale_buf = heap_caps_malloc(SCALED_W * SCALED_H * sizeof(uint16_t),
                                   MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!g_scale_buf)
        g_scale_buf = heap_caps_malloc(SCALED_W * SCALED_H * sizeof(uint16_t),
                                       MALLOC_CAP_8BIT);
    assert(g_scale_buf);
    g_dsc.data = (const uint8_t *)g_scale_buf;

    /* Pre-fill buffer with neutral frame before creating the LVGL object */
    for (int r = 0; r < PET_ANIM_ROWS; r++) {
        for (int c = 0; c < PET_ANIM_COLS; c++) {
            uint16_t pix = pet_neutral_frames[0][r * PET_ANIM_COLS + c];
            for (int dr = 0; dr < SCALE; dr++) {
                uint16_t *row = g_scale_buf + (r * SCALE + dr) * SCALED_W + c * SCALE;
                for (int dc = 0; dc < SCALE; dc++) row[dc] = pix;
            }
        }
    }

    lv_obj_t *scr = lv_scr_act();
    lv_obj_set_style_bg_color(scr, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(scr, LV_OPA_COVER, 0);

    g_img = lv_img_create(scr);
    lv_img_set_src(g_img, &g_dsc);
    /* Center the 360×444 sprite on the 368×448 display */
    lv_obj_set_pos(g_img, (LCD_W - SCALED_W) / 2, (LCD_H - SCALED_H) / 2);

    g_last_touch_tick = lv_tick_get();
    lv_timer_create(frame_tick, FRAME_MS, NULL);
    g_idle_timer = lv_timer_create(idle_tick, CHUFF_NORMAL_IDLE_PERIOD_MS, NULL);
}
