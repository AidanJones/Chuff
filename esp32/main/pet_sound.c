#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "es8311.h"
#include "pet_sound.h"

#define SAMPLE_RATE    16000
#define MCLK_MULTIPLE  384          /* MCLK = 16000 * 384 = 6 144 000 Hz */
#define VOICE_VOLUME   85
#define PA_GPIO        GPIO_NUM_46  /* power amplifier enable, HIGH = on */
#define I2S_MCLK       GPIO_NUM_16
#define I2S_BCLK       GPIO_NUM_9
#define I2S_WS         GPIO_NUM_45
#define I2S_DOUT       GPIO_NUM_8
#define AMPLITUDE      12000        /* 16-bit half-scale, ES8311 HW vol handles the rest */
#define BUF_FRAMES     512

typedef struct { uint32_t hz; uint32_t ms; } note_t;

static const note_t JINGLE[] = {
    {523,  75},   /* C5 */
    {659,  75},   /* E5 */
    {784,  75},   /* G5 */
    {1047, 115},  /* C6 */
};
#define JINGLE_LEN   (int)(sizeof(JINGLE) / sizeof(JINGLE[0]))
#define NOTE_GAP_MS  12

static const char *TAG = "snd";
static i2s_chan_handle_t g_tx      = NULL;
static volatile bool     g_playing = false;
static int16_t           g_buf[BUF_FRAMES * 2]; /* stereo; static to keep off task stack */

static void write_tone(uint32_t freq_hz, uint32_t dur_ms, bool silent)
{
    uint32_t total = (uint64_t)SAMPLE_RATE * dur_ms / 1000;
    uint32_t half  = freq_hz ? (SAMPLE_RATE / (2 * freq_hz)) : 1;
    uint32_t phase = 0;
    int16_t  level = silent ? 0 : AMPLITUDE;

    while (total) {
        uint32_t chunk = (total < BUF_FRAMES) ? total : BUF_FRAMES;
        for (uint32_t i = 0; i < chunk; i++) {
            g_buf[i * 2]     = level;
            g_buf[i * 2 + 1] = level;
            if (!silent && ++phase >= half) {
                phase = 0;
                level = -level;
            }
        }
        size_t written;
        i2s_channel_write(g_tx, g_buf, chunk * 4, &written, pdMS_TO_TICKS(500));
        total -= chunk;
    }
}

static void play_task(void *arg)
{
    for (int i = 0; i < JINGLE_LEN; i++) {
        uint32_t sound_ms = JINGLE[i].ms > NOTE_GAP_MS
                            ? JINGLE[i].ms - NOTE_GAP_MS : JINGLE[i].ms;
        write_tone(JINGLE[i].hz, sound_ms, false);
        write_tone(0, NOTE_GAP_MS, true);
    }
    g_playing = false;
    vTaskDelete(NULL);
}

void pet_sound_init(i2c_master_bus_handle_t i2c_bus)
{
    /* Enable power amplifier */
    gpio_config_t pa = {
        .pin_bit_mask = 1ULL << PA_GPIO,
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&pa));
    gpio_set_level(PA_GPIO, 1);

    /* I2S TX — TX only, no RX needed for playback */
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &g_tx, NULL));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT,
                                                         I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_MCLK,
            .bclk = I2S_BCLK,
            .ws   = I2S_WS,
            .dout = I2S_DOUT,
            .din  = GPIO_NUM_NC,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.clk_cfg.mclk_multiple = MCLK_MULTIPLE;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(g_tx, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(g_tx));

    /* ES8311 codec over the shared I2C bus */
    es8311_handle_t es = es8311_create(i2c_bus, ES8311_ADDRESS_0);
    if (!es) {
        ESP_LOGE(TAG, "es8311_create failed");
        return;
    }
    const es8311_clock_config_t clk = {
        .mclk_inverted      = false,
        .sclk_inverted      = false,
        .mclk_from_mclk_pin = true,
        .mclk_frequency     = SAMPLE_RATE * MCLK_MULTIPLE,
        .sample_frequency   = SAMPLE_RATE,
    };
    ESP_ERROR_CHECK(es8311_init(es, &clk, ES8311_RESOLUTION_16, ES8311_RESOLUTION_16));
    ESP_ERROR_CHECK(es8311_voice_volume_set(es, VOICE_VOLUME, NULL));
    ESP_ERROR_CHECK(es8311_microphone_config(es, false));

    ESP_LOGI(TAG, "ES8311 ready — built-in speaker active");
}

void pet_sound_play(void)
{
    if (!g_tx || g_playing) return;
    g_playing = true;
    xTaskCreate(play_task, "snd", 2048, NULL, 5, NULL);
}
