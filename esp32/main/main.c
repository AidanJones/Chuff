#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/spi_master.h"
#include "esp_timer.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "esp_check.h"
#include "esp_log.h"
#include "lvgl.h"
#include "esp_lcd_sh8601.h"
#include "esp_lcd_touch_ft5x06.h"
#include "pet_face.h"
#include "pet_sound.h"

static const char *TAG = "pet";

// ---------- Pin definitions ----------
#define LCD_HOST        SPI2_HOST
#define I2C_PORT        I2C_NUM_0

#define PIN_LCD_CS      GPIO_NUM_12
#define PIN_LCD_CLK     GPIO_NUM_11
#define PIN_LCD_D0      GPIO_NUM_4
#define PIN_LCD_D1      GPIO_NUM_5
#define PIN_LCD_D2      GPIO_NUM_6
#define PIN_LCD_D3      GPIO_NUM_7

#define PIN_I2C_SCL     GPIO_NUM_14
#define PIN_I2C_SDA     GPIO_NUM_15
#define PIN_TOUCH_INT   GPIO_NUM_21

// ---------- Display config ----------
#define LCD_H_RES   368
#define LCD_V_RES   448
#define LCD_BPP     16

// ---------- IO expander (TCAL9534 at 0x20) handles LCD + touch reset ----------
#define IO_EXP_ADDR     0x20
#define IO_EXP_REG_OUT  0x01
#define IO_EXP_REG_CFG  0x03
#define IO_EXP_LCD_RST  BIT(0)
#define IO_EXP_DSI_PWR  BIT(1)
#define IO_EXP_TCH_RST  BIT(2)
#define IO_EXP_SD_CS    BIT(7)
#define IO_EXP_OUTPUTS  (IO_EXP_LCD_RST | IO_EXP_DSI_PWR | IO_EXP_TCH_RST | IO_EXP_SD_CS)

// ---------- LVGL ----------
#define LVGL_BUF_LINES  (LCD_V_RES / 4)
#define LVGL_TICK_MS    2
#define LVGL_TASK_STACK (4 * 1024)
#define LVGL_TASK_PRIO  2

static SemaphoreHandle_t lvgl_mux;
static i2c_master_bus_handle_t i2c_bus;
static esp_lcd_touch_handle_t tp;

// ---------- SH8601 init sequence ----------
static const sh8601_lcd_init_cmd_t lcd_init_cmds[] = {
    {0x11, (uint8_t[]){0x00}, 0,   120},
    {0x44, (uint8_t[]){0x01, 0xD1}, 2, 0},
    {0x35, (uint8_t[]){0x00}, 1,   0},
    {0x53, (uint8_t[]){0x20}, 1,   10},
    {0x2A, (uint8_t[]){0x00, 0x00, 0x01, 0x6F}, 4, 0},
    {0x2B, (uint8_t[]){0x00, 0x00, 0x01, 0xBF}, 4, 0},
    {0x51, (uint8_t[]){0x00}, 1,   10},
    {0x29, (uint8_t[]){0x00}, 0,   10},
    {0x51, (uint8_t[]){0xFF}, 1,   0},
};

// ---------- LVGL callbacks ----------

static bool on_flush_ready(esp_lcd_panel_io_handle_t io, esp_lcd_panel_io_event_data_t *edata, void *ctx)
{
    lv_disp_drv_t *drv = (lv_disp_drv_t *)ctx;
    lv_disp_flush_ready(drv);
    return false;
}

static void flush_cb(lv_disp_drv_t *drv, const lv_area_t *area, lv_color_t *buf)
{
    esp_lcd_panel_handle_t panel = (esp_lcd_panel_handle_t)drv->user_data;
    esp_lcd_panel_draw_bitmap(panel, area->x1, area->y1, area->x2 + 1, area->y2 + 1, buf);
}

static void rounder_cb(lv_disp_drv_t *drv, lv_area_t *area)
{
    // SH8601 requires even-aligned coordinates
    area->x1 = (area->x1 >> 1) << 1;
    area->y1 = (area->y1 >> 1) << 1;
    area->x2 = ((area->x2 >> 1) << 1) + 1;
    area->y2 = ((area->y2 >> 1) << 1) + 1;
}

static void touch_cb(lv_indev_drv_t *drv, lv_indev_data_t *data)
{
    esp_lcd_touch_handle_t touch = (esp_lcd_touch_handle_t)drv->user_data;

    if (gpio_get_level(PIN_TOUCH_INT) != 0) {
        data->state = LV_INDEV_STATE_RELEASED;
        return;
    }
    uint16_t x, y;
    uint8_t cnt = 0;
    esp_lcd_touch_read_data(touch);
    bool pressed = esp_lcd_touch_get_coordinates(touch, &x, &y, NULL, &cnt, 1);
    if (pressed && cnt > 0) {
        data->point.x = x;
        data->point.y = y;
        data->state = LV_INDEV_STATE_PRESSED;
    } else {
        data->state = LV_INDEV_STATE_RELEASED;
    }

    /* Trigger pet reaction on the leading edge of each touch press */
    static lv_indev_state_t prev_state = LV_INDEV_STATE_RELEASED;
    if (data->state == LV_INDEV_STATE_PRESSED && prev_state == LV_INDEV_STATE_RELEASED) {
        pet_face_on_touch();
    }
    prev_state = data->state;
}

static void tick_cb(void *arg)
{
    lv_tick_inc(LVGL_TICK_MS);
}

static void lvgl_task(void *arg)
{
    uint32_t delay_ms = LVGL_TICK_MS;
    while (1) {
        if (xSemaphoreTake(lvgl_mux, portMAX_DELAY)) {
            delay_ms = lv_timer_handler();
            xSemaphoreGive(lvgl_mux);
        }
        if (delay_ms > 500) delay_ms = 500;
        if (delay_ms < 1)   delay_ms = 1;
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
}

// ---------- IO expander ----------

static esp_err_t io_exp_write(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t val)
{
    uint8_t buf[] = {reg, val};
    return i2c_master_transmit(dev, buf, sizeof(buf), 100);
}

static esp_err_t io_exp_reset(void)
{
    i2c_master_dev_handle_t dev;
    const i2c_device_config_t cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = IO_EXP_ADDR,
        .scl_speed_hz = 400000,
    };
    ESP_RETURN_ON_ERROR(i2c_master_bus_add_device(i2c_bus, &cfg, &dev), TAG, "IO expander add failed");

    esp_err_t ret = io_exp_write(dev, IO_EXP_REG_CFG, (uint8_t)~IO_EXP_OUTPUTS);
    if (ret == ESP_OK) ret = io_exp_write(dev, IO_EXP_REG_OUT, IO_EXP_SD_CS); // assert resets
    vTaskDelay(pdMS_TO_TICKS(200));
    if (ret == ESP_OK) ret = io_exp_write(dev, IO_EXP_REG_OUT, IO_EXP_OUTPUTS); // deassert resets

    esp_err_t rm = i2c_master_bus_rm_device(dev);
    return (ret == ESP_OK) ? rm : ret;
}

// ---------- app_main ----------

void app_main(void)
{
    static lv_disp_draw_buf_t disp_buf;
    static lv_disp_drv_t disp_drv;

    ESP_LOGI(TAG, "Init I2C");
    const i2c_master_bus_config_t i2c_cfg = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .i2c_port = I2C_PORT,
        .sda_io_num = PIN_I2C_SDA,
        .scl_io_num = PIN_I2C_SCL,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_cfg, &i2c_bus));
    ESP_ERROR_CHECK(io_exp_reset());

    ESP_LOGI(TAG, "Init SPI/QSPI");
    const spi_bus_config_t spi_cfg = SH8601_PANEL_BUS_QSPI_CONFIG(
        PIN_LCD_CLK, PIN_LCD_D0, PIN_LCD_D1, PIN_LCD_D2, PIN_LCD_D3,
        LCD_H_RES * LCD_V_RES * LCD_BPP / 8);
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &spi_cfg, SPI_DMA_CH_AUTO));

    ESP_LOGI(TAG, "Install panel IO");
    esp_lcd_panel_io_handle_t io;
    const esp_lcd_panel_io_spi_config_t io_cfg = SH8601_PANEL_IO_QSPI_CONFIG(
        PIN_LCD_CS, on_flush_ready, &disp_drv);
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &io_cfg, &io));

    ESP_LOGI(TAG, "Install SH8601 panel driver");
    sh8601_vendor_config_t vendor_cfg = {
        .init_cmds = lcd_init_cmds,
        .init_cmds_size = sizeof(lcd_init_cmds) / sizeof(lcd_init_cmds[0]),
        .flags.use_qspi_interface = 1,
    };
    const esp_lcd_panel_dev_config_t panel_cfg = {
        .reset_gpio_num = -1,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB,
        .bits_per_pixel = LCD_BPP,
        .vendor_config = &vendor_cfg,
    };
    esp_lcd_panel_handle_t panel;
    ESP_ERROR_CHECK(esp_lcd_new_panel_sh8601(io, &panel_cfg, &panel));
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel, true));

    ESP_LOGI(TAG, "Init touch (FT3168/FT5x06)");
    esp_lcd_panel_io_handle_t tp_io;
    esp_lcd_panel_io_i2c_config_t tp_io_cfg = ESP_LCD_TOUCH_IO_I2C_FT5x06_CONFIG();
    tp_io_cfg.scl_speed_hz = 400000;
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_i2c(i2c_bus, &tp_io_cfg, &tp_io));

    const esp_lcd_touch_config_t tp_cfg = {
        .x_max = LCD_H_RES,
        .y_max = LCD_V_RES,
        .rst_gpio_num = -1,
        .int_gpio_num = PIN_TOUCH_INT,
        .levels.reset = 0,
        .levels.interrupt = 0,
    };
    ESP_ERROR_CHECK(esp_lcd_touch_new_i2c_ft5x06(tp_io, &tp_cfg, &tp));

    ESP_LOGI(TAG, "Init LVGL");
    lv_init();
    lv_color_t *buf1 = heap_caps_malloc(LCD_H_RES * LVGL_BUF_LINES * sizeof(lv_color_t), MALLOC_CAP_DMA);
    lv_color_t *buf2 = heap_caps_malloc(LCD_H_RES * LVGL_BUF_LINES * sizeof(lv_color_t), MALLOC_CAP_DMA);
    assert(buf1 && buf2);
    lv_disp_draw_buf_init(&disp_buf, buf1, buf2, LCD_H_RES * LVGL_BUF_LINES);

    lv_disp_drv_init(&disp_drv);
    disp_drv.hor_res = LCD_H_RES;
    disp_drv.ver_res = LCD_V_RES;
    disp_drv.flush_cb = flush_cb;
    disp_drv.rounder_cb = rounder_cb;
    disp_drv.draw_buf = &disp_buf;
    disp_drv.user_data = panel;
    lv_disp_t *disp = lv_disp_drv_register(&disp_drv);

    static lv_indev_drv_t indev_drv;
    lv_indev_drv_init(&indev_drv);
    indev_drv.type = LV_INDEV_TYPE_POINTER;
    indev_drv.disp = disp;
    indev_drv.read_cb = touch_cb;
    indev_drv.user_data = tp;
    lv_indev_drv_register(&indev_drv);

    esp_timer_handle_t tick_timer;
    const esp_timer_create_args_t tick_args = {.callback = tick_cb, .name = "lvgl_tick"};
    ESP_ERROR_CHECK(esp_timer_create(&tick_args, &tick_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(tick_timer, LVGL_TICK_MS * 1000));

    lvgl_mux = xSemaphoreCreateMutex();
    xTaskCreate(lvgl_task, "lvgl", LVGL_TASK_STACK, NULL, LVGL_TASK_PRIO, NULL);

    ESP_LOGI(TAG, "Init sound");
    pet_sound_init(i2c_bus);

    ESP_LOGI(TAG, "Init Flowey face");
    if (xSemaphoreTake(lvgl_mux, portMAX_DELAY)) {
        pet_face_init();
        xSemaphoreGive(lvgl_mux);
    }
}
