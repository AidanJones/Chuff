#pragma once
#include "driver/i2c_master.h"

void pet_sound_init(i2c_master_bus_handle_t i2c_bus);
void pet_sound_play(void);
