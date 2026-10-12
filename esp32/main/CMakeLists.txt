idf_component_register(
    SRCS "main.c" "pet_face.c" "pet_sound.c"
    INCLUDE_DIRS ""
    REQUIRES es8311 esp_driver_i2s
)
target_link_libraries(${COMPONENT_LIB} PRIVATE m)
