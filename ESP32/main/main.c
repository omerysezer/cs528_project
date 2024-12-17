#include <stdio.h>
#include <driver/i2c.h>
#include <esp_timer.h>
#include <string.h>
#include "driver/uart.h"
#include "../managed_components/espressif__mpu6050/include/mpu6050.h"

#define DELAY_BETWEEN_SAMPLE_MS 2 // 500 hz
#define NUM_SAMPLES 500
#define UART_PORT UART_NUM_0

void i2c_init()
{
    // intiialize I2C
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = GPIO_NUM_13,
        .scl_io_num = GPIO_NUM_14,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 400000,
        .clk_flags = 0
    };

    ESP_ERROR_CHECK(i2c_param_config(I2C_NUM_0, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_NUM_0, I2C_MODE_MASTER, 0, 0, 0));
}

void uart_init()
{
    const uart_port_t uart_num = UART_PORT;
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    // Configure UART parameters
    ESP_ERROR_CHECK(uart_param_config(uart_num, &uart_config));

    ESP_ERROR_CHECK(uart_set_pin(UART_PORT, GPIO_NUM_43, GPIO_NUM_44, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    // Setup UART buffered IO with event queue
    const int uart_buffer_size = (1024 * 2);
    // Install UART driver using an event queue here
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, uart_buffer_size, 0, 0, NULL, 0));
}

mpu6050_handle_t imu_init()
{
    // initialize MPU6050 handler
    i2c_port_t port = I2C_NUM_0;
    const uint16_t imu_address = MPU6050_I2C_ADDRESS;
    mpu6050_handle_t imu_handler = mpu6050_create(port, imu_address);
    assert(imu_handler != NULL);

    ESP_ERROR_CHECK(mpu6050_wake_up(imu_handler));

    const mpu6050_acce_fs_t accel_sensitivity = ACCE_FS_4G;
    const mpu6050_gyro_fs_t gyro_sensitivity = GYRO_FS_500DPS;
    ESP_ERROR_CHECK(mpu6050_config(imu_handler, accel_sensitivity, gyro_sensitivity));

    return imu_handler;
}

void imu_destroy(mpu6050_handle_t imu_handler)
{
    ESP_ERROR_CHECK(mpu6050_sleep(imu_handler));
    mpu6050_delete(imu_handler);
}

struct sample
{
    float a_x;
    float a_y;
    float a_z;
    float g_x;
    float g_y;
    float g_z;
};

void app_main(void)
{
    i2c_init();
    mpu6050_handle_t imu = imu_init();
    uart_init();

    while(true)
    {
        unsigned long t = esp_timer_get_time();
        struct sample s;
        mpu6050_get_acce(imu, (mpu6050_acce_value_t*) &s);
        mpu6050_get_gyro(imu, (mpu6050_gyro_value_t*) &s.g_x);
        printf("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", s.a_x, s.a_y, s.a_z, s.g_x, s.g_y, s.g_z);
        while(esp_timer_get_time() - t < 4000); 
    }
}