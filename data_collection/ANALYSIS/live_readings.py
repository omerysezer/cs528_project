import threading
import serial
import numpy as np
from time import sleep
from svm import load_svm_from_file

SAMPLES_PER_WINDOW = 500
VALUES_PER_SAMPLE = 6
VALUES_DTYPE = np.float32
MOVEMENT_THRESHOLD = 17

run = False

data = np.zeros(SAMPLES_PER_WINDOW * VALUES_PER_SAMPLE * 10, dtype=VALUES_DTYPE)
end = 0
buffer_full = False


def data_reader():
    global data, run, end, buffer_full, SAMPLES_PER_WINDOW, VALUES_DTYPE, VALUES_PER_SAMPLE

    ser = serial.Serial(port="com12", baudrate=115200)
    ser.set_output_flow_control(False)
    ser.reset_input_buffer()
    ser.setRTS(False)
    ser.close()
    ser.open()

    ser.reset_input_buffer()

    while run:
        line = ser.readline().decode("ascii").strip()

        try:
            values = [float(val) for val in line.split(",")]
            values = np.array(values, dtype=VALUES_DTYPE)
        except:
            continue

        if len(values) != 6:
            continue

        data[end : end + VALUES_PER_SAMPLE] = values
        end = (end + VALUES_PER_SAMPLE) % len(data)
        if end >= SAMPLES_PER_WINDOW * VALUES_PER_SAMPLE:
            buffer_full = True

    ser.close()


def drone_controller():
    global data, run, end, buffer_full, SAMPLES_PER_WINDOW, VALUES_DTYPE, VALUES_PER_SAMPLE, MOVEMENT_THRESHOLD

    # DEFINE VARIABLES
    movement_window_start = 0
    movement_in_progress = False
    
    classifier = load_svm_from_file()

    while not buffer_full and run:
        print("filling")
        sleep(0.01)

    print("Ready")
    while run:
        sleep(0.1)
        curr_window_end = end

        # avoid taking a slice from - to + values, which is invalid
        if curr_window_end == 0:
            continue

        latest_reading_magnitude = np.linalg.norm(data[curr_window_end - 6 : curr_window_end])
        movement_is_occurring = latest_reading_magnitude >= MOVEMENT_THRESHOLD

        movement_started = movement_is_occurring and not movement_in_progress
        movement_ended = not movement_is_occurring and movement_in_progress

        if movement_started:
            movement_in_progress = True
            movement_window_start = curr_window_end
            print("Movement")
            continue
        # trick to avoid indenting for movement_ended case
        elif not movement_ended:
            continue
        
        print("No movement")
        # This section only runs if movement_ended is True
        movement_in_progress = False

        # find the midpoint of the window during which the motion occurred
        end_looped_around = curr_window_end < movement_window_start
        if end_looped_around:
            movement_window_length = len(data) - movement_window_start + curr_window_end
            movement_window_midpoint = (movement_window_start + movement_window_length // 2) % len(data)
        else:
            movement_window_midpoint = (movement_window_start + curr_window_end) // 2

        # wait for half of a standard 2 second window to pass so we have a full 2 second window to feed to classifier
        while True:
            sleep(0.01)
            curr_window_end = end
            curr_window_end_looped_around = curr_window_end < movement_window_midpoint
            if not curr_window_end_looped_around and curr_window_end - movement_window_midpoint > (
                VALUES_PER_SAMPLE * SAMPLES_PER_WINDOW // 2
            ):
                break
            if curr_window_end_looped_around and len(data) - movement_window_midpoint + end > (
                VALUES_PER_SAMPLE * SAMPLES_PER_WINDOW // 2
            ):
                break
        
        curr_window_end = end
        if curr_window_end - (VALUES_PER_SAMPLE * SAMPLES_PER_WINDOW) < 0:
            window_data = np.concatenate(
                (data[curr_window_end - (VALUES_PER_SAMPLE * SAMPLES_PER_WINDOW) :], data[: curr_window_end])
            )
            print(window_data.shape, 'a')
        else:
            window_data = data[curr_window_end - (VALUES_PER_SAMPLE * SAMPLES_PER_WINDOW) : curr_window_end]
            print(window_data.shape, curr_window_end, 'b')

        # reshape into 6 columns acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        window_data = window_data.reshape(-1, 6)
        normalized_window_data = (window_data - window_data.min(axis=0)) / (window_data.max(axis=0) - window_data.min(axis=0))
        motion = classifier.predict([normalized_window_data.flatten()])[0]
        print(motion)


if __name__ == "__main__":
    reader = threading.Thread(target=data_reader)
    controller = threading.Thread(target=drone_controller)

    try:
        run = True
        reader.start()
        controller.start()
        while True:
            sleep(0.01)
    except KeyboardInterrupt:
        run = False
        reader.join()
        controller.join()
        exit(0)
