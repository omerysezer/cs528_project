import threading
import serial
import numpy as np
from time import sleep
from svm import load_svm_from_file
import socket

NUM_SAMPLES_PER_WINDOW = 500
NUM_VALUES_PER_SAMPLE = 6
NUM_VALUES_PER_WINDOW = NUM_SAMPLES_PER_WINDOW * NUM_VALUES_PER_SAMPLE
VALUES_DTYPE = np.float32
MOVEMENT_THRESHOLD = 20

run = False

data = np.zeros(NUM_VALUES_PER_WINDOW * 10, dtype=VALUES_DTYPE)
end = 0
buffer_full = False


def data_reader():
    global data, run, end, buffer_full, NUM_SAMPLES_PER_WINDOW, VALUES_DTYPE, NUM_VALUES_PER_SAMPLE, NUM_SAMPLES_PER_WINDOW

    ser = serial.Serial(port="com12", baudrate=115200)
    ser.set_output_flow_control(False)
    ser.reset_input_buffer()
    ser.setRTS(False)
    ser.close()
    ser.open()

    ser.reset_input_buffer()

    while run:
        sleep(0.001)
        line = ser.readline().decode("ascii").strip()

        try:
            values = [float(val) for val in line.split(",")]
            values = np.array(values, dtype=VALUES_DTYPE)
        except:
            continue

        if len(values) != 6:
            continue

        data[end : end + NUM_VALUES_PER_SAMPLE] = values
        end = (end + NUM_VALUES_PER_SAMPLE) % len(data)
        if end >= NUM_VALUES_PER_WINDOW:
            buffer_full = True

    ser.close()


def drone_controller():
    global data, run, end, buffer_full, NUM_SAMPLES_PER_WINDOW, VALUES_DTYPE, NUM_VALUES_PER_SAMPLE, MOVEMENT_THRESHOLD, NUM_VALUES_PER_WINDOW

    # DEFINE VARIABLES
    movement_window_start = 0
    movement_in_progress = False

    position_bounds = [1, 1, 1]  # Max distance from center in x,y,z axes
    # Current x,y,z coordinates within the space, center of the bottom surface of the bounding box
    curr_position = [0, 0, -1]

    classifier_svm = load_svm_from_file("./classifiers/motion_classifier.mdl")

    ignore_motion = False

    host = ""
    port = 9000
    locaddr = (host, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1)
    tello_address = ("192.168.10.1", 8889)
    sock.bind(locaddr)

    sock.sendto("command".encode("utf-8"), tello_address)

    while not buffer_full and run:
        print("Preparing drone...")
        sleep(1)

    print("Ready To Control Drone!")
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
            print("MOvement Detected")
            continue
        # trick to avoid indenting for movement_ended case
        elif not movement_ended:
            continue

        print("Movement ended")
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
                NUM_VALUES_PER_WINDOW // 2
            ):
                break
            if curr_window_end_looped_around and len(data) - movement_window_midpoint + end > (
                NUM_VALUES_PER_WINDOW // 2
            ):
                break

        curr_window_end = end
        if curr_window_end - NUM_VALUES_PER_WINDOW < 0:
            window_data = np.concatenate(
                (
                    data[curr_window_end - NUM_VALUES_PER_WINDOW :],
                    data[:curr_window_end],
                )
            )
        else:
            window_data = data[curr_window_end - NUM_VALUES_PER_WINDOW : curr_window_end]

        # reshape into 6 columns acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        window_data = window_data.reshape(-1, 6)
        normalized_window_data = (window_data - window_data.min(axis=0)) / (
            window_data.max(axis=0) - window_data.min(axis=0)
        )

        motion = classifier_svm.predict([normalized_window_data.flatten()])[0].strip()
        print(f"Detected: {motion}")

        if ignore_motion:  # treat every other motion as a return to base stance, ignore them
            ignore_motion = False
            print("Motion ignored")
            sleep(2)
            print("ready")
            continue

        # ignore motion is meant to allow users to return hand to steady state
        # flip motions already return hand to steady state, so no need to ignore
        # motions after a flip
        if motion not in ["rflip", "lflip"]:
            ignore_motion = True

        if motion == "nm":
            sleep(2)
            print("ready")
            continue

        updated_position = curr_position.copy()
        out_of_bounds = False
        match motion:
            case "left" | "right":
                updated_position[0] += -0.2 if motion == "left" else 0.2
            case "forward" | "backward":
                updated_position[1] += -0.2 if motion == "backward" else 0.2
            case "up" | "down":
                updated_position[2] += -0.2 if motion == "down" else 0.2
            case _:
                pass

        print(f"Original Pos: {curr_position}")
        print(f"New Computed Pos: {updated_position}")
        for curr_axis_pos, curr_axis_bounds in zip(updated_position, position_bounds):
            if abs(curr_axis_pos) > curr_axis_bounds:
                out_of_bounds = True

        print("out of bounds: ", out_of_bounds)

        if out_of_bounds:
            print("Out of bounds")
            sleep(2)
            continue

        drone_command = motion
        print("Drone Command: ", drone_command)
        if drone_command == "lflip":
            drone_command = "flip l"
        elif drone_command == "rflip":
            drone_command = "flip r"
        elif drone_command == "backward":
            drone_command = "back"

        print("Massaged Drone Command: ", drone_command)
        # add min travel distance to commands that need it
        if drone_command in ["up", "down", "left", "right", "forward", "back"]:
            drone_command += " 50"
        print("Drone Command With Dist: ", drone_command)

        # HANDLE SPECIAL MOVEMENT CASES
        is_landed = curr_position[2] == -position_bounds[2]
        is_landing = not is_landed and updated_position[2] == -1 * position_bounds[2]
        if is_landed and motion == "up":
            print("Sending: ", drone_command)
            sock.sendto(drone_command.encode("utf-8"), tello_address)
            drone_command = "takeoff"
        elif is_landed and motion != "up":
            print("pull up!")
            continue
        elif is_landing:
            print("Sending: ", drone_command)
            sock.sendto(drone_command.encode("utf-8"), tello_address)
            drone_command = "land"

        print("Sending: ", drone_command)
        sock.sendto(drone_command.encode("utf-8"), tello_address)
        try:
            d, server = sock.recvfrom(1518)
            print(d.decode(encoding="utf-8"))
        except:
            print("err response")

        curr_position = updated_position.copy()

        print("\n\n\n\n")
        sleep(2)
        print("ready")

    sock.sendto("emergency".encode("utf-8"), tello_address)


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
