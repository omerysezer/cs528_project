"""
- get SVM model (https://stackoverflow.com/questions/56107259/how-to-save-a-trained-model-by-scikit-learn)
"""

import threading
import serial
import numpy as np
from time import time, sleep
from svm import load_svm_from_file, train_svm
import socket
import joblib
import pandas as pd
import matplotlib.pyplot as plt


class DroneFunctions(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.ready_flag = False
        self.end = 0
        self.full_window = False
        self.data_arr = np.zeros(30000, dtype=np.float32)  # cyclic data buffer
        self.window_size = 500 * 6

        self.threshold = 11
        self.motion_detected = False
        self.motion_window_start = 0

    def data_reader(self):
        while self.ready_flag is False:
            continue

        ser = serial.Serial(port="com3", baudrate=115200)
        ser.set_output_flow_control(False)
        ser.reset_input_buffer()
        ser.setRTS(False)
        ser.close()
        ser.open()

        sleep(3)
        ser.reset_input_buffer()
        ser.write([0xFF])  # trigger IMU recording for 2 seconds

        while "y" not in (line := ser.readline().decode("ascii").strip()):
            continue

        count = 0
        while True:
            line = ser.readline().decode("ascii").strip()
            count += 1
            values = line.split(",")

            for i, value in enumerate(values):
                self.data_arr[self.end + i] = float(value)

            self.end = (self.end + 6) % len(self.data_arr)
            if self.end >= self.window_size:
                self.full_window = True

    def drone_controller(self):
        # host = ''
        # port = 9000
        # locaddr = (host,port)
        # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # tello_address = ('192.168.10.1', 8889)
        # sock.bind(locaddr)

        # data, server = sock.recvfrom(1518)
        print("drone_controller start")
        classifier = load_svm_from_file()

        print("classifier loaded")

        last_motion = "nm"
        motion_opposites = {
            "nm": None,
            "lflip": None,
            "rflip": None,
            "left": "right",
            "right": "left",
            "up": "down",
            "down": "up",
            "forward": "backward",
            "backward": "forward",
            "cw": "ccw",
            "ccw": "cw",
        }
        motion_to_command = {
            "nm": None,
            "lflip": "lflip",
            "rflip": "rflip",
            "left": "left 20",
            "right": "right 20",
            "up": "up 20",
            "down": "down 20",
            "forward 20": "forward 20",
            "backward": "back 20",
            "cw": "cw 1",
            "ccw": "ccw 1",
        }
        repeat_motion_timer = None

        self.ready_flag = True

        print("Ready flag set to true")
        while True:
            while not self.full_window:
                sleep(1)
                print("waiting for full window")
                continue

            reading_magnitude = np.linalg.norm(self.data_arr[self.end - 6 : self.end])
            print("motion" if reading_magnitude > self.threshold else "nm", reading_magnitude)
            if reading_magnitude > self.threshold and not self.motion_detected:
                self.motion_detected = True
                self.motion_window_start = self.end
            elif reading_magnitude <= self.threshold and self.motion_detected:
                # print(1)
                self.motion_detected = False
                if self.end <= self.motion_window_start:
                    # print(2)
                    window_len = len(self.data_arr) - self.motion_window_start + self.end
                    window_midpoint = (self.motion_window_start + window_len // 2) % len(self.data_arr)
                else:
                    # print(3)
                    window_midpoint = (self.motion_window_start + self.end) // 2

                # print(4)
                while True:
                    print(self.end - window_midpoint)
                    if self.end >= window_midpoint and self.end - window_midpoint > 250 * 6:
                        break
                    if (
                        self.end < window_midpoint
                        and len(self.data_arr) - window_midpoint + self.end > 250 * 6
                    ):
                        break

                # print('break')
                if self.end - self.window_size < 0:
                    data = np.concatenate(
                        (self.data_arr[self.end - self.window_size :], self.data_arr[: self.end])
                    )
                else:
                    data = self.data_arr[self.end - self.window_size : self.end]

                data = data.reshape(-1, 6)
                data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
                data = data.flatten()
                sleep(0.1)
                motion = classifier.predict([data])[0]
                print(motion)
                sleep(2)
                # sleep(2)
                # sleep(0.5)
                # if cancelling previous motion
                # if motion_opposites[motion] == last_motion:
                #     curr_motion = "nm"
                # # repeating motion
                # elif motion == "nm":
                #     curr_motion = last_motion
                # # new motion
                # else:
                #     curr_motion = motion

                # if curr_motion == "nm":
                #     continue

                # if curr_motion != last_motion:
                #     # this is a new movement so remove the repeat_motion_timer
                #     repeat_motion_timer = None

                # # send the motion if the timer is not blocking us
                # if repeat_motion_timer is None or time() - repeat_motion_timer > 5:
                #     # cmd = motion_to_command[curr_motion]
                #     # cmd = cmd.encode(encoding="utf-8")
                #     # sent = sock.sendto(cmd, tello_address)
                #     repeat_motion_timer = time()

                # last_motion = curr_motion

    def run(self):
        reader = threading.Thread(target=self.data_reader)
        controller = threading.Thread(target=self.drone_controller)

        print("starting")
        reader.start()
        controller.start()

        print("finishing")
        reader.join()
        controller.join()


if __name__ == "__main__":
    drone = DroneFunctions()
    drone.run()
