"""
Main thread:
    - create ready flag
    - start Thread1, Thread2

Thread 1:
    - last_motion = "nm"
    - motion_opposites = {"nm": None, "left": "right", ....}
    - get SVM model (https://stackoverflow.com/questions/56107259/how-to-save-a-trained-model-by-scikit-learn)
    - establish drone communication
    - set ready flag to true
    - while True:
        - while len(buffer) < NUM_SAMPLES: loop
        - copy last NUM_SAMPLES from buffer into DATA
        - feed DATA into SVM model, get prediction
        - if motion_opposites[motion] == last_motion:
            - last_motion = "nm"
        - else:
            last_motion = motion
        - feed drone last_motion

Thread 2:
    - while ready_flag is not set: loop
    - clear input buffer
    - while True:
        - read from esp, add it to buffer

len(array) = 20_000
start = 19999
"""
import threading
import serial as ser
import numpy as np
import joblib

class DroneFunctions(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.ready_flag = False
        self.end = 0
        self.full_window = False
        self.data_arr = np.array(20000, dtype=np.float64) # cyclic data buffer 

    def data_reader(self):
        while self.ready_flag is False:
            continue

        ser.reset_input_buffer()
        ser.write([0xFF])  # trigger IMU recording for 2 seconds
        
        line = ser.readline().decode("ascii").strip()

        while "y" not in line:
            line = ser.readline().decode("ascii").strip()

        while True:
            line = ser.readline().decode("ascii").strip()
            values = line.split(",")
            
            for value in values:
                self.data_arr[self.end] = value
            
            end+=6
            if end > 2999:
                self.full_window = True
            
    def drone_controller(self):
        last_motion = "nm"
        motion_opposites = {"nm": None, "lflip": None, "rflip": None, "left": "right", "right": "left", "up": "down", "down": "up", "forward": "backward", "backward": "forward", "cw": "ccw", "ccw": "cw"} 

        # load
        clf = joblib.load("ENTER MODEL HERE")
    # - establish drone communication
        self.ready_flag = True

        while True:
            while not self.full_window:
                continue

            data = self.data_arr[self.end-2999:self.end]
            motion = clf.predict(data)

            if motion_opposites[motion] == last_motion:
                last_motion = "nm"
            else:
                last_motion = motion
        # - feed drone last_motion

    def run(self):
        reader = threading.Thread(self.data_reader)
        controller = threading.Thread(self.data_reader)

        reader.start()
        controller.start()

        reader.join()
        controller.join()




if __name__ == "__main__":
    drone = DroneFunctions()
    drone.run()

