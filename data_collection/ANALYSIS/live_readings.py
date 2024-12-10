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
import serial

class DroneFunctions(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.ready_flag = False

    def data_reader(self):
        while self.ready_flag is False:
            continue




    def drone_controller(self):
        pass

    def run(self):
        reader = threading.Thread(self.data_reader)
        controller = threading.Thread(self.data_reader)

        reader.start()
        controller.start()

        reader.join()
        controller.join()




if __name__ == "__main__":
    pass
