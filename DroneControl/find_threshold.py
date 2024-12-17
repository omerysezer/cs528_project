import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd 

sample_points = np.zeros(500, dtype=np.float64)
num_files = 0
directory = input("Enter path to data directory: ")

for root, _, files in os.walk(directory):
    num_files = len(files)
    for file in files:
        fname = file.split('_')
        name = fname[0]
        
        df = pd.read_csv(directory+'/'+file, sep=",")

        # Keep only accelerometer and gyroscope signals
        data = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values.astype(np.float32)
        i = 0

        for line in data:
            mag = np.linalg.norm(line)
            mag = math.log((mag+1))
            
            sample_points[i] += mag
            i+= 1

for val in range (0,len(sample_points)):
    sample_points[val] = math.exp((sample_points[val]/num_files))

plt.plot(sample_points)
plt.show()