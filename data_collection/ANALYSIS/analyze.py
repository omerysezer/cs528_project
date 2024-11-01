from scipy.fft import fft, fftfreq
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import serial
from time import sleep


HELLO = """
                                                                     .-'''-.           
                                     .---. .---.    '   _    \    .'/   \  
   .                  __.....__      |   | |   |  /   /` '.   \  / /     \ 
 .'|              .-''         '.    |   | |   | .   |     \  '  | |     | 
<  |             /     .-''"'-.  `.  |   | |   | |   '      |  ' | |     | 
 | |            /     /________\   \ |   | |   | \    \     / /  |/`.   .' 
 | | .'''-.     |                  | |   | |   |  `.   ` ..' /    `.|   |  
 | |/.'''. \    \    .-------------' |   | |   |     '-...-'`      ||___|  
 |  /    | |     \    '-.____...---. |   | |   |                   |/___/  
 | |     | |      `.             .'  |   | |   |                   .'.--.  
 | |     | |        `''-...... -'    '---' '---'                  | |    | 
 | '.    | '.                                                     \_\    / 
 '---'   '---'                                                     `''--'   
"""

GOODBYE = """
               .-'''-.         .-'''-.                                                                         ___   
              '   _    \      '   _    \   _______                                                          .'/   \  
            /   /` '.   \   /   /` '.   \  \  ___ `'.    /|                                 __.....__      / /     \ 
  .--./)   .   |     \  '  .   |     \  '   ' |--.\  \   ||          .-.          .-    .-''         '.    | |     | 
 /.''\\\\    |   '      |  ' |   '      |  '  | |    \  '  ||           \ \        / /   /     .-''"'-.  `.  | |     | 
| |  | |   \    \     / /  \    \     / /   | |     |  ' ||  __        \ \      / /   /     /________\   \ |/`.   .' 
 \`-' /     `.   ` ..' /    `.   ` ..' /    | |     |  | ||/'__ '.      \ \    / /    |                  |  `.|   |  
 /("'`         '-...-'`        '-...-'`     | |     ' .' |:/`  '. '      \ \  / /     \    .-------------'   ||___|  
 \ '---.                                    | |___.' /'  ||     | |       \ `  /       \    '-.____...---.   |/___/  
  /'""'.\                                  /_______.'/   ||\    / '        \  /         `.             .'    .'.--.  
 ||     ||                                 \_______|/    |/\\'..' /         / /            `''-...... -'     | |    | 
 \\'. __//                                                '  `'-'`      |`-' /                               \_\    / 
  `'---'                                                                '..'                                 `''--'  
"""


def get_data(file_path):
    with open(file_path, "r") as f:
        data = f.read()
        first_line = data.index("acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z")
        data = data[first_line:]  # remove boot up lines
        data = data.splitlines()[:-1]  # remove final line which is end of code
        data = StringIO("\n".join(data))

        df = pd.read_csv(data, sep=",")

        df["time"] -= df["time"][0]  # normalize times to start at 0
        df["time"] /= 1_000_000  # conert microseconds to seconds

        return df


def plot_data(dataframe, filename):
    shift_z = ""
    yes = ["yes", "y"]
    no = ["no", "n"]
    while shift_z not in yes and shift_z not in no:
        shift_z = input(
            "Shift Z-axis down [z_signal will be z_signal = z_signal - avg(z_signal). This helps deal with z-axis being shifted up due to gravity] [Y/N]? "
        )
        shift_z = shift_z.lower()

    df = dataframe

    if shift_z:
        df["acc_z"] -= sum(df["acc_z"]) / len(df["acc_z"])

    fig = plt.figure()

    fig.suptitle(f"{filename} analysis", fontsize=16)
    fig.subplots_adjust(hspace=0.3)

    acc_plot = fig.add_subplot(221)
    acc_plot.plot(df["time"], df["acc_x"], label="acc_x")
    acc_plot.plot(df["time"], df["acc_y"], label="acc_y")
    acc_plot.plot(df["time"], df["acc_z"], label="acc_z")
    acc_plot.title.set_text("Acceleration vs Time")
    acc_plot.set(xlabel="Time (seconds)", ylabel=r"Acceleration ($m/s^{2}$)")
    acc_plot.set_xlim(xmin=0)
    acc_plot.legend()

    gyro_plot = fig.add_subplot(222)
    gyro_plot.plot(df["time"], df["gyro_x"], label="gyro_x")
    gyro_plot.plot(df["time"], df["gyro_y"], label="gyro_y")
    gyro_plot.plot(df["time"], df["gyro_z"], label="gyro_z")
    gyro_plot.title.set_text("Gyro vs Time")
    gyro_plot.set(xlabel="Time (seconds)", ylabel="Angular Velocity (degrees/second)")
    gyro_plot.set_xlim(xmin=0)
    gyro_plot.legend()

    sample_timing_differences = [df["time"][i] - df["time"][i - 1] for i in range(1, len(df["time"]))]
    sample_spacing = round(sum(sample_timing_differences) / len(sample_timing_differences), 3)
    num_samples = len(df["acc_x"])

    fft_freqs = fftfreq(num_samples, sample_spacing)[: num_samples // 2]
    acc_x_fft = 2.0 / num_samples * np.abs(fft(df["acc_x"])[0 : num_samples // 2])
    acc_y_fft = 2.0 / num_samples * np.abs(fft(df["acc_y"])[0 : num_samples // 2])
    acc_z_fft = 2.0 / num_samples * np.abs(fft(df["acc_z"])[0 : num_samples // 2])

    for i in fft(df["acc_x"])[0 : num_samples // 2]:
        print(i)

    acc_fft_plot = fig.add_subplot(223)
    acc_fft_plot.plot(fft_freqs, acc_x_fft, label="acc_x fft")
    acc_fft_plot.plot(fft_freqs, acc_y_fft, label="acc_y fft")
    acc_fft_plot.plot(fft_freqs, acc_z_fft, label="acc_z fft")
    acc_fft_plot.title.set_text("FFT of Acceleration")
    acc_fft_plot.set(xlabel="Frequency (HZ)", ylabel="Magnitude")
    acc_fft_plot.set_xlim(xmin=-1)
    acc_fft_plot.legend()

    gyro_x_fft = 2.0 / num_samples * np.abs(fft(df["gyro_x"])[0 : num_samples // 2])
    gyro_y_fft = 2.0 / num_samples * np.abs(fft(df["gyro_y"])[0 : num_samples // 2])
    gyro_z_fft = 2.0 / num_samples * np.abs(fft(df["gyro_z"])[0 : num_samples // 2])

    gyro_fft_plot = fig.add_subplot(224)
    gyro_fft_plot.plot(fft_freqs, gyro_x_fft, label="gyro_x fft")
    gyro_fft_plot.plot(fft_freqs, gyro_y_fft, label="gyro_y fft")
    gyro_fft_plot.plot(fft_freqs, gyro_z_fft, label="gyro_z fft")
    gyro_fft_plot.title.set_text("FFT of Gyro")
    gyro_fft_plot.set(xlabel="Frequency (HZ)", ylabel="Magnitude")
    gyro_fft_plot.set_xlim(xmin=-1)
    gyro_fft_plot.legend()

    plt.show()

    return df


if __name__ == "__main__":
    print(HELLO)

    while True:
        try:
            analyze = input("Record new data (1) or analyze pre-existing data (2)? ")
            analyze = int(analyze)
            if analyze > 2 or analyze < 0:
                raise ValueError()
            break
        except ValueError:
            print("\rInvalid Option. Record new data (1) or analyze pre-existing data (2)? ")
        except KeyboardInterrupt:
            print(GOODBYE)
            exit(0)

    if analyze == 2:
        while True:
            try:
                file_name = input("Enter data file path: ")
                with open(file_name) as f:
                    plot_data(get_data(file_name), file_name)
            except FileNotFoundError:
                print(f"{file_name} not found. ")
            except KeyboardInterrupt:
                print(GOODBYE)
                exit(0)

    ser = None
    try:
        port = input("Enter the UART port (COM?): ")
        ser = serial.Serial(port=port, baudrate=115200)
        ser.set_output_flow_control(False)
        ser.setRTS(False)
        ser.close()
        ser.open()

        while True:
            file_name = input(
                "Enter a filename to save readings into (if file exists it will be overwritten): "
            )
            with open(file_name, "w+") as f:
                print("\nIMU will begin recording data in 3 seconds", end="", flush=True)
                sleep(1)
                print("\rIMU will begin recording data in 2 seconds", end="", flush=True)
                sleep(1)
                print("\rIMU will begin recording data in 1 seconds", end="", flush=True)
                sleep(1)
                print("\rDATA RECORDING IN PROGRESS" + " " * 100, end="", flush=True)

                ser.write([0xFF])  # trigger IMU recording for 2 seconds
                data = ""

                while True:
                    line = ser.readline().decode("ascii").strip()

                    if line == "$":  # recieved end of transmission
                        break

                    if data:
                        data += line + "\n"
                    elif "acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,time" in line:
                        data += line[line.index("acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,time") :] + "\n"
                        print("\rDATA RECORDING FINISHED. SAVING TO FILE." + " " * 100, end="", flush=True)

                f.write(data)
                sleep(1.5)
                print(f"\rDATA SAVED TO '{file_name}'" + " " * 100 + "\n", flush=True)
                sleep(2)
                analyze = input("Analyze data now? [Y/N]: ")
                if analyze.lower() == "y" or analyze.lower() == "yes":
                    plot_data(get_data(file_name), file_name)
    except KeyboardInterrupt:
        if ser:
            print("\rCLOSING PORT..." + " " * 100)
            ser.close()
        print(GOODBYE)
        exit(0)
