from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import serial
from time import sleep, time
import os


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


def get_input(input_request_str, input_validator=(lambda inpt: True)):
    """
    Will continually print the input_request string until input_validator(user_input) returns True
    Returns first user_input that satisfies input_validator
    """
    while True:
        inpt = input(input_request_str)
        if input_validator(inpt):
            return inpt


def get_input_yes_no(input_request_str):
    """
    Will continually prompt the user for a YES/NO answer to the given input_request_str
    Will return True if the response is YES, False otherwise
    """
    is_yes = lambda inpt: inpt.lower() in ["yes", "y"]
    is_no = lambda inpt: inpt.lower() in ["no", "n"]
    validator = lambda inpt: is_yes(inpt) or is_no(inpt)

    response = get_input(input_request_str, validator)
    return is_yes(response)


def record_data():
    valid_movements = ["up", "down", "left", "right", "forward", "backward", "lflip", "rflip", "cw", "ccw"]

    def movement_validator(movement):
        if movement.lower() in valid_movements:
            return True
        print("Invalid movment, enter one of: ", valid_movements)
        return False

    def num_movement_validator(inpt):
        try:
            inpt = int(inpt)
            return True
        except ValueError:
            return False

    try:
        port = get_input("Enter the UART port (COM?): ")

        ser = serial.Serial(port=port, baudrate=115200)
        ser.set_output_flow_control(False)
        ser.reset_input_buffer()
        ser.setRTS(False)
        ser.close()
        ser.open()

        while True:
            loop_length = int(
                get_input(
                    "How many movements do you want to record? [Enter a number or 0 for infinite loop]: ",
                    num_movement_validator,
                )
            )
            movement_class = get_input("Enter movement type: ", movement_validator).lower()
            file_suffix = get_input(
                "Enter your initials (1 or 2 characters only): ", lambda inpt: 0 < len(inpt) < 3
            )

            loop_length = loop_length if loop_length > 0 else float("inf")

            data_file_names = [
                os.path.splitext(entry)[0]
                for entry in os.listdir("./data/")
                if os.path.isfile(f"./data/{entry}")
            ]
            movement_class_files = [file for file in data_file_names if file.startswith(movement_class)] or [
                "_-1"
            ]
            movement_class_numbers = [int(file[file.rindex("_") + 1 :]) for file in movement_class_files]
            num_files = max(movement_class_numbers) + 1

            num_success_recordings = 0
            while num_success_recordings < loop_length:
                file_name = f"./data/{movement_class}_{file_suffix}_{num_files + num_success_recordings}.csv"

                with open(file_name, "w+") as f:
                    print("\nIMU will begin recording data in 3 seconds", end="", flush=True)
                    sleep(1)
                    print("\rIMU will begin recording data in 2 seconds", end="", flush=True)
                    sleep(1)
                    print("\rIMU will begin recording data in 1 seconds", end="", flush=True)
                    sleep(1)

                    ser.reset_input_buffer()
                    ser.write([0xFF])  # trigger IMU recording for 2 seconds
                    data = ""

                    line = ser.readline().decode("ascii").strip()
                    print("\rDATA RECORDING IN PROGRESS" + " " * 100, flush=True)

                    for i in range(3):
                        print(i)
                        sleep(1)

                    while True:
                        line = ser.readline().decode("ascii").strip()

                        if line == "$":  # recieved end of transmission
                            ser.reset_input_buffer()
                            break

                        if data:
                            data += line + "\n"
                        elif "acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,time" in line:
                            print(
                                "\rDATA RECORDING FINISHED. SAVING TO FILE." + " " * 100, end="", flush=True
                            )
                            data += line[line.index("acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,time") :] + "\n"

                    f.write(data)
                    print(f"\rDATA SAVED TO '{file_name}'" + " " * 100 + "\n", flush=True)

                analyze = get_input_yes_no("Analyze data now? [Y/N]: ")
                if analyze:
                    plot_data(get_data(file_name), file_name)

                keep_recording = get_input_yes_no("Do you want to keep this recording? [Y/N]: ")
                if keep_recording:
                    num_success_recordings += 1
                else:
                    os.remove(file_name)
    except KeyboardInterrupt:
        if ser:
            print("\rCLOSING PORT..." + " " * 100)
            ser.close()
        exit(0)


def analyze_data():
    def path_is_valid(path):
        if not os.path.exists(path):
            print(f"{path} not found. ")
            return False
        return True

    try:
        while True:
            file_path = get_input("Enter data file path: ", path_is_valid)
            plot_data(get_data(file_path), file_path)
    except KeyboardInterrupt:
        exit(0)


def plot_data(dataframe, filename):
    shift_z = get_input_yes_no(
        "Shift Z-axis down [z_signal will be z_signal = z_signal - avg(z_signal). This helps deal with z-axis being shifted up due to gravity] [Y/N]? "
    )
    show_spectogram = get_input_yes_no("Do you want to plot a spectogram? [Y/N]: ")

    df = dataframe

    if shift_z:
        df["acc_z"] -= sum(df["acc_z"]) / len(df["acc_z"])

    fig = plt.figure()

    fig.suptitle(f"{filename} analysis", fontsize=16)
    fig.subplots_adjust(hspace=0.3)

    acc_plot = fig.add_subplot(221)
    acc_plot.plot(df["time"], df["acc_x"] * 9.8, label="acc_x")
    acc_plot.plot(df["time"], df["acc_y"] * 9.8, label="acc_y")
    acc_plot.plot(df["time"], df["acc_z"] * 9.8, label="acc_z")
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

    fig.show()

    if show_spectogram:
        fig2 = plt.figure()
        fig2.suptitle(f"{filename} spectograms", fontsize=16)
        fig2.subplots_adjust(hspace=0.3)

        a_x = fig2.add_subplot(231)
        a_y = fig2.add_subplot(232)
        a_z = fig2.add_subplot(233)
        g_x = fig2.add_subplot(234)
        g_y = fig2.add_subplot(235)
        g_z = fig2.add_subplot(236)

        fs = 250
        noverlap = 149
        nperseg = 150
        color = "auto"

        a_x_freqs, a_x_times, a_x_magnitudes = spectrogram(
            x=df["acc_x"], fs=fs, noverlap=noverlap, nperseg=nperseg
        )
        fig2.colorbar(a_x.pcolormesh(a_x_times, a_x_freqs, a_x_magnitudes, shading=color))
        a_x.set_title("Accel X")
        a_x.set_ylim(0, 25)

        a_y_freqs, a_y_times, a_y_magnitudes = spectrogram(
            x=df["acc_y"], fs=fs, noverlap=noverlap, nperseg=nperseg
        )
        fig2.colorbar(a_y.pcolormesh(a_y_times, a_y_freqs, a_y_magnitudes, shading=color))
        a_y.set_title("Accel Y")
        a_y.set_ylim(0, 25)

        a_z_freqs, a_z_times, a_z_magnitudes = spectrogram(
            x=df["acc_z"], fs=fs, noverlap=noverlap, nperseg=nperseg
        )
        fig2.colorbar(a_z.pcolormesh(a_z_times, a_z_freqs, a_z_magnitudes, shading=color))
        a_z.set_title("Accel Z")
        a_z.set_ylim(0, 25)

        g_x_freqs, g_x_times, g_x_magnitudes = spectrogram(
            x=df["gyro_x"], fs=fs, noverlap=noverlap, nperseg=nperseg
        )
        fig2.colorbar(g_x.pcolormesh(g_x_times, g_x_freqs, g_x_magnitudes, shading=color))
        g_x.set_title("Gyro X")
        g_x.set_ylim(0, 25)

        g_y_freqs, g_y_times, g_y_magnitudes = spectrogram(
            x=df["gyro_y"], fs=fs, noverlap=noverlap, nperseg=nperseg
        )
        fig2.colorbar(g_y.pcolormesh(g_y_times, g_y_freqs, g_y_magnitudes, shading=color))
        g_y.set_title("Gyro Y")
        g_y.set_ylim(0, 25)

        g_z_freqs, g_z_times, g_z_magnitudes = spectrogram(
            x=df["gyro_z"], fs=fs, noverlap=noverlap, nperseg=nperseg
        )
        fig2.colorbar(g_z.pcolormesh(g_z_times, g_z_freqs, g_z_magnitudes, shading=color))
        g_z.set_title("Gyro Z")
        g_z.set_ylim(0, 25)

        fig2.show()

    plt.show()
    return df


if __name__ == "__main__":
    try:
        while True:

            def validator(inpt):
                try:
                    inpt = int(inpt)
                except ValueError:
                    print("\rInvalid Option. Record new data (1) or analyze pre-existing data (2)? ")
                    return False
                if inpt == 1 or inpt == 2:
                    return True

            analyze = int(get_input("Record new data [1] or analyze existing data [2]? ", validator))
            if analyze == 1:
                record_data()
            elif analyze == 2:
                analyze_data()
    except KeyboardInterrupt:
        exit(0)
