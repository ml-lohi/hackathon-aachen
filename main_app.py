#!/usr/bin/env python
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
from pyparsing import col
import utils.processing as processing
from utils.hr_br_math import (
    calculate_br_with_fft,
    calculate_rates_with_peaks,
    calculate_hr_with_fft,
)
from utils.helper import filter_gauss
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
from tensorflow import keras

CONFIG_FILE = (
    "F:\Projects\hackathon\hackathon-private\\radar_configs\RadarIfxBGT60.json"
)
SEC = 1
THRESHOLD = 10
MIN_RANGE = 0.6
MAX_RANGE = 4


class Application(tk.Frame):
    def __init__(self, root=None):
        tk.Frame.__init__(self, root)
        self.model = keras.models.load_model("hackathon-private\models\CNN_Sviat")
        self.root = root
        self.xdata, self.ydata, self.ydata_static, self.hrs, self.brs = (
            [],
            [],
            [],
            [],
            [],
        )
        self.br, self.hr, self.br_fft = np.nan, np.nan, np.nan
        self.moving_counter, self.static_counter = 0, 0
        self.moving = False
        self.predicted = [[0, 1]]
        (
            self.strVarPredicted,
            self.strVarBr,
            self.strVarHr,
            self.strVarBrFFT,
            self.strVarMoving,
        ) = (
            tk.StringVar(),
            tk.StringVar(),
            tk.StringVar(),
            tk.StringVar(),
            tk.StringVar(),
        )
        self._reset_vars()
        self.createWidgets()

    def _reset_vars(self):
        """Reset the variables"""
        self.strVarPredicted.set(
            "CNN Moving"
            if self.predicted[0][1] > self.predicted[0][0]
            else "CNN Static"
        )
        self.strVarBr.set(f"BR rate: {self.br:.2f} bpm")
        self.strVarHr.set(f"HR rate: {self.hr:.2f} bpm")
        self.strVarBrFFT.set(f"BR rate FFT: {self.br_fft:.2f} bpm")
        self.strVarMoving.set("Moving" if self.moving else "Static")
        self.root.update()

    def createWidgets(self):
        """Create labels and plot on the canvas"""
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.get_tk_widget().grid(
            row=0, column=4, rowspan=10, columnspan=10, sticky="nsew"
        )
        canvas.draw()
        self.thread = threading.Thread(target=self._process, args=(canvas, axs))
        self.thread.setDaemon(True)
        self.thread.start()
        self.plotbutton = tk.Button(
            master=self.root, text="Quit", command=self.root.quit
        )
        self.plotbutton.grid(row=0, column=0)
        my_font = tk.font.Font(self.root, family="Anurati Regular")
        lbl = tk.Label(root, textvariable=self.strVarMoving)
        lbl.grid(row=0, column=2, columnspan=2, sticky="nsew")
        lbl = tk.Label(root, textvariable=self.strVarPredicted)
        lbl.grid(row=1, column=2, columnspan=2, sticky="nsew")
        lbl = tk.Label(root, textvariable=self.strVarHr)
        lbl.grid(row=2, column=2, columnspan=2, sticky="nsew")
        lbl = tk.Label(root, textvariable=self.strVarBr)
        lbl.grid(row=3, column=2, columnspan=2, sticky="nsew")
        lbl = tk.Label(root, textvariable=self.strVarBrFFT)
        lbl.grid(row=4, column=2, columnspan=2, sticky="W")

    def _process(self, canvas, ax):
        """Process the infinite data flow"""
        raw_data = []
        with RadarIfxAvian(
            CONFIG_FILE
        ) as device:  # Initialize the radar with configurations
            static_counter, moving_counter = 0, 0  # Initialize the counters

            for i_frame, frame in enumerate(
                device
            ):  # Loop through the frames coming from the radar

                raw_data.append(
                    np.squeeze(frame["radar"].data / (4095.0))
                )  # Dividing by 4095.0 to scale the data

                if (
                    len(raw_data) > SEC * 1000 - 1 and len(raw_data) % SEC * 1000 == 0
                ):  # 5000 is the number of frames. which corresponds to 5seconds

                    data = np.swapaxes(np.asarray(raw_data), 0, 1)
                    phases, _, _, _ = processing.do_processing(
                        data
                    )  # preprocessing to get the phase information
                    phases = np.mean(phases, axis=0).reshape(-1)
                    moving_value = np.mean(
                        np.abs(phases)
                    )  # Determine the moving value (later used to determine the state)
                    self.predicted = self.model.predict(
                        phases.reshape(1, -1)
                    )  # Predict the state using CNN

                    self._process_step(moving_value, phases)

                    new_x = np.linspace(
                        len(self.xdata),
                        len(self.xdata) + len(phases),
                        len(phases),  # x values to append
                    )
                    self.xdata.extend(new_x / 1000)
                    self.ydata.extend(phases)

                    self._plot(canvas, ax)  # plot the data
                    raw_data = []
                    self._reset_vars()

    def _process_step(self, moving_value, phases):
        """Process the data each second"""
        if moving_value > THRESHOLD:
            self.moving = True
            self.moving_counter += 1
            if self.moving_counter == 3:
                (
                    self.moving_counter,
                    self.static_counter,
                    self.hr,
                    self.br,
                    self.br_fft,
                ) = (
                    0,
                    0,
                    np.nan,
                    np.nan,
                    np.nan,
                )  # Reset the counters
                self.ydata_static, self.brs, self.hrs = ([], [], [])  # Reset the data

        else:
            self.moving = False
            self.ydata_static.extend(
                filter_gauss(phases, kernel_factor=3, sigma=20)
            )  # Save the data for HR and BR determintation (only when static)
            if (
                self.static_counter % 5 == 0 and self.static_counter != 0
            ):  # every 5 seconds
                self.hr, _ = calculate_rates_with_peaks(
                    self.ydata_static[-5000:],
                    sigma=15,
                    bandpass_range=[1, 3],
                    fs=1000,
                    n=10,
                )
                if len(self.hrs) > 0:
                    self.hr = np.average(
                        [self.hr, self.hrs[-1]],
                        weights=[0.6, 0.4],  # average the HR with the previous one
                    )
                self.hrs.append(self.hr)

            if (self.static_counter % 15 == 0 and self.static_counter != 0) or (
                self.static_counter > 15 and self.static_counter % 5 == 0
            ):  # firstly every 15 seconds and than every 5 seconds

                self.br_fft, _, _, _ = calculate_br_with_fft(
                    self.ydata_static[-15000:], kernel_factor=3, thr=11000
                )
                self.br, _ = calculate_rates_with_peaks(
                    np.abs(self.ydata_static[-15000:]),
                    sigma=200,
                    bandpass_range=[0.1, 1],
                    fs=1000,
                    amplitude_threshold=3,
                    min_peaks=3,
                )
                if len(self.brs) > 0 and self.br_fft != 0:
                    self.br = np.average(
                        [self.br_fft, self.brs[-1]],
                        weights=[0.7, 0.3],  # average the BR with the previous one
                    )
                self.brs.append(self.br_fft)

            self.static_counter += 1
            self.moving_counter = 0

    def _plot(self, canvas, axs):
        """Plot the data on the canvas"""
        for i, ax in enumerate(axs):
            ax.clear()  # clear axes from previous plot
            ax.grid()

        axs[0].set_title(label=("Moving" if self.moving else "Static"))
        axs[0].plot(
            self.xdata, self.ydata, label=("Moving" if self.moving else "Static")
        )
        state = ""
        if self.hr is np.nan:
            state = "Waiting for data"
        elif self.hr == 0:
            state = "No Puls"
        elif self.hr < 120:
            state = "Normal Heart Rate"
        else:
            state = "Heart Rate after Sport or because of Stress"

        axs[1].set_title(f"HR: {self.hr:.2f} || {state}")
        axs[1].plot(np.arange(len(self.hrs)), self.hrs)

        state = ""
        if self.br_fft is np.nan:
            state = "Waiting for data"
        elif self.br_fft == 0:
            state = "No Breathing"
        elif self.br_fft < 30:
            state = "Normal Breathing"
        else:
            state = "Breathing after Sport or because of Stress"

        axs[2].set_title(f"BR: {self.br_fft:.2f}|| {state}")
        axs[2].plot(np.arange(len(self.brs)), self.brs)
        if self.brs:
            axs[2].set_xlim(0, len(self.brs) + 5)
            axs[2].set_ylim(np.min(self.brs) - 5, np.max(self.brs) + 5)

        canvas.draw()


if "__main__" == __name__:
    root = tk.Tk()
    root.title("Amazing Radar App")
    root.geometry("750x650")
    app = Application(root=root)
    app.mainloop()
