#!/usr/bin/env python
from cProfile import label
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
import time
from utils.helper import fft
import ifxdaq
import utils.processing as processing
from utils.hr_br_math import calculate_br_with_fft, calculate_rates_with_peaks
from utils.helper import filter_gauss
import numpy as np
import matplotlib.pyplot as plt
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
from tensorflow import keras

CONFIG_FILE = (
    "F:\Projects\hackathon\hackathon-private\\radar_configs\RadarIfxBGT60.json"
)
SEC = 1
THRESHOLD = 10
MIN_RANGE = 0.6
MAX_RANGE = 4
MODEL = keras.models.load_model("hackathon-private\models\CNN_Sviat")
print(MODEL)


class Application(tk.Frame):
    def __init__(self, root=None):
        tk.Frame.__init__(self, root)
        self.model = keras.models.load_model("hackathon-private\models\CNN_Sviat")
        self.root = root
        self.xdata = []
        self.ydata = []
        self.br = np.nan
        self.hr = np.nan
        self.br_fft = np.nan
        self.moving = False
        self.predicted = [[0, 1]]
        self.strVarPredicted = tk.StringVar()
        self.strVarBr = tk.StringVar()
        self.strVarHr = tk.StringVar()
        self.strVarBrFFT = tk.StringVar()
        self.strVarMoving = tk.StringVar()
        self._reset_hr_br()
        self.createWidgets()

    def _reset_hr_br(self):
        self.strVarPredicted.set(
            "CNN Moving"
            if self.predicted[0][1] > self.predicted[0][0]
            else "CNN Static"
        )
        self.strVarBr.set(f"BR rate: {self.br:.2f} bpm")
        self.strVarHr.set(f"HR rate: {self.hr:.2f} bpm")
        self.strVarBrFFT.set(f"BR rate FFT: {self.br_fft:.2f} bpm")
        if self.moving:
            self.strVarMoving.set(f"Moving")
        else:
            self.strVarMoving.set(f"Static")
        self.root.update()

    def createWidgets(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.get_tk_widget().grid(row=0, column=1)
        canvas.draw()
        self.thread = threading.Thread(target=self._process, args=(canvas, axs))
        self.thread.setDaemon(True)
        self.thread.start()

        self.plotbutton = tk.Button(
            master=self.root, text="Quit", command=self.root.quit
        )
        self.plotbutton.grid(row=0, column=0)

        lbl = tk.Label(root, textvariable=self.strVarMoving)
        lbl.grid()
        lbl = tk.Label(root, textvariable=self.strVarPredicted)
        lbl.grid()

        lbl = tk.Label(root, textvariable=self.strVarHr)
        lbl.grid()

        lbl = tk.Label(root, textvariable=self.strVarBr)
        lbl.grid()

        lbl = tk.Label(root, textvariable=self.strVarBrFFT)
        lbl.grid()

    def _process(self, canvas, ax):
        x = 0
        xdata, ydata, ydata_static, raw_data = [], [], [], []
        hrs, hr_errs, brs, br_errs, brs_fft = [], [], [], [], []
        hr_min_range, hr_max_range = MIN_RANGE, MAX_RANGE
        with RadarIfxAvian(
            CONFIG_FILE
        ) as device:  # Initialize the radar with configurations
            static_counter, moving_counter = 0, 0

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
                    moving_value = np.mean(np.abs(phases))
                    self.predicted = self.model.predict(phases.reshape(1, -1))
                    # print("PREDICTED: ", self.predicted)
                    if moving_value > THRESHOLD:
                        # self.ax[0].set_title(f"Moving {moving_value:.2f}")
                        self.moving = True
                        moving_counter += 1
                        if moving_counter == 3:
                            moving_counter, static_counter, self.hr, self.br, br_fft = (
                                0,
                                0,
                                np.nan,
                                np.nan,
                                np.nan,
                            )
                            ydata_static = []
                            brs = []
                            hrs = []
                            brs_fft = []

                    else:
                        self.moving = False
                        ydata_static.extend(
                            filter_gauss(phases, kernel_factor=3, sigma=20)
                        )
                        if static_counter % 5 == 0 and static_counter != 0:
                            self.hr, hr_err = calculate_rates_with_peaks(
                                ydata_static[-5000:],
                                sigma=100,
                                bandpass_range=[hr_min_range, hr_max_range],
                                fs=1000,
                                n=10,
                            )
                            if len(hrs) > 0:
                                self.hr = np.average(
                                    [self.hr, hrs[-1]], weights=[0.6, 0.4]
                                )
                            hrs.append(self.hr)
                            hr_min_range = 60 / (self.hr + 50)
                            hr_max_range = 60 / (self.hr - 50)
                            if hr_min_range < MIN_RANGE:
                                hr_min_range = MIN_RANGE
                            if hr_max_range > MAX_RANGE:
                                hr_max_range = MAX_RANGE
                            print(f"NEW RANGE: {hr_min_range} - {hr_max_range}")
                        if static_counter % 10 == 0 and static_counter != 0:
                            # print(f"in BR state {static_counter}")
                            self.br_fft, _, _, _ = calculate_br_with_fft(
                                ydata_static[-10000:], kernel_factor=3
                            )
                            self.br, br_err = calculate_rates_with_peaks(
                                ydata_static[-10000:],
                                sigma=500,
                                bandpass_range=[0.1, 0.6],
                                fs=1000,
                            )
                            if len(brs) > 0:
                                # print("here br = ", br, brs[-1])
                                self.br = np.average(
                                    [self.br, brs[-1]], weights=[0.6, 0.4]
                                )
                                # print("br", br)

                            # print(f"BR: {br}, BR_FFT: {br_fft*60}")
                            brs.append(self.br)
                            # br_errs.append(br_err)
                            brs_fft.append(self.br_fft)
                        static_counter += 1
                        moving_counter = 0

                    new_x = np.linspace(
                        len(self.xdata), len(self.xdata) + len(phases), len(phases)
                    )
                    self.xdata.extend(new_x / 1000)
                    self.ydata.extend(phases)

                    self._plot(canvas, ax, hrs, brs, brs_fft)
                    raw_data = []
                    self._reset_hr_br()
                    # # print(f"state_counter: {state_counter}")
                    # if self.xdata[-1] * 1000 > self.max_x:
                    #     self.figure.canvas.flush_events()
                    #     self.xdata = []
                    #     self.ydata = []

    def _plot(self, canvas, axs, hrs, brs, brs_fft):
        for i, ax in enumerate(axs):
            ax.clear()
            ax.grid()  # clear axes from previous plot
        axs[0].set_title(label=("Moving" if self.moving else "Static"))
        axs[0].plot(
            self.xdata, self.ydata, label=("Moving" if self.moving else "Static")
        )
        axs[1].set_title(f"HR: {self.hr:.2f}")
        axs[1].plot(np.arange(len(hrs)), hrs, label=f"HR = {self.hr}")
        axs[2].set_title(f"BR: {self.br:.2f}|| BR_FFT: {self.br_fft:.2f}")
        # axs[2].plot(np.arange(len(brs_fft)), brs_fft, label=f"BR_FFT = {self.br_fft}")
        axs[2].plot(np.arange(len(brs)), brs, label=f"BR = {self.br}")
        # for ax in axs:
        #     ax.legend()

        canvas.draw()


if "__main__" == __name__:
    root = tk.Tk()
    root.title("Amazing Radar App")
    root.geometry("750x650")
    app = Application(root=root)
    app.mainloop()
