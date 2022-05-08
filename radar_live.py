import ifxdaq
import utils.processing as processing
from utils.hr_br_math import calculate_br_with_fft, calculate_rates_with_peaks
from utils.helper import filter_gauss
import numpy as np
import matplotlib.pyplot as plt

# print(ifxdaq.__version__)
from ifxdaq.sensor.radar_ifx import RadarIfxAvian
import pandas as pd

plt.ion()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def gaussian(x, mu, sig):
    return 1.0 / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


CONFIG_FILE = (
    "F:\Projects\hackathon\hackathon-private\\radar_configs\RadarIfxBGT60.json"
)
SEC = 1
THRESHOLD = 20


class DynamicUpdate:
    # Suppose we know the x range
    min_x = 0
    max_x = SEC * 30 * 1000

    def on_launch(self):
        # Set up plot
        self.figure, self.ax = plt.subplots(3, 1)
        (self.lines0,) = self.ax[0].plot([], [], "b-")
        (self.lines1,) = self.ax[1].plot([], [], "r-")
        (self.lines21,) = self.ax[2].plot([], [], "b-")
        (self.lines22,) = self.ax[2].plot([], [], "r-")

        # Autoscale on unknown axis and known lims on the other
        for i in range(3):
            self.ax[i].set_autoscaley_on(True)
            self.ax[i].grid()

        # self.ax[0].set_autoscaley_on(True)
        self.ax[0].set_xlim(self.min_x, self.max_x / 1000)
        self.ax[1].set_xlim(0, 30)
        # self.ax[1].set_ylim(30, 200)
        self.ax[2].set_xlim(0, 15)
        # self.ax[2].set_ylim(10, 50)
        # Other stuff
        # self.ax[0].grid()
        ...

    def on_running(self, xdata, ydata, hrs, brs, brs_fft):
        # Update data (with the new _and_ the old points)
        self.lines0.set_xdata(xdata)
        self.lines0.set_ydata(ydata)
        self.lines1.set_ydata(hrs)
        self.lines1.set_xdata(np.arange(len(hrs)))
        self.lines21.set_data(np.arange(len(brs)), brs)
        self.lines22.set_data(np.arange(len(brs_fft)), brs_fft * 60)
        # self.ax1.lines.set_ydata(hrs)
        # self.ax2.lines.set_ydata(brs)
        # Need both of these in order to rescale
        for i in range(3):
            self.ax[i].relim()
            self.ax[i].autoscale_view()

        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # Example
    def __call__(self):
        import numpy as np
        import time

        self.on_launch()
        xdata, ydata, ydata_static, raw_data = [], [], [], []
        hrs, hr_errs, brs, br_errs, brs_fft = [], [], [], [], []
        with RadarIfxAvian(
            CONFIG_FILE
        ) as device:  # Initialize the radar with configurations
            hr_counter, br_counter, static_counter, moving_counter = 0, 0, 0, 0
            hr, br, br_fft = np.nan, np.nan, np.nan

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
                    phases, abses, _, _ = processing.do_processing(
                        data
                    )  # preprocessing to get the phase information
                    phases = np.mean(phases, axis=0).reshape(-1)
                    moving_value = np.mean(np.abs(phases))

                    if moving_value > THRESHOLD:
                        self.ax[0].set_title(f"Moving {moving_value:.2f}")
                        moving_counter += 1
                        if moving_counter == 3:
                            moving_counter, static_counter, hr, br, br_fft = (
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
                        self.ax[0].set_title(
                            f"Static {moving_value:.2f} HR = {hr:.2f}, BR = {br:.2f}, BR_FFT = {(br_fft*60):.2f}"
                        )
                        ydata_static.extend(
                            filter_gauss(phases, kernel_factor=3, sigma=20)
                        )
                        if static_counter % 5 == 0 and static_counter != 0:
                            hr, hr_err = calculate_rates_with_peaks(
                                ydata_static[-5000:],
                                sigma=100,
                                bandpass_range=[0.6, 4],
                                fs=1000,
                            )
                            print(f"HR Before = {hr:.2f}")
                            if len(hrs) > 0:
                                hr = np.average([hr, hrs[-1]], weights=[0.7, 0.3])
                            print(f"HR After = {hr:.2f}")
                            # print(f"HR: {hr}")
                            hrs.append(hr)
                            hr_errs.append(hr_err)
                        if static_counter % 10 == 0 and static_counter != 0:
                            print(f"in BR state {static_counter}")
                            br_fft, _, _, _ = calculate_br_with_fft(
                                ydata_static[-10000:], kernel_factor=3
                            )
                            br, br_err = calculate_rates_with_peaks(
                                ydata_static[-10000:],
                                sigma=500,
                                bandpass_range=[0.1, 0.6],
                                fs=1000,
                            )
                            if len(brs) > 0:
                                print("here br = ", br, brs[-1])
                                br = np.average([br, brs[-1]], weights=[0.7, 0.3])
                                print("br", br)

                            print(f"BR: {br}, BR_FFT: {br_fft*60}")
                            brs.append(br)
                            br_errs.append(br_err)
                            brs_fft.append(br_fft)
                        static_counter += 1
                        moving_counter = 0

                    new_x = np.linspace(
                        len(xdata), len(xdata) + len(phases), len(phases)
                    )
                    xdata.extend(new_x / 1000)
                    ydata.extend(phases)

                    self.on_running(
                        xdata,
                        ydata,
                        hrs,
                        brs,
                        np.asarray(brs_fft),
                    )
                    raw_data = []
                    # print(f"state_counter: {state_counter}")
                    if xdata[-1] * 1000 > self.max_x:
                        self.figure.canvas.flush_events()
                        xdata = []
                        ydata = []

        return xdata, ydata


d = DynamicUpdate()
d()
