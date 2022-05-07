import ifxdaq
import utils.processing as processing
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
        self.figure, self.ax = plt.subplots()
        (self.lines,) = self.ax.plot([], [], "r-")
        # Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x / 1000)
        # Other stuff
        self.ax.grid()
        ...

    def on_running(self, xdata, ydata):
        # Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    # Example
    def __call__(self):
        import numpy as np
        import time

        self.on_launch()
        xdata = []
        ydata = []
        raw_data = []
        with RadarIfxAvian(
            CONFIG_FILE
        ) as device:  # Initialize the radar with configurations
            idx = 0
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
                        self.ax.set_title(f"Moving {moving_value}")
                    else:
                        self.ax.set_title(f"Static {moving_value}")
                    new_x = np.linspace(
                        len(xdata), len(xdata) + len(phases), len(phases)
                    )
                    xdata.extend(new_x / 1000)
                    ydata.extend(phases)

                    self.on_running(xdata, ydata)
                    raw_data = []
                    if xdata[-1] * 1000 > self.max_x:
                        self.figure.canvas.flush_events()
                        xdata = []
                        ydata = []

        return xdata, ydata


d = DynamicUpdate()
d()
