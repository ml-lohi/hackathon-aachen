from scipy import signal
from utils.helper import filter_gauss
from utils.helper import fft
import numpy as np
import matplotlib.pyplot as plt


def calculate_br_with_fft(data, kernel_factor=20, sigma=25):
    smoothed_data = filter_gauss(data, kernel_factor=kernel_factor, sigma=sigma)
    bpm = signal.butter(1, [0.1, 1], "bandpass", fs=1000, output="sos")
    filtered = signal.sosfilt(bpm, smoothed_data)
    fft_x_fitlered, fft_y_filtered = fft(filtered)
    bpm_frequency = fft_x_fitlered[np.argmax(fft_y_filtered)]
    return bpm_frequency, filtered, fft_x_fitlered, fft_y_filtered


def calculate_rates_with_peaks(
    data,
    kernel_factor=3,
    sigma=100,
    bandpass_range=[0.1, 0.6],
    n=10,
    fs=1000,
    to_plot=False,
):
    bandpass = signal.butter(n, bandpass_range, "bandpass", fs=fs, output="sos")
    filtered = signal.sosfilt(bandpass, data)
    filtered_gauss = filter_gauss(filtered, kernel_factor=kernel_factor, sigma=sigma)
    x_filtered = np.arange(len(filtered_gauss))
    frequencies, fft_wave = fft(filtered_gauss, samples_per_second=fs)
    peaks, _ = signal.find_peaks(filtered_gauss, distance=fs / bandpass_range[1])
    if to_plot:
        fig, ax = plt.subplots(3, 1, sharex=False, figsize=(8, 8))
        ax[0].plot(filtered)
        ax[1].plot(filtered_gauss)
        ax[1].plot(x_filtered[peaks], filtered_gauss[peaks], "x")
        ax[2].plot(frequencies, fft_wave)
        ax[2].set_xlim(0, 4)

    periods = np.diff(x_filtered[peaks])
    period_mean, period_err = np.mean(periods), np.std(periods)
    frequency = fs / period_mean * 60
    frequency_err = 1 / period_mean**2 * period_err * fs / 60
    return frequency, frequency_err
