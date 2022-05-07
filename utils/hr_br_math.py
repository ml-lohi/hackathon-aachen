from scipy import signal
from utils.helper import filter_gauss
from utils.helper import fft
import numpy as np


def calculate_br_with_fft(data, kernel_factor=20, sigma=25):
    smoothed_data = filter_gauss(data, kernel_factor=kernel_factor, sigma=sigma)
    bpm = signal.butter(1, [0.1, 1], 'bandpass', fs=1000, output='sos')
    filtered = signal.sosfilt(bpm, smoothed_data)
    fft_x_fitlered, fft_y_filtered = fft(filtered)
    bpm_frequency = fft_x_fitlered[np.argmax(fft_y_filtered)]
    return bpm_frequency, filtered, fft_x_fitlered, fft_y_filtered