import numpy as np
from utils import processing
from scipy import ndimage
from scipy.fftpack import fft

def read_data(file_name):
    """
    Reads the data from the file csv and returns the data as a numpy array
    """
    loaded_data = np.loadtxt(file_name, delimiter=",")
    data_original = loaded_data.reshape(
        loaded_data.shape[0], loaded_data.shape[1] // 64, 64
    )

    return data_original

def calculate_phases(data, N, sample_size):
    phases_full = np.array([])
    for i in range(N):
        phases, abses, _, _ = processing.do_processing(data[i]) 
        phases_full = np.append(phases_full, np.mean(phases, axis=0))
    phases_full = phases_full.reshape((int(phases_full.shape[0] / sample_size),sample_size))
    return phases_full

def gauss(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def filter_gauss(image, kernel_factor, sigma):
    faktor = kernel_factor * sigma * 2 + 1 
    line = np.linspace(-kernel_factor*sigma, kernel_factor*sigma, faktor)
    filter = gauss(line, sigma=sigma)
    data_smoothed = ndimage.convolve(image, filter, mode='wrap')
    return data_smoothed

def fft(data, samples_per_second=1000):
    """
    Args:
        data: numpy array with the data
        samples_per_second: samples per second
    Returns:
        freqs, fft_wave.real: x and y to plot
    """
    fft_wave = np.fft.fft(data)
    freqs = np.fft.fftfreq(n=data.size, d = 1/samples_per_second) 
    return freqs, fft_wave.real
    