import numpy as np
from utils import processing

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
