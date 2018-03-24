import h5py
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # mapping
    idx_map = {"cx": 0, "cvx": 1, "px": 2, "py": 3, "pav": 4}

    # read data
    with h5py.File("../data/preprocessed/CartPoleData_full.h5", 'r') as f:
        cp_data = f["CartPoleData_full"][...]
        rewards = f["EpisodeRewards"][...]


        # ToDo: high-pass, so that low frequs are cut - then boxplot, 1900 episodes, 19 boxplots, 100 episodes per box
        # ToDo: FFT after initial swing up, plot development of amp and freq depending on episode and on reward.
        # ToDo: Show that Agent learns to find the correct mps and freqs as learning progresses