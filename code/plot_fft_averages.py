import h5py
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def find_segment_border(py, threshold, i_iter=0):
    # determine segement border with sliding time window analysis
    seg_idx = 0
    windowsize = int(0.1 * N_episode)  # fixed window size
    for idx in range(N_episode - windowsize):
        py_mean = np.mean(py[idx:idx + windowsize])
        if py_mean > threshold:
            seg_idx = idx
            break
    if seg_idx == 0:
        if i_iter >= 2:
            print("No segmentation border can be found within the depth of 2 iterations! Taking whole signal.")
            seg_idx = 0
            return seg_idx
        else:
            new_threshold = list(str(threshold))
            for i in range(i_iter+1):
                new_threshold[-i-1] = '0'
            new_threshold = float("".join(new_threshold))
            print("No segmentation border found! Threshold too high. Recalculating with lowered threshold {0}".format(new_threshold))
            seg_idx = find_segment_border(py, threshold=new_threshold, i_iter=i_iter+1)

    return seg_idx


if __name__ == "__main__":

    # mapping
    idx_map = {"cx": 0, "cvx": 1, "px": 2, "py": 3, "pav": 4}

    # read data
    with h5py.File("../data/preprocessed/CartPoleData_full.h5", 'r') as f:
        cp_data = f["CartPoleData_full"][...]
        rewards = f["EpisodeRewards"][...]

    N_episode = 252  # number of data points per episode
    N_total = len(cp_data)  # number of total data points
    episode_idxs = np.squeeze(np.argwhere(cp_data[:, idx_map['cx']] == 0.0)) # indexes for episode starts
    # cut the last and the first 8 episodes so that 1900 episodes remain
    episode_idxs = episode_idxs[8:-1]
    num_episodes = len(episode_idxs)
    rewards = rewards[8:]

    # order episodes according to max reward
    episode_order = np.argsort(rewards)

    # create data objects that contain the data for each single episode
    cx_episodes = np.zeros((num_episodes, N_episode))
    px_episodes = np.zeros((num_episodes, N_episode))
    py_episodes = np.zeros((num_episodes, N_episode))
    phi_episodes = np.zeros((num_episodes, N_episode))
    for i_episode in range(num_episodes):
        cx_episodes[i_episode] = cp_data[episode_idxs[i_episode]:episode_idxs[i_episode] + N_episode, idx_map['cx']]
        px_episodes[i_episode] = cp_data[episode_idxs[i_episode]:episode_idxs[i_episode] + N_episode, idx_map['px']]
        py_episodes[i_episode] = cp_data[episode_idxs[i_episode]:episode_idxs[i_episode] + N_episode, idx_map['py']]
        phi_episodes[i_episode] = np.arctan2(py_episodes[i_episode], px_episodes[i_episode]) * 180 / np.pi

    # create data object for FFT data
    T = 16.65
    dt = T / N_episode
    fs = 1 / dt
    t = np.linspace(0, T, N_episode)

    seg = 3.0
    s12 = int(N_episode * seg / T)  # seconds

    N_fft = len(cx_episodes[0][s12:])
    x_cx_fft = np.linspace(0, 1 / (2 * dt), N_fft // 2)

    cx_fft_episodes = np.zeros((num_episodes, N_fft // 2 - 1))
    for i_episode in range(num_episodes):
        # perform FFT on filtered movement data to detect high frequency component
        cx_filt = butter_bandpass_filter(cx_episodes[i_episode], lowcut=2.0, highcut=5.0, fs=fs, order=5)
        cx_filt_fft = fft(cx_filt[s12:])
        cx_fft_episodes[i_episode] = 2 / N_fft * np.abs(cx_filt_fft[1:N_fft // 2])

    # for i_episode in episode_order[::-1]:
    #     fig = plt.figure(figsize=(12, 6))
    #     fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.6)
    #     ax = fig.add_subplot(111)
    #     ax.plot(x_cx_fft[1:], cx_fft_episodes[i_episode], lw=2, c='k')
    #     ax.set_title('episode_ID: {0}, maximum reward: {1}'.format(i_episode + 1, rewards[i_episode]))
    #     plt.show()

    # reorder
    cx_fft_episodes = cx_fft_episodes[episode_order]

    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.6)
    for i in range(19):
        avr_fft = np.mean(cx_fft_episodes[i*100:(i+1)*100], axis=0)
        ax = fig.add_subplot(4,5,i+1)
        ax.plot(x_cx_fft[1:], avr_fft, lw=2, c='k')
        ax.set_title('eIDs: {0}-{1}, av_rev {2:.2f}'.format(i*100, (i+1)*100, np.mean(rewards[episode_order][i*100:(i+1)*100])))
        ax.axvline(x=3.4, c='k', ls='--')
    plt.show()

    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.6)
    for i in range(20):
        avr_fft = np.mean(cx_fft_episodes[-400:][i*20:(i+1)*20], axis=0)
        ax = fig.add_subplot(4,5,i+1)
        ax.plot(x_cx_fft[1:], avr_fft, lw=2, c='k')
        ax.set_title('eIDs: {0}-{1}, av_rev {2:.2f}'.format(1500+i*20, 1500+(i+1)*20, np.mean(rewards[episode_order][-400:][i*20:(i+1)*20])))
        ax.axvline(x=3.4, c='k', ls='--')
    plt.show()

    # Interpretation and Story
