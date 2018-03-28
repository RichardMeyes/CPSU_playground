import h5py
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
        cx_filt = butter_bandpass_filter(cx_episodes[i_episode], lowcut=2.0, highcut=7.5, fs=fs, order=5)
        cx_filt_fft = fft(cx_filt[s12:])
        cx_fft_episodes[i_episode] = 2 / N_fft * np.abs(cx_filt_fft[1:N_fft // 2])

    # for i_episode in episode_order[::-1]:
    #     fig = plt.figure(figsize=(12, 6))
    #     fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.6)
    #     ax = fig.add_subplot(111)
    #     ax.plot(x_cx_fft[1:], cx_fft_episodes[i_episode], lw=2, c='k')
    #     ax.set_title('episode_ID: {0}, maximum reward: {1}'.format(i_episode + 1, rewards[i_episode]))
    #     plt.show()

    # calculate indexes for frequency band around 3.4 Hz
    freq1 = 3.4  # Hz
    margin = 0.1  # Hz
    low_freq1 = freq1 - margin
    high_freq1 = freq1 + margin
    idx_low1 = int(len(x_cx_fft)*low_freq1/x_cx_fft[-1]) - 1  # -1, becasue the first data point of x_cx_fft is cut for plotting!
    idx_high1 = int(len(x_cx_fft) * high_freq1 / x_cx_fft[-1]) - 1  # -1, becasue the first data point of x_cx_fft is cut for plotting!
    peak_idx1 = int(len(x_cx_fft)*(low_freq1+high_freq1)/2/x_cx_fft[-1]) - 1

    # calculate indexes for frequency band around 2.3 Hz
    freq2 = 2.3  # Hz
    margin = 0.2  # Hz
    low_freq2 = freq2 - margin
    high_freq2 = freq2 + margin
    idx_low2 = int(len(x_cx_fft) * low_freq2 / x_cx_fft[-1]) - 1  # -1, becasue the first data point of x_cx_fft is cut for plotting!
    idx_high2 = int(len(x_cx_fft) * high_freq2 / x_cx_fft[-1]) - 1  # -1, becasue the first data point of x_cx_fft is cut for plotting!
    peak_idx2 = int(len(x_cx_fft) * (low_freq2 + high_freq2) / 2 / x_cx_fft[-1]) - 1


    # reorder
    cx_fft_episodes = cx_fft_episodes[episode_order]

    # store avr_fft amps for 3.4 Hz peak
    amps_high = []
    amps_low = []

    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.6)
    for i in range(19):
        avr_fft = np.mean(cx_fft_episodes[i*100:(i+1)*100], axis=0)
        avr_fft_std = np.std(cx_fft_episodes[i * 100:(i + 1) * 100], axis=0)
        ax = fig.add_subplot(4,5,i+1)
        ax.plot(x_cx_fft[1:], avr_fft, lw=2, c='k')
        ax.fill_between(x_cx_fft[1:], avr_fft - avr_fft_std, avr_fft + avr_fft_std, color='k', alpha=0.3)
        ax.set_title('eIDs: {0}-{1}, av_rev {2:.2f}'.format(i*100, (i+1)*100, np.mean(rewards[episode_order][i*100:(i+1)*100])))
        ax.axvline(x=freq1, ymin=0, ymax=avr_fft[peak_idx1]/0.006, c='b', ls='--')
        ax.axvline(x=freq2, ymin=0, ymax=avr_fft[peak_idx2]/0.006, c='orange', ls='--')
        ax.axhline(y=avr_fft[peak_idx1], xmin=0, xmax=freq1/x_cx_fft[-1], c='b', ls='--')
        ax.axhline(y=avr_fft[peak_idx2], xmin=0, xmax=freq2 / x_cx_fft[-1], c='orange', ls='--')
        ax.set_ylim(0, 0.006)
        amps_high.append(np.mean(avr_fft[idx_low1:idx_high1+1]))
        amps_low.append(np.mean(avr_fft[idx_low2:idx_high2 + 1]))

    def func_lin(x, a, b):
        return a*x + b
    popt1, pcov1 = curve_fit(func_lin, np.arange(len(amps_high))[3:], amps_high[3:])
    popt2, pcov2 = curve_fit(func_lin, np.arange(len(amps_low))[3:-4], amps_low[3:-4])
    popt3, pcov3 = curve_fit(func_lin, np.arange(len(amps_low))[-6:], amps_low[-6:])

    # ToDo: chose more windows with same window size but smaller window step size so that after each step 50% of the previous window is kept
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.14, right=0.95, wspace=0.6)
    ax = fig.add_subplot(111)
    ax.plot(amps_high, c='b', marker='o', lw=2, alpha=0.5)
    ax.plot(np.arange(len(amps_high))[3:], func_lin(np.arange(len(amps_high))[3:], *popt1), lw=2, ls='--', c='b')
    ax.plot(amps_low, c='orange', marker='o', lw=2, alpha=0.5)
    ax.plot(np.arange(len(amps_low))[3:-4], func_lin(np.arange(len(amps_low))[3:-4], *popt2), lw=2, ls='--', c='orange')
    ax.plot(np.arange(len(amps_low))[-6:], func_lin(np.arange(len(amps_low))[-6:], *popt3), lw=2, ls='--', c='orange')
    ax.set_xlabel("episode window")
    ax.set_ylabel("amplitude")
    ax.set_xticks(np.arange(19))
    ax.set_xticklabels(np.arange(1, 20))
    plt.show()

    # episode zoom in
    # ToDo: CLEAN THIS COPY PASTE MESS UP!

    # store avr_fft amps for 3.4 Hz peak
    amps_high = []
    amps_low = []

    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.6)
    for i in range(19):
        avr_fft = np.mean(cx_fft_episodes[-400:][i*20:(i+1)*20], axis=0)
        avr_fft_std = np.std(cx_fft_episodes[-400:][+ i * 20:(i + 1) * 20], axis=0)
        ax = fig.add_subplot(4,5,i+1)
        ax.plot(x_cx_fft[1:], avr_fft, lw=2, c='k')
        ax.fill_between(x_cx_fft[1:], avr_fft - avr_fft_std, avr_fft + avr_fft_std, color='k', alpha=0.3)
        ax.set_title('eIDs: {0}-{1}, av_rev {2:.2f}'.format(1500 + i*20, 1500 + (i+1)*20, np.mean(rewards[episode_order][-400:][i*20:(i+1)*20])))
        ax.axvline(x=freq1, ymin=0, ymax=avr_fft[peak_idx1]/0.006, c='b', ls='--')
        ax.axvline(x=freq2, ymin=0, ymax=avr_fft[peak_idx2]/0.006, c='orange', ls='--')
        ax.axhline(y=avr_fft[peak_idx1], xmin=0, xmax=freq1/x_cx_fft[-1], c='b', ls='--')
        ax.axhline(y=avr_fft[peak_idx2], xmin=0, xmax=freq2 / x_cx_fft[-1], c='orange', ls='--')
        ax.set_ylim(0, 0.006)
        amps_high.append(np.mean(avr_fft[idx_low1:idx_high1+1]))
        amps_low.append(np.mean(avr_fft[idx_low2:idx_high2 + 1]))

    def func_lin(x, a, b):
        return a*x + b
    popt1, pcov1 = curve_fit(func_lin, np.arange(len(amps_high))[3:], amps_high[3:])
    popt2, pcov2 = curve_fit(func_lin, np.arange(len(amps_low))[3:-4], amps_low[3:-4])
    popt3, pcov3 = curve_fit(func_lin, np.arange(len(amps_low))[-6:], amps_low[-6:])

    # ToDo: chose more windows with same window size but smaller window step size so that after each step 50% of the previous window is kept
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.14, right=0.95, wspace=0.6)
    ax = fig.add_subplot(111)
    ax.plot(amps_high, c='b', marker='o', lw=2, alpha=0.5)
    ax.plot(np.arange(len(amps_high))[3:], func_lin(np.arange(len(amps_high))[3:], *popt1), lw=2, ls='--', c='b')
    ax.plot(amps_low, c='orange', marker='o', lw=2, alpha=0.5)
    ax.plot(np.arange(len(amps_low))[3:-4], func_lin(np.arange(len(amps_low))[3:-4], *popt2), lw=2, ls='--', c='orange')
    ax.plot(np.arange(len(amps_low))[-6:], func_lin(np.arange(len(amps_low))[-6:], *popt3), lw=2, ls='--', c='orange')
    ax.set_xlabel("episode window")
    ax.set_ylabel("amplitude")
    ax.set_xticks(np.arange(19))
    ax.set_xticklabels(np.arange(1, 20))
    plt.show()

    # Interpretation and Story:
    # agents learns to develop 3.4 Hz peak with increasing reward!
    # ToDo: show correlation of amplitude peak with increasing reward!!!
    # retrain agent with new reward function not solely based on pendulum y pos but with dependence on frequency of the movement.
    # check whether refined reward function accelerates learning
    # suggest that agent can learn to develop his own reward
    # questionable is its ability to recognize the 3.4 Hz peak as a feature to be incorporated into his own reward+

    # ToDo: add ratio plot between low and high freq amplitude, showing the development of the ratio between the frequency compopnents
    # ToDo: check develpment of fraction of the 3.4 Hz in the power spectral density. (integral around 3.4Hz, compare to integral across all frequencies)
    # ToDo: plot development of average y-pos