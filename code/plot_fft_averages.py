import h5py
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr


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

    # # plot single fft as a test
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

    # reorder data according to cumulative episodic reward
    cx_fft_episodes = cx_fft_episodes[episode_order]

    # create data storage for avr_fft amps for 3.4 Hz peak and 2.3 Hz peak
    amps_high = []
    amps_low = []
    window_rewards = []

    # sliding time window analysis params
    episode_offset = 0  # exclude episodes before this offset
    window_size = 100  # episodes
    window_step = 25  # no overlap
    num_window_steps = int((num_episodes-episode_offset)/window_step)

    # calculate data
    avr_ffts = np.zeros((num_window_steps, len(x_cx_fft)-1))
    avr_fft_stds = np.zeros((num_window_steps, len(x_cx_fft) - 1))
    for i_step in range(num_window_steps):
        avr_ffts[i_step] = np.mean(cx_fft_episodes[i_step*window_step+episode_offset:i_step*window_step+window_size+episode_offset], axis=0)
        avr_fft_stds[i_step] = np.std(cx_fft_episodes[i_step*window_step+episode_offset:i_step*window_step+window_size+episode_offset], axis=0)
        amps_high.append(np.mean(avr_ffts[i_step][idx_low1:idx_high1+1]))
        amps_low.append(np.mean(avr_ffts[i_step][idx_low2:idx_high2 + 1]))
        window_rewards.append(np.mean(rewards[episode_order][i_step*window_step+episode_offset: i_step*window_step+window_size+episode_offset]))

    # plot fft development over time
    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.6)
    k = int(window_size/window_step)  # plot every k-th window.
    for i_step in range(19):
        # plot
        ax = fig.add_subplot(4, 5, i_step+1)
        ax.plot(x_cx_fft[1:], avr_ffts[k*i_step], lw=2, c='k')
        ax.fill_between(x_cx_fft[1:],
                        avr_ffts[k*i_step] - avr_fft_stds[k*i_step],
                        avr_ffts[k*i_step] + avr_fft_stds[k*i_step],
                        color='k', alpha=0.3)
        ax.set_title('eIDs: {0}-{1}, av_rev {2:.2f}'.format(i_step*window_size+episode_offset,
                                                            (i_step+1)*window_size+episode_offset,
                                                            np.mean(rewards[episode_order][i_step*window_size+episode_offset: (i_step+1)*window_size+episode_offset])))
        ax.axvline(x=freq1, ymin=0, ymax=avr_ffts[k*i_step][peak_idx1]/0.006, c='b', ls='--')
        ax.axvline(x=freq2, ymin=0, ymax=avr_ffts[k*i_step][peak_idx2]/0.006, c='orange', ls='--')
        ax.axhline(y=avr_ffts[k*i_step][peak_idx1], xmin=0, xmax=freq1/x_cx_fft[-1], c='b', ls='--')
        ax.axhline(y=avr_ffts[k*i_step][peak_idx2], xmin=0, xmax=freq2 / x_cx_fft[-1], c='orange', ls='--')
        ax.set_ylim(0, 0.006)

    # define fitting borders
    low_border = 9  # chosen such that episodes with negative reward are cut off
    med_border = -20

    def func_lin(x, a, b):
        return a*x + b
    popt1, pcov1 = curve_fit(func_lin, np.arange(len(amps_high))[low_border:], amps_high[low_border:])
    popt2, pcov2 = curve_fit(func_lin, np.arange(len(amps_low))[low_border:med_border], amps_low[low_border:med_border])
    popt3, pcov3 = curve_fit(func_lin, np.arange(len(amps_low))[med_border-2:], amps_low[med_border-2:])

    # amps dev over ordered episode
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.08, right=0.95, wspace=0.2)
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax3 = ax2.twinx()

    ax.plot(amps_high, c='dodgerblue', marker='o', lw=2, alpha=0.5)
    ax.plot(np.arange(len(amps_high))[low_border:], func_lin(np.arange(len(amps_high))[low_border:], *popt1), lw=2, ls='--', c='royalblue')
    ax.plot(amps_low, c='darkorange', marker='o', lw=2, alpha=0.5)
    ax.plot(np.arange(len(amps_low))[low_border:med_border], func_lin(np.arange(len(amps_low))[low_border:med_border], *popt2), lw=2, ls='--', c='orangered')
    ax.plot(np.arange(len(amps_low))[med_border-2:], func_lin(np.arange(len(amps_low))[med_border-2:], *popt3), lw=2, ls='--', c='orangered')
    ax.set_xlabel("episode window")
    ax.set_ylabel("amplitude")
    ax.set_ylim(0, 0.006)
    ax.grid()

    # amps vs rewards corr
    r_high, p_high = pearsonr(window_rewards[low_border:], amps_high[low_border:])
    r_low, p_low = pearsonr(window_rewards[low_border:], amps_low[low_border:])
    r_ratio, p_ratio = pearsonr(window_rewards[low_border:], np.array(amps_high[low_border:])/np.array(amps_low[low_border:]))
    h1, = ax2.plot(window_rewards[low_border:], amps_high[low_border:], c='dodgerblue', marker='o', lw=2, alpha=0.5, label='R = {0:.2f}, p = {1:.2E}'.format(r_high, p_high))
    h2, = ax2.plot(window_rewards[low_border:], amps_low[low_border:], c='darkorange', marker='o', lw=2, alpha=0.5, label='R = {0:.2f}, p = {1:.2E}'.format(r_low, p_low))
    h3, = ax3.plot(window_rewards[low_border:], np.array(amps_high[low_border:])/np.array(amps_low[low_border:]), c='g', marker='o', lw=2, alpha=0.5, label='R = {0:.2f}, p = {1:.2E}'.format(r_ratio, p_ratio))
    ax2.set_xlabel("average reward within episode window")
    ax2.set_ylim(0, 0.0065)
    ax2.set_xlim(xmax=1000)
    ax2.grid()
    ax2.legend(loc=2, handles=[h1, h2, h3])
    plt.show()

    # ToDo: check develpment of fraction of the 3.4 Hz in the power spectral density. (integral around 3.4Hz, compare to integral across all frequencies)