import h5py
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt


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

    episode_idxs = np.argwhere(cp_data[:, idx_map['cx']] == 0.0)  # indexes for episode starts
    num_episodes = len(episode_idxs) - 1  # cut last episode because it was deprecated

    # create data objects that contain the data for each single episode
    cx_episodes = np.zeros((num_episodes, N_episode))
    px_episodes = np.zeros((num_episodes, N_episode))
    py_episodes = np.zeros((num_episodes, N_episode))
    phi_episodes = np.zeros((num_episodes, N_episode))
    for i_episode in range(num_episodes):
        cx_episodes[i_episode] = cp_data[episode_idxs[i_episode][0]:episode_idxs[i_episode][0] + N_episode, idx_map['cx']]
        px_episodes[i_episode] = cp_data[episode_idxs[i_episode][0]:episode_idxs[i_episode][0] + N_episode, idx_map['px']]
        py_episodes[i_episode] = cp_data[episode_idxs[i_episode][0]:episode_idxs[i_episode][0] + N_episode, idx_map['py']]
        phi_episodes[i_episode] = np.arctan2(py_episodes[i_episode], px_episodes[i_episode]) * 180 / np.pi

    # get order of episodes in descending order of maximum reward
    episode_order = np.argsort(rewards)[::-1]

    for i_episode in episode_order:
        cx = cx_episodes[i_episode]
        px = px_episodes[i_episode]
        py = py_episodes[i_episode]
        phi = phi_episodes[i_episode]

        T = 16.65
        dt = T / N_episode
        t = np.linspace(0, T, N_episode)

        seg = 3.0
        s12 = int(N_episode*seg/T)  # seconds

        # perform FFT on cart pole movement
        cx_fft = fft(cx[s12:])
        N_fft = len(cx[s12:])
        x_cx_fft = np.linspace(0, 1 / (2 * dt), N_fft // 2)

        # create figure
        fig = plt.figure(figsize=(12, 6))
        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.6)
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.twinx(ax1)
        ax3 = plt.subplot2grid((1, 2), (0, 1))

        # plot data
        line_cp, = ax1.plot(t, cx, lw=2, label='cartpole_x')
        line_py, = ax1.plot(t, py, lw=2, label='pendulum_y')
        line_phi, = ax2.plot(t, phi, lw=2, label='angle', color='g')

        # plot segments
        ax2.axhspan(ymin=85, ymax=95, lw=2, ls='--', color='g', alpha=0.3)
        ax1.axvline(x=s12 * dt, lw=2, ls='--', c='k')

        # plot FFT
        ax3.plot(x_cx_fft[1:], 2 / N_fft * np.abs(cx_fft[1:N_fft // 2]), lw=2, label='FFT of cartpole_x')

        # cosmetics
        ax1.legend(handles=[line_cp, line_py, line_phi])
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('normalized cart_x position')
        ax2.set_ylabel('phi [degrees]')
        ax1.set_title('episode_ID: {0}, maximum reward: {1}'.format(i_episode + 1, rewards[i_episode]))
        ax2.set_ylim(-180, 180)
        ax2.set_yticks(np.arange(-180, 181, 45))
        ax1.set_xticks(np.arange(0, 17.6, 2.5))
        ax1.set_yticks(np.arange(-1.0, 1.1, 0.25))

        ax3.legend()
        ax3.set_xlabel('frequency [Hz]')
        ax3.set_ylabel('Amplitude')

        # save and close figure
        plt.show()
        # plt.savefig("../pics/cp_episode_{0}".format(i_episode))
        plt.close()

        # ToDo: high-pass, so that low frequs are cut - then boxplot, 1900 episodes, 19 boxplots, 100 episodes per box
        # ToDo: FFT after initial swing up, plot development of amp and freq depending on episode and on reward.
        # ToDo: Show that Agent learns to find the correct mps and freqs as learning progresses