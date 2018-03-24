import h5py
import numpy as np
import matplotlib.pyplot as plt


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
    num_episodes = len(episode_idxs)

    # cut the last and the first 8 episodes so that 1900 episodes remain
    episode_idxs = episode_idxs[8:-1]
    rewards = rewards[8:]

    # order episodes according to max reward
    episode_order = np.argsort(rewards)

    # calculate episodes per box
    num_boxes = 19
    num_episodes_per_box = int(num_episodes/num_boxes)

    rewards_box = np.zeros([num_episodes_per_box, num_boxes])
    for i_episode, reward in enumerate(rewards):
        row = int(i_episode%num_episodes_per_box)
        col = int(i_episode//num_episodes_per_box)
        rewards_box[row, col] = rewards[episode_order][i_episode]

    # create figure
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.6)
    ax = fig.add_subplot(111)

    ax.boxplot(rewards_box)
    plt.show()