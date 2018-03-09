import h5py
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # mapping
    idx_map = {"cx": 0, "cvx": 1, "px": 2, "py": 3, "pav": 4}

    # read data
    with h5py.File("../data/preprocessed/CartPoleData_full.h5", 'r') as f:
        cp_data = f["CartPoleData_full"][...]

    N_trial = 252  # number of data points per trial
    N_total = len(cp_data)  # number of total data points

    trial_idxs = np.argwhere(cp_data[:, idx_map['cx']] == 0.0)  # indexes for trial starts
    num_trials = len(trial_idxs) - 1  # cut last trial becasue it was depricated

    # create data objects that contain the data for each single trial
    cx_trials = np.zeros((num_trials, N_trial))
    px_trials = np.zeros((num_trials, N_trial))
    py_trials = np.zeros((num_trials, N_trial))
    phi_trials = np.zeros((num_trials, N_trial))
    for i_trial in range(num_trials):
        cx_trials[i_trial] = cp_data[trial_idxs[i_trial][0]:trial_idxs[i_trial][0] + N_trial, idx_map['cx']]
        px_trials[i_trial] = cp_data[trial_idxs[i_trial][0]:trial_idxs[i_trial][0] + N_trial, idx_map['px']]
        py_trials[i_trial] = cp_data[trial_idxs[i_trial][0]:trial_idxs[i_trial][0] + N_trial, idx_map['py']]
        phi_trials[i_trial] = np.arctan2(py_trials[i_trial], px_trials[i_trial]) * 180 / np.pi

    for i_trial in range(num_trials):
        cx = cx_trials[i_trial]
        px = px_trials[i_trial]
        py = py_trials[i_trial]
        phi = phi_trials[i_trial]

        T = 16.65
        dt = T / N_trial
        t = np.linspace(0, T, N_trial)

        # create figure
        fig = plt.figure(figsize=(8, 6))
        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.90, wspace=0.6)
        ax1 = fig.add_subplot(111)
        ax2 = plt.twinx(ax1)

        # plot data
        line_cp, = ax1.plot(t, cx, lw=2, label='cartpole_x')
        line_py, = ax1.plot(t, py, lw=2, label='pendulum_y')
        line_phi, = ax2.plot(t, phi, lw=2, label='angle', color='g')

        # cosmetics
        ax1.legend(handles=[line_cp, line_py, line_phi])
        ax1.set_xlabel('time [s]')
        ax1.set_ylabel('normalized cart_x position')
        ax2.set_ylabel('phi [degrees]')
        ax1.set_title('Trial_ID: {0}'.format(i_trial + 1))

        # save and close figure
        plt.savefig("../pics/cp_trial_{0}".format(i_trial))
        plt.close()