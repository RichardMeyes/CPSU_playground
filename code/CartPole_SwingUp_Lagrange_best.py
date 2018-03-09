import numpy as np
import pandas as pd
from scipy.fftpack import fft
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # read data
    fp_cp_data = "../data/preprocessed/CartPoleData.xlsx"
    cp_data = pd.read_excel(io=fp_cp_data, sheet_name=0, header=0, index_col=0)

    cx = cp_data['cx']
    px = cp_data['px']
    py = cp_data['py']
    phi = np.arctan2(py, px) * 180 / np.pi

    T = 16.65
    dt = T / len(cx)
    t = np.linspace(0, T, len(cx))

    # find segments
    s12_start = np.argwhere(py > 0.99)[0][0]
    s12_end = np.argwhere(py > 0.9999)[0][0]
    # ToDo: find start of the third segment with sliding time window and variance analysis in this time window
    s23 = 186  # hard coded for now, found by visual inspection

    # calculate some parameters
    cx_2_mean = np.mean(cx[s12_end:s23])
    cx_3_mean = np.mean(cx[s23:])

    # perform FFT on cart pole movement
    N2 = len(cx[s12_end:s23])
    N3 = len(cx[s23:])
    N = N2 + N3
    cx_fft_2 = fft(cx[s12_end:s23])
    cx_fft_3 = fft(cx[s23:])
    x_cx_fft_2 = np.linspace(0, 1 / (2 * dt), N2 // 2)
    x_cx_fft_3 = np.linspace(0, 1 / (2 * dt), N3 // 2)

    # example FFTs
    # segment 2
    x_dummy2 = np.linspace(0, N2 * dt, N2) + s12_end*dt
    dummy2 = 0.06 * np.sin(0.6 * 2 * np.pi * x_dummy2) + 0.005 * np.sin(3.4 * 2 * np.pi * x_dummy2) + cx_2_mean
    dummy2_fft = fft(dummy2)
    x_dummy2_fft = np.linspace(0, 1 / (2 * dt), N2 // 2)

    # segment 3
    x_dummy3 = np.linspace(0, N3 * dt, N3) + s23*dt
    dummy3 = 0.007 * np.sin(0.5 * 2 * np.pi * x_dummy3) + 0.005 * np.sin(3.6 * 2 * np.pi * x_dummy3) + cx_3_mean
    dummy3_fft = fft(dummy3)
    x_dummy3_fft = np.linspace(0, 1 / (2 * dt), N3 // 2)

    # create figure
    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.98, wspace=0.6)

    # add subplots
    ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=2)
    ax2 = plt.twinx(ax1)
    ax3 = plt.subplot2grid((1, 4), (0, 2))
    ax4 = plt.subplot2grid((1, 4), (0, 3))

    # plot data
    line_cp, = ax1.plot(t, cx, lw=2, label='cartpole_x')
    line_py, = ax1.plot(t, py, lw=2, label='pendulum_y')
    line_phi, = ax2.plot(t, phi, lw=2, label='angle', color='g')
    line_dum1, = ax1.plot(x_dummy2, dummy2, lw=2, c='r', label='dummy signal in segment 2')
    line_dum2, = ax1.plot(x_dummy3, dummy3, lw=2, c='m', label='dummy signal in segment 3')

    # plot segments
    ax1.axvspan(xmin=s12_start*dt, xmax=s12_end*dt, lw=2, ls='--', color='k', alpha=0.2)
    ax1.axvline(x=s23*dt, lw=2, ls='--', c='k')

    # cosmetics
    ax1.legend(handles=[line_cp, line_py, line_phi, line_dum1, line_dum2, ])
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('normalized cart_x position')
    ax2.set_ylabel('phi [degrees]')
    ax1.set_title('maximum reward: {0}'.format(cp_data['reward'].data[-1]))

    # plot FFT
    ax3.plot(x_cx_fft_2[1:], 2 / N2 * np.abs(cx_fft_2[1:N2 // 2]), lw=2, label='FFT of cartpole_x')
    ax3.plot(x_dummy2_fft[1:], 2 / N2 * np.abs(dummy2_fft[1:N2 // 2]), lw=2, c='r', label='dummy FFT')
    ax4.plot(x_cx_fft_3[1:], 2 / N3 * np.abs(cx_fft_3[1:N3 // 2]), lw=2, label='FFT of cartpole_x')
    ax4.plot(x_dummy3_fft[1:], 2 / N3 * np.abs(dummy3_fft[1:N3 // 2]), lw=2, c='m', label='dummy FFT')

    # cosmetics
    for i, ax in enumerate([ax3, ax4]):
        ax.legend()
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('Amplitude')
        ax.set_title('FFT in segment {0}'.format(i+2))

    plt.show()