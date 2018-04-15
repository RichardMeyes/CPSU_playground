import numpy as np
import matplotlib.pyplot as plt


def freq_rew(freqs_, f_target_, function):
    """

    :param freqs_: array containing the frequencies for which the reward should be evaluated

    :param f_target_: target frequency, which gives the highest reward

    :param function_: shape of the reward function. Can either be 'log', 'gaussian' or 'combined'

    :return: array containing the calculated reward for the given frequency array
    """

    if function == 'log':
        c1 = 81.41
        c2 = 0.2
        c3 = 0
        return c1 * -np.log(c2 * (freqs_ - f_target_) ** 2) + c3
    elif function == 'gaussian':
        c1 = 100
        c2 = 0.5
        c3 = 1100
        return c1 * -np.exp(c2 * (freqs_ - f_target_) ** 2) + c3
    elif function == 'combined':
        c1 = 500
        c2 = 1.0
        c3 = 1500
        gaussian = c1 * -np.exp(c2 * (freqs_ - f_target_) ** 2) + c3
        zp1 = np.where(gaussian > 0)[0][0]
        zp2 = len(gaussian) - zp1  # symmetric function!

        c1 = 81.41
        c2 = 0.9
        c3 = 0
        log = c1 * -np.log(c2 * (freqs_ - f_target_) ** 2) + c3

        """
        combination is made at the zero points of the gaussian function. The merging of the logarithmic function could
        be made earlier towards the target frequency in order to push the development of the target frequency more.
        Some experimentation is needed to find out what reward shape is best.
        """

        combined = np.zeros(len(freqs_))
        combined[:zp1] = log[:zp1]
        combined[zp1:zp2] = gaussian[zp1:zp2]
        combined[zp2:] = log[zp2:]

        return combined


if __name__ == "__main__":

    f_target = 3.4  # Hz
    margin = 5  # Hz
    freqs = np.linspace(f_target - margin, f_target + margin, 1000, endpoint=True)
    rew_function = 'combined'
    reward = freq_rew(freqs_=freqs, f_target_=f_target, function=rew_function)
    print("maximum reward:", np.max(reward))

    zero_point1 = np.where(reward > 0)[0][0]
    zero_point2 = len(reward) - zero_point1  # symmetric function!
    zero_reward_freq1 = freqs[zero_point1]
    zero_reward_freq2 = freqs[zero_point2]
    print("frequency range for positive rewards from {0:.2f} to {1:.2f}".format(zero_reward_freq1, zero_reward_freq2))

    # plot
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(top=0.96, bottom=0.1, left=0.08, right=0.98, wspace=0.2)
    ax = fig.add_subplot(111)
    ax.plot(freqs, reward, lw=3, c='royalblue', zorder=10)
    ax.axvline(x=f_target, lw=3, ls='--', c='orange', label="target frequency: {0} Hz".format(f_target))
    ax.axvline(x=0, lw=2, ls='-', c='k')
    ax.axhline(y=0, lw=2, ls='-', c='k')
    ax.set_ylim(-1100, 1100)
    ax.set_ylabel("reward")
    ax.set_xlabel("frequency [Hz]")
    ax.grid()
    ax.legend()
    plt.show()