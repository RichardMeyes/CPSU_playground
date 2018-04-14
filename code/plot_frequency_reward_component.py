import numpy as np
import matplotlib.pyplot as plt


def freq_rew(freqs, f_target, c1, c2, c3):
    """

    :param freqs: array containing the frequencies for which the reward should be evaluated

    :param f_target: target frequency, which gives the highest reward

    :param c1: constant that essentially determines the maximum reward at the peak frequency.
    Chosen so that reward_max = 1000

    :param c2: constant that determines the curvature of the reward function, i.e. the fall-off with growing distance
    from the target frequency. Smaller values  lead to a flatter fall-off.

    :param c3: constant that determines the reward-offset. Changes the maximum reward too.

    :return: array containing the calculated reward for the given frequency array
    """
    return c1 * -np.log(c2 * (freqs - f_target) ** 2) + c3

if __name__ == "__main__":

    f_target = 3.4  # Hz
    margin = 5  # Hz
    freqs = np.linspace(f_target - margin, f_target + margin, 1000, endpoint=True)
    reward = freq_rew(freqs=freqs, f_target=f_target, c1=81.94, c2=0.2, c3=0)
    print("maximum reward:", np.max(reward))

    zero_point1 = np.where(reward > 0)[0][0]
    zero_point2 = len(reward) - zero_point1  # symetric function!
    zero_reward_freq1 = freqs[zero_point1]
    zero_reward_freq2 = freqs[zero_point2]
    print("frequency range for positive rewards from {0:.2f} to {1:.2f}".format(zero_reward_freq1, zero_reward_freq2))

    # plot
    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.12, right=0.95, wspace=0.2)
    ax = fig.add_subplot(111)
    ax.plot(freqs, reward, lw=3, c='royalblue', zorder=10)
    ax.axvline(x=f_target, lw=3, ls='--', c='orange', label="target frequency: {0} Hz".format(f_target))
    ax.axvline(x=0, lw=2, ls='-', c='k')
    ax.axhline(y=0, lw=2, ls='-', c='k')
    ax.set_ylabel("reward")
    ax.set_xlabel("frequency")
    ax.grid()
    ax.legend()
    plt.show()