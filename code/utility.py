import h5py
import numpy as np
import pandas as pd


def convert_data():
    # convert data to hdf5 file format
    fp_cp_data = "../data/raw/CartPoleData_full.xlsx"
    cp_data = pd.read_excel(io=fp_cp_data, sheet_name=0, header=0, index_col=0)

    with h5py.File("../data/preprocessed/CartPoleData_full.h5", 'w') as f:
        f.create_dataset("CartPoleData_full", data=cp_data)


def add_rewards():
    # get rewards and add them to the h5 file
    fp_cp_data = "../data/raw/CartPoleData_full_2.csv"
    cp_data = pd.read_csv(fp_cp_data, delimiter=';', header=0, index_col=0)
    rewards = cp_data['reward'][~np.isnan(cp_data['reward'])]

    with h5py.File("../data/preprocessed/CartPoleData_full.h5", 'a') as f:
        f.create_dataset("EpisodeRewards", data=np.array(rewards))


if __name__ == "__main__":
    pass
    # convert_data()
    # add_rewards()