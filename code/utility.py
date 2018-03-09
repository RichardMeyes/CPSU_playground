import h5py
import pandas as pd


def convert_data():
    # convert data to hdf5 file format
    fp_cp_data = "../data/raw/CartPoleData_full.xlsx"
    cp_data = pd.read_excel(io=fp_cp_data, sheet_name=0, header=0, index_col=0)

    with h5py.File("../data/preprocessed/CartPoleData_full.h5", 'w') as f:
        f.create_dataset("CartPoleData_full", data=cp_data)


if __name__ == "__main__":

    convert_data()