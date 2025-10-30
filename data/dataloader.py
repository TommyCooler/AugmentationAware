import os
import pandas as pd
import numpy as np


def smd_sub_ds_processing(data_path, filename, normalized=True):

    train_data, test_data, labels = None, None, None

    for file_name in os.listdir(data_path):
        if file_name.startswith(filename):
            file_path = os.path.join(data_path, file_name)

            if file_name.endswith("train.npy"):
                train_data = np.load(file_path).T
                print("Train shape:", train_data.shape)

            elif file_name.endswith("test.npy"):
                test_data = np.load(file_path).T
                print("Test shape:", test_data.shape)

            elif file_name.endswith("labels.npy"):
                raw_labels = np.load(file_path)
                print("Labels shape:", raw_labels.shape)
                # Convert multi-dimensional labels to binary 0/1
                labels = np.asarray([1.0 if 1.0 in row else 0.0 for row in raw_labels])
                print("Labels shape:", labels.shape)

    # Normalize train and test data to [0, 1]
    if normalized and train_data is not None and test_data is not None:
        max_val = np.nanmax(train_data)
        min_val = np.nanmin(train_data)

        # Normalize each channel independently
        train_data = (train_data - min_val) / (max_val - min_val)
        test_data = (test_data - min_val) / (max_val - min_val)

    return train_data, test_data, labels


def ecg_sub_ds_processing(data_path, filename, normalized=True):
    train_file = os.path.join(data_path, "labeled/train", filename)
    test_file = os.path.join(data_path, "labeled/test", filename)

    train_df = pd.DataFrame(pd.read_pickle(open(train_file, "rb")))
    test_df = pd.DataFrame(pd.read_pickle(open(test_file, "rb")))

    train_data = train_df[[0, 1]].to_numpy().T  # [2, T]
    test_data = test_df[[0, 1]].to_numpy().T
    test_label = test_df[2].to_numpy().reshape(-1)  # [T]

    if normalized:
        f_max = max(np.max(train_data), np.max(test_data))
        f_min = min(np.min(train_data), np.min(test_data))
        train_data = (train_data - f_min) / (f_max - f_min)
        test_data = (test_data - f_min) / (f_max - f_min)

    print(type(train_data), type(test_data), type(test_label))
    breakpoint()

    return train_data, test_data, test_label


"""process dataset Power-Demand"""


def pd_sub_ds_processing(data_path, filename=None, normalized=True):
    fn = "power_data.pkl"
    train_path = os.path.join(data_path, "labeled/train", fn)
    test_path = os.path.join(data_path, "labeled/test", fn)

    tr_df = pd.DataFrame(pd.read_pickle(open(train_path, "rb")))
    te_df = pd.DataFrame(pd.read_pickle(open(test_path, "rb")))

    train_data = tr_df[[0]].to_numpy().T  # shape: (1, time)
    test_data = te_df[[0]].to_numpy().T
    test_label = te_df[1].to_numpy().flatten()

    # Normalize to [0, 1]
    if normalized:
        min_val = np.min(train_data)
        max_val = np.max(train_data)
        if max_val > min_val:  # prevent divide-by-zero
            train_data = (train_data - min_val) / (max_val - min_val)
            test_data = (test_data - min_val) / (max_val - min_val)

    return train_data, test_data, test_label


"--------------------------------------------------"


def gesture_sub_ds_processing(
    data_path, filename="ann_gun_CentroidA.pkl", normalized=True
):
    fn = "ann_gun_CentroidA.pkl"
    # Load pickle files
    train_path = os.path.join(data_path, "train/", fn)
    test_path = os.path.join(data_path, "test/", fn)

    tr_data = pd.DataFrame(pd.read_pickle(open(train_path, "rb")))
    te_data = pd.DataFrame(pd.read_pickle(open(test_path, "rb")))

    # train_df = pd.DataFrame(pd.read_pickle(open(train_file, 'rb')))
    # test_df = pd.DataFrame(pd.read_pickle(open(test_file, 'rb')))

    # Convert DataFrame to NumPy arrays
    train_data = tr_data[[0, 1]].to_numpy().T
    test_data = te_data[[0, 1]].to_numpy().T
    test_label = te_data[2].to_numpy().flatten()

    # Normalize both X and Y coordinates to [0, 1]
    if normalized:
        min_val = min(np.min(train_data[0]), np.min(train_data[1]))
        max_val = max(np.max(train_data[0]), np.max(train_data[1]))
        scale = max_val - min_val

        if scale > 0:
            train_data = (train_data - min_val) / scale
            test_data = (test_data - min_val) / scale

    return train_data, test_data, test_label


"---------------------------------------------------------"


def ucr_sub_ds_processing(data_path, filename, normalized=True):

    train_data = None
    test_data = None
    test_labels = None

    # Load each relevant file
    for file_name in os.listdir(data_path):
        if file_name.startswith(filename):
            full_path = os.path.join(data_path, file_name)

            if file_name.endswith("train.npy"):
                train_data = np.load(full_path).T

            elif file_name.endswith("test.npy"):
                test_data = np.load(full_path).T

            elif file_name.endswith("labels.npy"):
                test_labels = np.load(full_path).flatten()

    # Normalize only the first channel if enabled
    if normalized and train_data is not None and test_data is not None:
        min_val = np.min(train_data[0])
        max_val = np.max(train_data[0])

        train_data[0] = (train_data[0] - min_val) / (max_val - min_val)
        test_data[0] = (test_data[0] - min_val) / (max_val - min_val)

    # train_data = np.asarray(train_data,dtype=np.float32)
    # print
    # test_data = np.asarray(test_data,dtype=np.float32)
    # test_labels = np.asarray(test_labels,dtype=np.float32)
    return train_data, test_data, test_labels


"----------------------------------------------------------"


def smap_msl_sub_ds_processing(data_path, filename, normalized=True):
    train_data, test_data, labels = None, None, None

    print("Loading files...")

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)

        if file_name.startswith(filename):
            if file_name.endswith("train.npy"):
                train_data = np.load(file_path).T
                print("Train shape:", train_data.shape)

            elif file_name.endswith("test.npy"):
                test_data = np.load(file_path).T
                print("Test shape:", test_data.shape)

            elif file_name.endswith("labels.npy"):
                raw_labels = np.load(file_path)
                print("Raw label shape:", raw_labels.shape)
                # Convert multi-label to binary labels
                labels = np.array([1.0 if 1.0 in row else 0.0 for row in raw_labels])

    # Normalize if enabled
    if normalized and train_data is not None and test_data is not None:
        min_val = np.nanmin(train_data)
        max_val = np.nanmax(train_data)
        train_data = (train_data - min_val) / (max_val - min_val)
        test_data = (test_data - min_val) / (max_val - min_val)

    return train_data, test_data, labels


"-------------------------------------------------------------------"


def psm_sub_ds_processing(data_path, filename=None, normalized=True):
    # Load CSV files
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    label_df = pd.read_csv(os.path.join(data_path, "test_label.csv"))

    # Convert to numpy arrays (excluding first column)
    train_data = train_df.iloc[:, 1:].to_numpy().T
    test_data = test_df.iloc[:, 1:].to_numpy().T
    test_label = label_df.iloc[:, -1].to_numpy().flatten()

    # Normalize if requested
    if normalized:
        min_val = np.nanmin(train_data)
        max_val = np.nanmax(train_data)

        # Avoid division by zero
        if max_val > min_val:
            train_data = (train_data - min_val) / (max_val - min_val)
            test_data = (test_data - min_val) / (max_val - min_val)

        # Fill NaNs in train_data using simple linear extrapolation
        for dim in range(train_data.shape[0]):
            for i in range(2, train_data.shape[1]):
                if np.isnan(train_data[dim, i]):
                    prev2 = train_data[dim, i - 2]
                    prev1 = train_data[dim, i - 1]
                    if np.isfinite(prev1) and np.isfinite(prev2):
                        train_data[dim, i] = prev1 + (prev1 - prev2)

    return train_data, test_data, test_label


"---------------------------------------------------------------"


def swat_sub_ds_processing(data_path, filename="_10.npy", normalized=True):
    train_data = None
    test_data = None
    labels = None

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)

        if file_name.endswith("train" + filename):
            train_data = np.load(file_path).T
            print("Train shape:", train_data.shape)

        elif file_name.endswith("test" + filename):
            test_data = np.load(file_path).T
            print("Test shape:", test_data.shape)

        elif file_name.endswith("labels" + filename):
            labels = np.load(file_path).flatten()
            print("Labels shape:", labels.shape)

    # Normalize to [0, 1] if requested
    if normalized and train_data is not None and test_data is not None:
        max_val = np.nanmax(train_data)
        min_val = np.nanmin(train_data)

        train_data = (train_data - min_val) / (max_val - min_val)
        test_data = (test_data - min_val) / (max_val - min_val)

    return train_data, test_data, labels


"---------------------------------------------------------"


def wadi_sub_ds_processing(data_path, filename="_10.npy", normalized=True):
    train_data, test_data, labels = None, None, None

    for file_name in os.listdir(data_path):
        full_path = os.path.join(data_path, file_name)

        if file_name.endswith("train" + filename):
            train_data = np.load(full_path).T
            print("Train shape:", train_data.shape)

        elif file_name.endswith("test" + filename):
            test_data = np.load(full_path).T
            print("Test shape:", test_data.shape)

        elif file_name.endswith("labels" + filename):
            labels = np.load(full_path).flatten()
            print("Labels shape:", labels.shape)

    # Normalize if requested
    if normalized and train_data is not None and test_data is not None:
        max_val = np.nanmax(train_data)
        min_val = np.nanmin(train_data)

        # Normalize all values to [0, 1]
        train_data = (train_data - min_val) / (max_val - min_val)
        test_data = (test_data - min_val) / (max_val - min_val)

    return train_data, test_data, labels


"-------------------------------------------------------------"
