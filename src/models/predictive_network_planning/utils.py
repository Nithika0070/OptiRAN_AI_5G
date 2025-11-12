# src/models/predictive_network_planning/utils.py
import json, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATS = [
    "traffic_load","num_users","average_throughput","peak_throughput",
    "signal_strength_dBm","interference_level_dBm","energy_consumption_kW",
    "handover_failures","latency_ms"
]

BIN_FEATS = ["sleep_mode_enabled"]  # expected as 0/1
ID_COL = "site_id"
TIME_COL = "timestamp"
TARGET_COL = "traffic_load_next"

def _ensure_datetime(df):
    if not np.issubdtype(df[TIME_COL].dtype, np.datetime64):
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])

def make_supervised_frame(csv_path, lags=(1,2,3), roll_windows=(3,6)):
    df = pd.read_csv(csv_path)
    _ensure_datetime(df)
    df = df.sort_values([ID_COL, TIME_COL])

    # Create target per site
    df[TARGET_COL] = df.groupby(ID_COL)["traffic_load"].shift(-1)

    # Lags
    for L in lags:
        df[f"traffic_load_lag{L}"] = df.groupby(ID_COL)["traffic_load"].shift(L)

    # Rolling means (past windows)
    for W in roll_windows:
        df[f"traffic_load_roll{W}"] = (
            df.groupby(ID_COL)["traffic_load"]
              .apply(lambda s: s.shift(1).rolling(W, min_periods=1).mean())
              .reset_index(level=0, drop=True)
        )

    # Drop rows with NA introduced by shift
    df = df.dropna().reset_index(drop=True)

    feature_cols = NUMERIC_FEATS + BIN_FEATS + \
                   [f"traffic_load_lag{L}" for L in lags] + \
                   [f"traffic_load_roll{W}" for W in roll_windows]

    X = df[feature_cols].astype(float).values
    y = df[TARGET_COL].astype(float).values

    return df, X, y, feature_cols

def split_scale(X, y, test_size=0.2, val_size=0.1, random_state=42):
    # first, train+temp vs test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    # then temp -> val/test
    val_ratio = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    return (X_train_s, y_train), (X_val_s, y_val), (X_test_s, y_test), scaler

def save_feature_meta(out_dir, feature_cols):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "planning_features.json"), "w") as f:
        json.dump({"feature_cols": feature_cols}, f)

'''import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(data_path):
    """
    Load data from csv file.

    Args:
        data_path (str): Path to the csv file.

    Returns:
        X (numpy array): Features of the dataset.
        y (numpy array): Labels of the dataset.
    """
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def preprocess_data(X_train, X_test):
    """
    Preprocess data using standard scaling.

    Args:
        X_train (numpy array): Features of the training set.
        X_test (numpy array): Features of the test set.

    Returns:
        X_train_scaled (numpy array): Scaled features of the training set.
        X_test_scaled (numpy array): Scaled features of the test set.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
'''