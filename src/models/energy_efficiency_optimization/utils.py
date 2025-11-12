# src/models/energy_efficiency_optimization/utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

FEATURES = [
    "traffic_load",
    "num_users",
    "average_throughput",
    "peak_throughput",
    "signal_strength_dBm",
    "interference_level_dBm",
    "sleep_mode_enabled",
    "handover_failures",
    "latency_ms",
]

TARGET = "energy_consumption_kW"


def fix_value(v):
    """Convert booleans/strings to numeric floats safely."""
    if pd.isna(v):
        return 0.0
    s = str(v).strip().lower()
    if s in ("true", "yes", "on"): return 1.0
    if s in ("false", "no", "off"): return 0.0
    try: return float(s)
    except: return 0.0


def coerce_numeric(df, cols):
    out = df[cols].copy()
    for c in cols:
        out[c] = out[c].apply(fix_value)
    return out.astype("float32")


class EEODataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET])
    return df


def split_and_scale(df, test_size=0.2, val_size=0.1):
    # 1) Coerce all features to numeric
    X_df = coerce_numeric(df, FEATURES)
    y = df[TARGET].apply(fix_value).values

    X = X_df.values.astype("float32")

    # 2) time-aware splits (no shuffling)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, shuffle=False
    )

    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, shuffle=False
    )

    # 3) Standard scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (
        EEODataset(X_train_s, y_train),
        EEODataset(X_val_s, y_val),
        EEODataset(X_test_s, y_test),
        scaler,
        FEATURES,
    )


'''# src/models/energy_efficiency_optimization/utils.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

FEATURES = [
    "traffic_load",
    "num_users",
    "average_throughput",
    "peak_throughput",
    "signal_strength_dBm",
    "interference_level_dBm",
    "sleep_mode_enabled",
    "handover_failures",
    "latency_ms",
]

TARGET = "energy_consumption_kW"


class EEODataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def split_and_scale(df, test_size=0.2, val_size=0.1):
    X = df[FEATURES].astype(float).values
    y = df[TARGET].astype(float).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, shuffle=False
    )
    val_ratio = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (
        EEODataset(X_train_s, y_train),
        EEODataset(X_val_s, y_val),
        EEODataset(X_test_s, y_test),
        scaler,
        FEATURES,
    )
'''

'''import numpy as np
import pandas as pd
from typing import List

def read_data(filename: str) -> pd.DataFrame:
    """
    Reads the dataset from the given filename and returns a pandas DataFrame object
    """
    df = pd.read_csv(filename)
    return df

def prepare_data(df: pd.DataFrame, input_features: List[str], target: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the dataset and returns the input and target arrays
    """
    X = df[input_features].values
    y = df[target].values
    return X, y

def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalizes the input data to have zero mean and unit variance
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm

def denormalize_data(X_norm: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Denormalizes the input data back to its original scale
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_denorm = X_norm * std + mean
    return X_denorm

def save_model(model, filename: str) -> None:
    """
    Saves the trained model to disk
    """
    joblib.dump(model, filename)

def load_model(filename: str):
    """
    Loads the trained model from disk
    """
    return joblib.load(filename)


'''