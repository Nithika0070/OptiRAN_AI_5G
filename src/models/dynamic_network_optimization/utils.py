# src/models/dynamic_network_optimization/utils.py

from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset

FEATURES = [
    "traffic_load", "num_users", "average_throughput", "peak_throughput",
    "signal_strength_dBm", "interference_level_dBm", "energy_consumption_kW",
    "sleep_mode_enabled", "handover_failures", "latency_ms"
]
TARGET = "optimization_action"


def fix_value(v):
    """Coerce everything into numeric."""
    if pd.isna(v):
        return 0.0
    s = str(v).strip().lower()
    if s in ("true", "yes", "on"):
        return 1.0
    if s in ("false", "no", "off"):
        return 0.0
    if s in ("nan", ""):
        return 0.0
    try:
        return float(s)
    except:
        return 0.0


class DNODataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET])
    return df


def split_scale_encode(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    time_column: str = "timestamp"
) -> Tuple[DNODataset, DNODataset, DNODataset, StandardScaler, LabelEncoder, List[str]]:

    # 1) Coerce all numeric features
    X_df = df[FEATURES].copy()
    for col in FEATURES:
        X_df[col] = X_df[col].apply(fix_value)

    # 2) Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET].astype(str).values)

    # 3) Scale numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    # 4) Time-aware split
    df_sorted = df.sort_values(time_column)
    idx = df_sorted.index.values
    split1 = int(len(idx) * (1 - (test_size + val_size)))
    split2 = int(len(idx) * (1 - test_size))
    train_idx, val_idx, test_idx = idx[:split1], idx[split1:split2], idx[split2:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    return (
        DNODataset(X_train, y_train),
        DNODataset(X_val,   y_val),
        DNODataset(X_test,  y_test),
        scaler,
        le,
        le.classes_.tolist(),
    )


'''# src/models/dynamic_network_optimization/utils.py
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset

FEATURES = [
    "traffic_load","num_users","average_throughput","peak_throughput",
    "signal_strength_dBm","interference_level_dBm","energy_consumption_kW",
    "sleep_mode_enabled","handover_failures","latency_ms"
]
TARGET = "optimization_action"

class DNODataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # class ids
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # optional sanity cleanup
    df = df.dropna(subset=FEATURES + [TARGET])
    return df

def split_scale_encode(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    time_aware: bool = True,
    site_column: str = "site_id",
    time_column: str = "timestamp",
) -> Tuple[DNODataset, DNODataset, DNODataset, StandardScaler, LabelEncoder, List[str]]:
    X_df = df[FEATURES].copy()
    y_series = df[TARGET].astype(str)

    # encode label
    le = LabelEncoder()
    y = le.fit_transform(y_series.values)

    # scale numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    # time-aware split (avoid leakage): sort by time if available
    if time_aware and time_column in df.columns:
        df_sorted = df.sort_values(time_column)
        idx = df_sorted.index.values
        split1 = int(len(idx) * (1 - (test_size + val_size)))
        split2 = int(len(idx) * (1 - test_size))
        train_idx, val_idx, test_idx = idx[:split1], idx[split1:split2], idx[split2:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]
    else:
        X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=(test_size + val_size), random_state=42, stratify=y)
        rel = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=rel, random_state=42, stratify=y_tmp)

    return (
        DNODataset(X_train, y_train),
        DNODataset(X_val,   y_val),
        DNODataset(X_test,  y_test),
        scaler,
        le,
        le.classes_.tolist(),
    )
'''
'''import numpy as np
import pandas as pd

def load_data(file_path):
    """
    Load data from a given file path.
    Args:
        file_path (str): The path to the file to load.
    Returns:
        data (pd.DataFrame): A pandas DataFrame containing the loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the loaded data.
    Args:
        data (pd.DataFrame): A pandas DataFrame containing the loaded data.
    Returns:
        processed_data (np.ndarray): A numpy array containing the preprocessed data.
    """
    # Perform some data preprocessing steps, such as scaling, normalization, etc.
    processed_data = data.to_numpy()
    processed_data = np.divide(processed_data, 100)
    return processed_data

def save_data(data, file_path):
    """
    Save processed data to a given file path.
    Args:
        data (np.ndarray): A numpy array containing the processed data.
        file_path (str): The path to the file to save.
    """
    pd.DataFrame(data).to_csv(file_path, index=False)

'''