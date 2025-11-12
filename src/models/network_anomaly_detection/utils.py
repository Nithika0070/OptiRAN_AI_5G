# src/models/network_anomaly_detection/utils.py
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features we’ll use from your dataset
FEATURE_COLUMNS = [
    "traffic_load",
    "num_users",
    "average_throughput",
    "peak_throughput",
    "signal_strength_dBm",
    "interference_level_dBm",
    "energy_consumption_kW",
    "sleep_mode_enabled",      # keep as 0/1
    "handover_failures",
    "latency_ms",
]

DROP_COLUMNS = ["site_id", "timestamp", "optimization_action"]

def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # basic guards: ensure columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df

def make_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # drop columns we don't train on (ok if not present)
    for c in DROP_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=c)
    # keep only feature columns (order matters!)
    X = df[FEATURE_COLUMNS].copy()
    # coerce boolean-like to 0/1
    if X["sleep_mode_enabled"].dtype == bool:
        X["sleep_mode_enabled"] = X["sleep_mode_enabled"].astype(int)
    return X

def train_val_split(X: pd.DataFrame, val_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    X_train, X_val = train_test_split(X.values, test_size=val_size, random_state=seed, shuffle=True)
    return X_train, X_val

def fit_scale(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def save_artifacts(out_dir: str, scaler: StandardScaler, feature_cols: List[str], threshold: float) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, Path(out_dir) / "scaler.pkl")
    with open(Path(out_dir) / "features.json", "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)
    with open(Path(out_dir) / "threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f, indent=2)

def load_artifacts(art_dir: str):
    scaler = joblib.load(Path(art_dir) / "scaler.pkl")
    with open(Path(art_dir) / "features.json") as f:
        feature_cols = json.load(f)["features"]
    with open(Path(art_dir) / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return scaler, feature_cols, threshold

'''import numpy as np
from sklearn.preprocessing import StandardScaler

class DataUtils:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_data(self, x):
        x = self.scaler.fit_transform(x)
        return x

    def split_data(self, x, y, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        x_train = self.preprocess_data(x_train)
        x_test = self.preprocess_data(x_test)
        y_train = np.asarray(y_train).astype('float32')
        y_test = np.asarray(y_test).astype('float32')
        return x_train, x_test, y_train, y_test
'''