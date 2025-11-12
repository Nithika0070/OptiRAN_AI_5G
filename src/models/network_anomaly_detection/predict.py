# src/models/network_anomaly_detection/predict.py
import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from .model import AE, reconstruction_error
from .utils import load_dataframe, make_feature_matrix, load_artifacts

def main():
    parser = argparse.ArgumentParser(description="Score anomalies with trained Autoencoder")
    parser.add_argument("--data_csv", required=True, help="CSV to score")
    parser.add_argument("--model_dir", default="models/network_anomaly_detection", help="Folder with model.pt & artifacts")
    parser.add_argument("--output_csv", default="anomaly_scores.csv", help="Where to write results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # load artifacts
    scaler, feature_cols, threshold = load_artifacts(args.model_dir)
    model_path = Path(args.model_dir) / "model.pt"

    # load model
    input_dim = len(feature_cols)
    model = AE(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    # prepare data
    df = load_dataframe(args.data_csv)
    X_df = make_feature_matrix(df)
    X = scaler.transform(X_df.values).astype(np.float32)
    X_tensor = torch.from_numpy(X).to(args.device)

    with torch.no_grad():
        x_hat = model(X_tensor)
        per_row_mse = reconstruction_error(X_tensor, x_hat, reduction="none").cpu().numpy()

    is_anomaly = (per_row_mse > threshold).astype(int)

    out = df.copy()
    out["anomaly_score"] = per_row_mse
    out["is_anomaly"] = is_anomaly

    out.to_csv(args.output_csv, index=False)
    print(f"Wrote scores to {args.output_csv}")
    print(f"Threshold={threshold:.6f} | anomalies={int(is_anomaly.sum())}/{len(is_anomaly)}")

if __name__ == "__main__":
    main()

'''import argparse
import joblib
import numpy as np
from utils import load_data, preprocess_data

# Define argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to saved model file")
parser.add_argument("--data_path", type=str, required=True, help="Path to data file")
args = parser.parse_args()

# Load saved model
model = joblib.load(args.model_path)

# Load and preprocess data
data = load_data(args.data_path)
preprocessed_data = preprocess_data(data)

# Make predictions
predictions = model.predict(preprocessed_data)

# Print predicted labels
print("Predictions:")
print(predictions)
'''