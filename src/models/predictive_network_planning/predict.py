# src/models/predictive_network_planning/predict.py

import argparse
import os
import json
import joblib
import pandas as pd
import torch
import numpy as np

from .model import PlanningMLP
from .utils import make_supervised_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Input CSV file (same feature format as training).")
    parser.add_argument("--artifacts", default="models/predictive_network_planning",
                        help="Directory containing best.pt, planning_scaler.pkl, planning_features.json.")
    parser.add_argument("--out_csv", default="pnp_predictions.csv",
                        help="Where to write output predictions.")
    args = parser.parse_args()

    # Validate artifacts exist
    feat_path = os.path.join(args.artifacts, "planning_features.json")
    scaler_path = os.path.join(args.artifacts, "planning_scaler.pkl")
    model_path = os.path.join(args.artifacts, "best.pt")

    if not (os.path.exists(feat_path) and os.path.exists(scaler_path) and os.path.exists(model_path)):
        raise FileNotFoundError(
            f"Missing required artifact files in {args.artifacts}. "
            f"Expected: best.pt, planning_scaler.pkl, planning_features.json"
        )

    # Load feature metadata
    with open(feat_path, "r") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Create supervised frame from CSV
    df, X, y, _ = make_supervised_frame(args.csv)

    # Apply scaler
    X_scaled = scaler.transform(X)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlanningMLP(input_dim=X_scaled.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()

    # Build output DataFrame
    out = df[["site_id", "timestamp"]].copy()
    out["traffic_load_next_pred"] = preds

    # Save output
    out_path = args.out_csv
    out.to_csv(out_path, index=False)
    print(f"[PNP] Saved predictions → {out_path}")


if __name__ == "__main__":
    main()


'''import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from models.predictive_network_planning.model import PredictiveNetworkPlanningModel
from models.predictive_network_planning.utils import PNPDataset, collate_fn


def predict(model: torch.nn.Module, data: PNPDataset, scaler: MinMaxScaler, device: str) -> np.ndarray:
    loader = DataLoader(data, batch_size=1, collate_fn=collate_fn)
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            outputs.append(output.item())
    outputs = np.array(outputs).reshape(-1, 1)
    outputs = scaler.inverse_transform(outputs)
    return outputs


def main(args: argparse.Namespace) -> None:
    # Load model
    model = PredictiveNetworkPlanningModel(input_dim=args.input_dim,
                                           hidden_dim=args.hidden_dim,
                                           num_layers=args.num_layers,
                                           output_dim=args.output_dim,
                                           dropout=args.dropout)
    device = torch.device(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Load data and scaler
    data = pd.read_csv(args.data_file)
    scaler = MinMaxScaler()
    scaler.fit(data.values)

    # Convert data to PyTorch dataset
    data = PNPDataset(data_file=args.data_file)
    inputs, targets = data[0]
    input_dim = inputs.shape[-1]
    output_dim = targets.shape[-1]

    # Make prediction
    prediction = predict(model, data, scaler, device)
    logging.info(f"Input Dimension: {input_dim}, Output Dimension: {output_dim}")
    logging.info(f"Prediction: {prediction}")

    # Save prediction
    os.makedirs(args.output_dir, exist_ok=True)
    np.savetxt(os.path.join(args.output_dir, args.output_file), prediction, delimiter=",")

'''