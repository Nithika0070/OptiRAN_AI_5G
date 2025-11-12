# src/models/dynamic_network_optimization/predict.py

import argparse, os
import numpy as np
import pandas as pd
import torch
from .model import DNOPolicyNet

def fix_value(v):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="final_results/dno.csv")
    args = ap.parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu")
    features = ckpt["features"]
    classes  = ckpt["classes"]
    mean     = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scale    = np.array(ckpt["scaler_scale"], dtype=np.float32)

    df = pd.read_csv(args.csv)

    # 1) Clean + convert to numeric
    X_df = coerce_numeric(df, features)

    # 2) Standardize
    X = (X_df.values - mean) / scale

    # 3) Convert to tensor
    X = torch.tensor(X, dtype=torch.float32)

    # 4) Model inference
    model = DNOPolicyNet(input_dim=len(features), num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).numpy()
        pred_ids = probs.argmax(axis=1)

    # 5) Output only merge-critical columns
    out = df[["site_id", "timestamp"]].copy()
    out["dno_action"] = [classes[i] for i in pred_ids]
    out["dno_confidence"] = probs.max(axis=1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[DNO] Saved predictions → {args.out}")


if __name__ == "__main__":
    main()


'''# src/models/dynamic_network_optimization/predict.py
import argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .model import DNOPolicyNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--csv", required=True, help="CSV with the same feature columns as training")
    ap.add_argument("--out", default="output/dno_predictions.csv")
    args = ap.parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu")
    features = ckpt["features"]
    classes  = ckpt["classes"]
    mean     = np.array(ckpt["scaler_mean"])
    scale    = np.array(ckpt["scaler_scale"])

    df = pd.read_csv(args.csv)
    X = df[features].values
    X = (X - mean) / scale
    X = torch.tensor(X, dtype=torch.float32)

    model = DNOPolicyNet(input_dim=len(features), num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).numpy()
        pred_ids = probs.argmax(axis=1)

    df_out = df.copy()
    df_out["pred_action"] = [classes[i] for i in pred_ids]
    df_out["confidence"]  = probs.max(axis=1)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"Saved predictions → {args.out}")

if __name__ == "__main__":
    main()
'''

'''import torch
import numpy as np
from utils import load_data, preprocess_data
from model import NetworkOptimizer

def predict(model_path, data_path):
    # Load the model from the specified path
    model = NetworkOptimizer()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the data from the specified path
    data = load_data(data_path)
    preprocessed_data = preprocess_data(data)

    # Convert the preprocessed data to a tensor
    input_tensor = torch.from_numpy(preprocessed_data).float()

    # Make predictions using the model
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output = output_tensor.numpy()

    # Postprocess the predictions and return the results
    result = postprocess_predictions(output)
    return result

def postprocess_predictions(predictions):
    # Convert the predictions to a list of recommended actions
    actions = []
    for prediction in predictions:
        if prediction > 0.5:
            actions.append("Add a new network node")
        else:
            actions.append("Remove an existing network node")
    return actions
'''