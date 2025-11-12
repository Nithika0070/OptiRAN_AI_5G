# src/models/network_anomaly_detection/train.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.models.network_anomaly_detection.model import AE, reconstruction_error


def train_autoencoder(data_csv, out_path="models/network_anomaly_detection/model.pt"):
    df = pd.read_csv(data_csv)

    # Only numeric features
    numeric_df = df.select_dtypes(include="number")
    X = numeric_df.values.astype(np.float32)

    # Scale
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    input_dim = X.shape[1]

    model = AE(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_s, dtype=torch.float32)

    # Train
    for epoch in range(20):
        optimizer.zero_grad()
        recon = model(X_t)
        loss = loss_fn(recon, X_t)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/20  Loss={loss.item():.4f}")

    # Compute reconstruction error threshold
    with torch.no_grad():
        recon = model(X_t)
        errors = reconstruction_error(X_t, recon).numpy()

    threshold = np.percentile(errors, 95)

    # ✅ Save ALL needed metadata
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "threshold": float(threshold)
        },
        out_path
    )

    print(f"Saved anomaly model with metadata → {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", required=True)
    args = parser.parse_args()

    train_autoencoder(args.data_csv)


'''# src/models/network_anomaly_detection/train.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import AE, reconstruction_error
from .utils import (load_dataframe, make_feature_matrix, train_val_split,
                    fit_scale, save_artifacts, FEATURE_COLUMNS)

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder for Network Anomaly Detection")
    parser.add_argument("--data_csv", required=True, help="Path to your dataset CSV")
    parser.add_argument("--out_dir", default="models/network_anomaly_detection", help="Where to save model & artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bottleneck", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--quantile", type=float, default=0.99, help="Threshold quantile for recon error")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    df = load_dataframe(args.data_csv)
    X_df = make_feature_matrix(df)

    X_train_np, X_val_np = train_val_split(X_df, val_size=0.2, seed=42)
    scaler = fit_scale(X_train_np)
    X_train = scaler.transform(X_train_np).astype(np.float32)
    X_val   = scaler.transform(X_val_np).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = AE(input_dim=X_train.shape[1],
               hidden_dim=args.hidden_dim,
               bottleneck=args.bottleneck,
               dropout=args.dropout).to(args.device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(args.device)
            x_hat = model(xb)
            loss = reconstruction_error(xb, x_hat, reduction="mean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)

        # validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for (xb,) in val_loader:
                xb = xb.to(args.device)
                x_hat = model(xb)
                loss = reconstruction_error(xb, x_hat, reduction="none")
                val_losses.append(loss.cpu().numpy())
            val_losses = np.concatenate(val_losses)
            val_mean = float(val_losses.mean())

        print(f"Epoch {epoch:03d} | train_mse={epoch_loss:.6f} | val_mse={val_mean:.6f}")

        if val_mean < best_val:
            best_val = val_mean
            torch.save(model.state_dict(), out_dir / "model.pt")

    # pick threshold from validation distribution (high quantile)
    threshold = float(np.quantile(val_losses, args.quantile))
    save_artifacts(str(out_dir), scaler, FEATURE_COLUMNS, threshold)
    print(f"Saved model + scaler + threshold to: {out_dir}")
    print(f"Chosen anomaly threshold (quantile {args.quantile}): {threshold:.6f}")

if __name__ == "__main__":
    main()
    '''

'''import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.network_anomaly_detection import NetworkAnomalyDetection
from utils import NetworkAnomalyDataset, load_data, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network anomaly detection model')
    parser.add_argument('--data_path', type=str, default='data/network_data.csv', help='Path to network data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()

    # Load and preprocess data
    df = pd.read_csv(args.data_path)
    data = load_data(df)

    # Split data into training and validation sets
    train_set = NetworkAnomalyDataset(data[:8000])
    val_set = NetworkAnomalyDataset(data[8000:])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Instantiate model and optimizer
    model = NetworkAnomalyDetection(input_size=6, hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    train(model, train_loader, val_loader, optimizer, num_epochs=args.num_epochs)

    # Save trained model
    torch.save(model.state_dict(), 'network_anomaly_detection.pt')
'''