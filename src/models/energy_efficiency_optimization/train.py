# src/models/energy_efficiency_optimization/train.py

import argparse, os, joblib
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

from .utils import load_dataframe, split_and_scale
from .model import EnergyEfficiencyNet


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--save_path", default="models/energy_efficiency_optimization/best_model.pt")
    args = ap.parse_args()

    df = load_dataframe(args.csv)
    train_ds, val_ds, test_ds, scaler, features = split_and_scale(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnergyEfficiencyNet(input_dim=len(features),
                                hidden=args.hidden,
                                dropout=args.dropout).to(device)

    loss_fn = MSELoss()
    opt = Adam(model.parameters(), lr=args.lr)

    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tl = train_epoch(model, tr_loader, opt, loss_fn, device)
        vl = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}: Train {tl:.4f} | Val {vl:.4f}")

        if vl < best_val:
            best_val = vl
            best_state = model.state_dict()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    torch.save({
        "state_dict": best_state,
        "features": features,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }, args.save_path)

    # Final test
    model.load_state_dict(best_state)
    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss = {test_loss:.4f}")


if __name__ == "__main__":
    main()


'''import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import CustomDataset

class EnergyEfficiencyModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EnergyEfficiencyModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = train_loss/len(train_loader)
    return avg_loss

def main():
    # Hyperparameters
    input_size = 8
    hidden_size = 64
    output_size = 1
    learning_rate = 0.01
    num_epochs = 50
    batch_size = 16

    # Load data
    train_dataset = CustomDataset("data/energy_efficiency/train.csv")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnergyEfficiencyModel(input_size, hidden_size, output_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train model
    criterion = torch.nn.MSELoss()
    for epoch in range(1, num_epochs+1):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        print('Epoch: [{}/{}]\tTrain Loss: {:.4f}'.format(epoch, num_epochs, loss))

    # Save model
    torch.save(model.state_dict(), "models/energy_efficiency.pt")

if __name__ == '__main__':
    main()
'''
