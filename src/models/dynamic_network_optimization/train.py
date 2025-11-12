
# src/models/dynamic_network_optimization/train.py

import argparse, os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score
from .model import DNOPolicyNet
from .utils import load_dataframe, split_scale_encode, FEATURES


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_y, all_p = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss_sum += loss.item() * X.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += X.size(0)
        all_y.extend(y.cpu())
        all_p.extend(preds.cpu())
    acc = correct / total
    f1 = f1_score(all_y, all_p, average="weighted")
    return loss_sum / total, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--hidden", default=128, type=int)
    ap.add_argument("--dropout", default=0.2, type=float)
    ap.add_argument("--save_path", default="models/dynamic_network_optimization/best.pt")
    args = ap.parse_args()

    df = load_dataframe(args.csv)
    train_ds, val_ds, test_ds, scaler, le, classes = split_scale_encode(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNOPolicyNet(input_dim=len(FEATURES), num_classes=len(classes),
                         hidden=args.hidden, dropout=args.dropout).to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)

    best_f1, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, opt, loss_fn, device)
        vl, va, vf1 = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:02d} | train_loss {tl:.4f} acc {ta:.3f} | val_loss {vl:.4f} acc {va:.3f} f1 {vf1:.3f}")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = model.state_dict().copy()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": classes,
        "features": FEATURES
    }, args.save_path)

    # Final test results
    model.load_state_dict(best_state)
    tl, ta, tf1 = evaluate(model, test_loader, loss_fn, device)
    print(f"TEST | loss {tl:.4f} acc {ta:.3f} f1 {tf1:.3f}")


if __name__ == "__main__":
    main()

'''# src/models/dynamic_network_optimization/train.py
import argparse, os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score
from .model import DNOPolicyNet
from .utils import load_dataframe, split_scale_encode, FEATURES

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, running = 0, 0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        running += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return running/total, correct/total

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, running = 0, 0, 0.0
    all_y, all_p = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        running += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
        all_y.extend(y.cpu().tolist())
        all_p.extend(preds.cpu().tolist())
    acc = correct/total
    f1  = f1_score(all_y, all_p, average="weighted")
    return running/total, acc, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--save_path", default="models/dynamic_network_optimization/best.pt")
    args = ap.parse_args()

    df = load_dataframe(args.csv)
    train_ds, val_ds, test_ds, scaler, le, classes = split_scale_encode(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNOPolicyNet(input_dim=len(FEATURES), num_classes=len(classes), hidden=args.hidden, dropout=args.dropout).to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    best_val_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} acc {tr_acc:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": le.classes_.tolist(),
        "features": FEATURES
    }, args.save_path)

    # final test report
    model.load_state_dict(torch.load(args.save_path)["state_dict"])
    te_loss, te_acc, te_f1 = evaluate(model, test_loader, loss_fn, device)
    print(f"TEST | loss {te_loss:.4f} acc {te_acc:.3f} f1 {te_f1:.3f}")

if __name__ == "__main__":
    main()

'''

'''import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(units=64, input_shape=input_shape, return_sequences=True))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=output_shape, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train):
    model = build_model(X_train.shape[1:], y_train.shape[1])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return model, history

'''