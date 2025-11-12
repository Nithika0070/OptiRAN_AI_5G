# src/models/network_anomaly_detection/model.py
import torch
import torch.nn as nn

class AE(nn.Module):
    """
    Simple fully-connected Autoencoder for tabular RAN KPIs.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, bottleneck: int = 16, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def reconstruction_error(x, x_hat, reduction="none"):
    # per-sample MSE
    mse = (x - x_hat).pow(2).mean(dim=1)
    if reduction == "mean":
        return mse.mean()
    return mse



'''import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class NetworkAnomalyDetectionModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._create_model()

    def _create_model(self):
        model = Sequential([
            Dense(256, activation='relu', input_shape=self.input_shape),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, x_val, y_val, batch_size, epochs):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

    def predict(self, x):
        return self.model.predict(x)
'''