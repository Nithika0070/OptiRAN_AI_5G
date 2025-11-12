'''
import os
import json
import joblib
import torch
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# -----------------------------------------------------
# MODEL DIRECTORIES
# -----------------------------------------------------
PNP_DIR = "models/predictive_network_planning"
DNO_DIR = "models/dynamic_network_optimization"
EEO_DIR = "models/energy_efficiency_optimization"
NAD_DIR = "models/network_anomaly_detection"

# -----------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------
app = FastAPI()

# Serve UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/home", response_class=HTMLResponse)
def home():
    return open("static/index.html").read()

@app.get("/")
def root():
    return {"message": "OptiRan 5G API Running"}


# -----------------------------------------------------
# LOAD 1: Predictive Network Planning
# -----------------------------------------------------
from src.models.predictive_network_planning.model import PlanningMLP

with open(f"{PNP_DIR}/planning_features.json", "r") as f:
    pnp_features = json.load(f)["feature_cols"]

pnp_scaler = joblib.load(f"{PNP_DIR}/planning_scaler.pkl")

pnp_model = PlanningMLP(input_dim=len(pnp_features))
pnp_model.load_state_dict(torch.load(f"{PNP_DIR}/best.pt", map_location="cpu"))
pnp_model.eval()


# -----------------------------------------------------
# LOAD 2: Dynamic Network Optimization
# -----------------------------------------------------
from src.models.dynamic_network_optimization.model import DNOPolicyNet

dno_ckpt = torch.load(f"{DNO_DIR}/best.pt", map_location="cpu")

dno_features = dno_ckpt["features"]
dno_classes = dno_ckpt["classes"]

dno_mean = torch.tensor(dno_ckpt["scaler_mean"])
dno_scale = torch.tensor(dno_ckpt["scaler_scale"])

dno_model = DNOPolicyNet(
    input_dim=len(dno_features),
    num_classes=len(dno_classes)
)
dno_model.load_state_dict(dno_ckpt["state_dict"])
dno_model.eval()


# -----------------------------------------------------
# LOAD 3: Energy Efficiency Optimization
# -----------------------------------------------------
from src.models.energy_efficiency_optimization.model import EnergyEfficiencyNet

eeo_ckpt = torch.load(f"{EEO_DIR}/best_model.pt", map_location="cpu")

eeo_features = eeo_ckpt["features"]
eeo_mean = torch.tensor(eeo_ckpt["scaler_mean"])
eeo_scale = torch.tensor(eeo_ckpt["scaler_scale"])

eeo_model = EnergyEfficiencyNet(input_dim=len(eeo_features))
eeo_model.load_state_dict(eeo_ckpt["state_dict"])
eeo_model.eval()


# -----------------------------------------------------
# LOAD 4: Network Anomaly Detection (Autoencoder)
# -----------------------------------------------------
from src.models.network_anomaly_detection.model import AE, reconstruction_error

nad_ckpt = torch.load(f"{NAD_DIR}/model.pt", map_location="cpu")

nad_input_dim = len(nad_ckpt["scaler_mean"])        # FIX
nad_mean = torch.tensor(nad_ckpt["scaler_mean"])
nad_scale = torch.tensor(nad_ckpt["scaler_scale"])
nad_threshold = nad_ckpt["threshold"]

nad_model = AE(input_dim=nad_input_dim)
nad_model.load_state_dict(nad_ckpt["state_dict"])
nad_model.eval()


# -----------------------------------------------------
# PREDICTION ENDPOINT
# -----------------------------------------------------
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # ===============================
        # ✅ Feature Engineering
        # ===============================
        if "traffic_load" in df.columns:
            df["traffic_load_lag1"] = df["traffic_load"].shift(1)
            df["traffic_load_lag2"] = df["traffic_load"].shift(2)
            df["traffic_load_lag3"] = df["traffic_load"].shift(3)
            df["traffic_load_roll3"] = df["traffic_load"].rolling(window=3, min_periods=1).mean()
            df["traffic_load_roll6"] = df["traffic_load"].rolling(window=6, min_periods=1).mean()

        df = df.fillna(method="bfill").fillna(0)

        # ===============================
        # ✅ Feature Validation
        # ===============================
        missing = [c for c in pnp_features if c not in df.columns]
        if missing:
            return {"error": f"Missing columns in uploaded CSV: {missing}"}

        # ===============================
        # PNP
        # ===============================
        X_pnp = df[pnp_features].values
        X_pnp_s = pnp_scaler.transform(X_pnp)
        pnp_pred = (
            pnp_model(torch.tensor(X_pnp_s, dtype=torch.float32))
            .detach()
            .numpy()
            .flatten()
        )
        df["traffic_load_next_pred"] = pnp_pred

        # ===============================
        # DNO
        # ===============================
        X_dno = df[dno_features].copy()
        # Ensure numeric (convert strings to NaN)
        X_dno = X_dno.apply(pd.to_numeric, errors="coerce")
        # Replace NaN or inf with 0
        X_dno = X_dno.fillna(0).replace([np.inf, -np.inf], 0)
        # Scale
        X_dno_s = (X_dno.values - dno_mean.numpy()) / dno_scale.numpy()
        # Convert to tensor
        logits = dno_model(torch.tensor(X_dno_s, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).detach().numpy()
        df["dno_action"] = [dno_classes[i] for i in probs.argmax(axis=1)]
        df["dno_confidence"] = probs.max(axis=1)

        # ===============================
        # EEO
        # ===============================
        X_eeo = df[eeo_features].values
        X_eeo_s = (X_eeo - eeo_mean.numpy()) / eeo_scale.numpy()
        eeo_pred = (
            eeo_model(torch.tensor(X_eeo_s, dtype=torch.float32))
            .detach()
            .numpy()
            .flatten()
        )
        df["energy_pred_kW"] = eeo_pred

        # ===============================
        # NAD
        # ===============================
        X_nad = df.select_dtypes(include="number").values[:, :nad_input_dim]
        X_nad_s = (X_nad - nad_mean.numpy()) / nad_scale.numpy()
        X_t = torch.tensor(X_nad_s, dtype=torch.float32)
        recon = nad_model(X_t).detach()
        errors = reconstruction_error(X_t, recon).numpy()
        df["anomaly_score"] = errors
        df["is_anomaly"] = errors > nad_threshold

        # ✅ Return dataframe as JSON
        return df.to_dict(orient="records")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("🔥 Internal Error:\n", error_details)
        return {"error": str(e), "traceback": error_details}
'''
# src/api.py

import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# -----------------------------------------------------
# MODEL DIRECTORIES
# -----------------------------------------------------
PNP_DIR = "models/predictive_network_planning"
DNO_DIR = "models/dynamic_network_optimization"
EEO_DIR = "models/energy_efficiency_optimization"
NAD_DIR = "models/network_anomaly_detection"

# -----------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------
app = FastAPI(title="OptiRan 5G")

# Serve static frontend (optional)
'''app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    html = """
    <html>
        <head><title>OptiRan 5G</title></head>
        <body style="font-family:Arial;text-align:center;padding:40px;">
            <h2>🚀 OptiRan 5G</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <p><input type="file" name="file" accept=".csv"></p>
                <p><input type="submit" value="Run Optimization"></p>
            </form>
            <p><a href="/docs">Or use Swagger UI</a></p>
        </body>
    </html>
    """
    return HTMLResponse(content=html)
'''
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    """
    Serve the main frontend page for the AI-Powered 5G OpenRAN Optimizer.
    This page provides a file upload form and displays results via JavaScript.
    """
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(
            content="<h3 style='color:red;'>Error: static/index.html not found.</h3>"
        )
# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------

# 1️⃣ Predictive Network Planning
from src.models.predictive_network_planning.model import PlanningMLP

with open(f"{PNP_DIR}/planning_features.json", "r") as f:
    pnp_features = json.load(f)["feature_cols"]

pnp_scaler = joblib.load(f"{PNP_DIR}/planning_scaler.pkl")

pnp_model = PlanningMLP(input_dim=len(pnp_features))
pnp_model.load_state_dict(torch.load(f"{PNP_DIR}/best.pt", map_location="cpu"))
pnp_model.eval()


# 2️⃣ Dynamic Network Optimization
from src.models.dynamic_network_optimization.model import DNOPolicyNet

dno_ckpt = torch.load(f"{DNO_DIR}/best.pt", map_location="cpu")
dno_features = dno_ckpt["features"]
dno_classes = dno_ckpt["classes"]
dno_mean = torch.tensor(dno_ckpt["scaler_mean"])
dno_scale = torch.tensor(dno_ckpt["scaler_scale"])

dno_model = DNOPolicyNet(input_dim=len(dno_features), num_classes=len(dno_classes))
dno_model.load_state_dict(dno_ckpt["state_dict"])
dno_model.eval()


# 3️⃣ Energy Efficiency Optimization
from src.models.energy_efficiency_optimization.model import EnergyEfficiencyNet

eeo_ckpt = torch.load(f"{EEO_DIR}/best_model.pt", map_location="cpu")
eeo_features = eeo_ckpt["features"]
eeo_mean = torch.tensor(eeo_ckpt["scaler_mean"])
eeo_scale = torch.tensor(eeo_ckpt["scaler_scale"])

eeo_model = EnergyEfficiencyNet(input_dim=len(eeo_features))
eeo_model.load_state_dict(eeo_ckpt["state_dict"])
eeo_model.eval()


# 4️⃣ Network Anomaly Detection
from src.models.network_anomaly_detection.model import AE, reconstruction_error

nad_ckpt = torch.load(f"{NAD_DIR}/model.pt", map_location="cpu")
nad_input_dim = len(nad_ckpt["scaler_mean"])
nad_mean = torch.tensor(nad_ckpt["scaler_mean"])
nad_scale = torch.tensor(nad_ckpt["scaler_scale"])
nad_threshold = nad_ckpt["threshold"]

nad_model = AE(input_dim=nad_input_dim)
nad_model.load_state_dict(nad_ckpt["state_dict"])
nad_model.eval()


# -----------------------------------------------------
# HELPER FUNCTION: Normalize timestamps
# -----------------------------------------------------
def normalize_timestamp(df):
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    return df


# -----------------------------------------------------
# MAIN ENDPOINT: /predict
# -----------------------------------------------------
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    import traceback
    try:
        df = pd.read_csv(file.file)
        df = df.fillna(method="bfill").fillna(0)

        # ===============================
        # 1️⃣ Feature Engineering
        # ===============================
        if "traffic_load" in df.columns:
            df["traffic_load_lag1"] = df["traffic_load"].shift(1)
            df["traffic_load_lag2"] = df["traffic_load"].shift(2)
            df["traffic_load_lag3"] = df["traffic_load"].shift(3)
            df["traffic_load_roll3"] = df["traffic_load"].rolling(window=3, min_periods=1).mean()
            df["traffic_load_roll6"] = df["traffic_load"].rolling(window=6, min_periods=1).mean()
        df = df.fillna(0)

        # ===============================
        # Utility cleaner for all models
        # ===============================
        def clean_numeric(df_in, feature_list):
            """Force numeric conversion for selected model features."""
            temp = df_in[feature_list].copy()
            temp = temp.apply(pd.to_numeric, errors="coerce")
            temp = temp.replace([np.inf, -np.inf], np.nan).fillna(0)
            return temp.astype(np.float32)

        # ===============================
        # 2️⃣ Predictive Network Planning
        # ===============================
        X_pnp = clean_numeric(df, pnp_features)
        X_pnp_s = pnp_scaler.transform(X_pnp)
        pnp_pred = (
            pnp_model(torch.tensor(X_pnp_s, dtype=torch.float32))
            .detach()
            .numpy()
            .flatten()
        )
        df["traffic_load_next_pred"] = pnp_pred

        # ===============================
        # 3️⃣ Dynamic Network Optimization
        # ===============================
        X_dno = clean_numeric(df, dno_features)
        X_dno_s = (X_dno.values - dno_mean.numpy()) / dno_scale.numpy()
        logits = dno_model(torch.tensor(X_dno_s, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).detach().numpy()
        df["dno_action"] = [dno_classes[i] for i in probs.argmax(axis=1)]
        df["dno_confidence"] = probs.max(axis=1)

        # ===============================
        # 4️⃣ Energy Efficiency Optimization
        # ===============================
        X_eeo = clean_numeric(df, eeo_features)
        X_eeo_s = (X_eeo.values - eeo_mean.numpy()) / eeo_scale.numpy()
        eeo_pred = (
            eeo_model(torch.tensor(X_eeo_s, dtype=torch.float32))
            .detach()
            .numpy()
            .flatten()
        )
        df["energy_pred_kW"] = eeo_pred

        # ===============================
        # 5️⃣ Network Anomaly Detection
        # ===============================
        X_nad = df.select_dtypes(include="number").values[:, :nad_input_dim]
        X_nad_s = (X_nad - nad_mean.numpy()) / nad_scale.numpy()
        X_t = torch.tensor(X_nad_s, dtype=torch.float32)
        recon = nad_model(X_t).detach()
        errors = reconstruction_error(X_t, recon).numpy()
        df["anomaly_score"] = errors
        df["is_anomaly"] = errors > nad_threshold

        # ===============================
        # 6️⃣ Save Output
        # ===============================
        os.makedirs("final_results", exist_ok=True)
        out_path = "final_results/final_results.csv"
        df.to_csv(out_path, index=False)

        # ===============================
        # 7️⃣ Return Summary
        # ===============================
        sample = df.head(5).to_dict(orient="records")
        return {
            "message": "✅ Prediction pipeline executed successfully.",
            "rows_processed": len(df),
            "output_file": out_path,
            "columns_generated": [
                "traffic_load_next_pred", "dno_action", "dno_confidence",
                "energy_pred_kW", "anomaly_score", "is_anomaly"
            ],
            "sample_output": sample
        }

    except Exception as e:
        tb = traceback.format_exc()
        print("🔥 Internal Server Error:\n", tb)
        return {"error": str(e), "traceback": tb}

'''
uvicorn src.api:app --reload    

git bash: curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/network_ran_dataset.csv"

powershell: Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -Form @{file = Get-Item "data/network_ran_dataset.csv"} `
  -ContentType "multipart/form-data"


'''