⚡ OptiRAN AI 5G

AI-Powered OpenRAN Optimization Suite

A system powered by artificial intelligence that forecasts, enhances, and oversees 5G OpenRAN networks in real time.

Overview

OptiRAN AI 5G is a modular AI system designed to optimize 5G Open Radio Access Networks (OpenRAN) by combining predictive analytics, anomaly detection, and energy-efficient network management.
It automates critical network decisions such as traffic load prediction, energy optimization, and dynamic configuration — all driven by AI models.

This project demonstrates a data-to-decision AI pipeline, showcasing how machine learning can improve telecom network reliability, energy efficiency, and user experience.

Core Modules
🔹 1. Predictive Network Planning (PNP)

Goal: Forecast future network traffic load.
How it works:
Takes site-level KPIs like user count, throughput, and signal quality.
Predicts traffic_load_next using a neural network trained on historical data.
Helps operators anticipate demand and allocate resources proactively.

🔹 2. Dynamic Network Optimization (DNO)

Goal: Recommend optimal network actions dynamically.
How it works:
Classifies the best optimization_action (e.g., increase_capacity, no_change, reduce_power) based on network state.
Uses reinforcement learning-inspired policy networks.
Balances network throughput and stability while avoiding overload.

🔹 3. Energy Efficiency Optimization (EEO)

Goal: Minimize power consumption without affecting performance.
How it works:
Predicts energy consumption (kW) for given network conditions.
Suggests sleep modes or capacity adjustments.
Encourages greener, more sustainable network operation.

🔹 4. Network Anomaly Detection (NAD)

Goal: Detect unusual patterns or faults in real-time KPIs.
How it works:
Uses an autoencoder to reconstruct normal behavior from KPIs.
High reconstruction error → potential anomaly.
Flags anomalies for operator intervention.

📊 Dataset

Input CSV: data/network_ran_dataset.csv
Columns:

Feature	Description
site_id	Unique identifier for cell tower/site
timestamp	Date & time of record
traffic_load	Total current traffic load
num_users	Number of active users
average_throughput	Average throughput per user
peak_throughput	Peak throughput observed
signal_strength_dBm	Signal power received (dBm)
interference_level_dBm	Noise/interference level (dBm)
energy_consumption_kW	Power consumed by the site
sleep_mode_enabled	Whether energy-saving mode is active
handover_failures	Count of failed user handovers
latency_ms	Average network latency
optimization_action	Network’s actual applied action

🧰 Tech Stack

Language: Python
Framework: FastAPI
ML Libraries: PyTorch, scikit-learn, joblib, pandas
Serving: Uvicorn (local FastAPI server)
Interface: Static HTML frontend + Swagger UI (/docs)

⚙️ Installation & Setup
# 1️⃣ Clone the repository
git clone https://github.com/Nithika0070/OptiRAN_AI_5G.git
cd OptiRAN_AI_5G
# 2️⃣ Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # (Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run API
uvicorn src.api:app --reload

🖥️ Usage
🌐 Access the Web Interface

Open your browser and visit:
👉 http://127.0.0.1:8000

Upload your network_ran_dataset.csv file to run optimization.
Swagger UI (API Playground)

Available at:
👉 http://127.0.0.1:8000/docs

🧮 Example cURL Command
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/network_ran_dataset.csv"

📦 Output

Output file:
final_results/final_results.csv

Sample Columns:

Column	Description
traffic_load_next_pred	Forecasted network load
dno_action	AI-suggested optimization action
dno_confidence	Confidence level of decision
energy_pred_kW	Predicted energy consumption
anomaly_score	Anomaly likelihood
is_anomaly	Boolean flag for anomaly detection
🧱 Architecture Overview
               ┌─────────────────────────────┐
               │   Data Preprocessing Layer  │
               │ (Feature Engineering, Scaling) │
               └─────────────┬───────────────┘
                             │
       ┌─────────────────────┼──────────────────────┐
       │                     │                      │
┌──────────────┐     ┌─────────────────┐     ┌───────────────────────┐
│   PNP Model  │     │  DNO Model      │     │  EEO Model            │
│ (Traffic Pred)│    │ (Policy Actions)│     │ (Energy Optimization) │
└──────────────┘     └─────────────────┘     └───────────────────────┘
                             │
                        ┌────┴────┐
                        │  NAD AE │   ← Detect anomalies
                        └────┬────┘
                             │
                    ┌────────┴────────┐
                    │ Final Results CSV│
                    └──────────────────┘

🏷️ Acknowledgment
This project was inspired by the original AI-Powered 5G OpenRAN Optimizer repository.
Significant modifications and improvements were made for educational and portfolio purposes, maintaining open-source licensing integrity.