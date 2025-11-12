# src/main.py

''' 
What the new main pipeline will do

A single main.py will:

Step 1: Load full CSV
→ one raw dataset for all modules.

Step 2: Run Predictive Network Planning
→ predicts traffic_load_next.

Step 3: Run Dynamic Network Optimization
→ predicts the best network action.

Step 4: Run Energy Efficiency Optimization
→ predicts energy usage.

Step 5: Run Network Anomaly Detection
→ flags anomalies.

Step 6: Merge all outputs
→ produce final_results.csv with:

| site_id | timestamp | next_load | dno_action | energy_prediction | anomaly_score | is_anomaly |



Train each model (once):
python -m src.models.predictive_network_planning.train --csv data.csv
python -m src.models.dynamic_network_optimization.train --csv data.csv
python -m src.models.energy_efficiency_optimization.train --csv data.csv
python -m src.models.network_anomaly_detection.train --data_csv data.csv

Then run main pipeline:
python -m src.main --csv data.csv


Output will be generated at:

/final_results/final_results.csv
'''
# src/main.py

import os
import sys
import subprocess
import pandas as pd


def normalize_timestamp(df):
    """Convert timestamps to a unified format across all modules."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    return df


def run_pipeline(data_csv, out_dir="final_results"):
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # 1) Predictive Network Planning
    # --------------------------------------------------------
    print("=== 1) Predictive Network Planning ===")
    pnp_out = f"{out_dir}/pnp.csv"
    subprocess.run([
        sys.executable, "-m", "src.models.predictive_network_planning.predict",
        "--csv", data_csv,
        "--artifacts", "models/predictive_network_planning",
        "--out_csv", pnp_out
    ], check=True)

    # --------------------------------------------------------
    # 2) Dynamic Network Optimization
    # --------------------------------------------------------
    print("=== 2) Dynamic Network Optimization ===")
    dno_out = f"{out_dir}/dno.csv"
    subprocess.run([
        sys.executable, "-m", "src.models.dynamic_network_optimization.predict",
        "--model_path", "models/dynamic_network_optimization/best.pt",
        "--csv", data_csv,
        "--out", dno_out
    ], check=True)

    # --------------------------------------------------------
    # 3) Energy Efficiency Optimization
    # --------------------------------------------------------
    print("=== 3) Energy Efficiency Optimization ===")
    eeo_out = f"{out_dir}/eeo.csv"
    subprocess.run([
        sys.executable, "-m", "src.models.energy_efficiency_optimization.predict",
        "--model_path", "models/energy_efficiency_optimization/best_model.pt",
        "--csv", data_csv,
        "--out", eeo_out
    ], check=True)

    # --------------------------------------------------------
    # 4) Network Anomaly Detection
    # --------------------------------------------------------
    print("=== 4) Network Anomaly Detection ===")
    nad_out = f"{out_dir}/nad.csv"
    subprocess.run([
        sys.executable, "-m", "src.models.network_anomaly_detection.predict",
        "--data_csv", data_csv,
        "--model_dir", "models/network_anomaly_detection",
        "--output_csv", nad_out
    ], check=True)

    # --------------------------------------------------------
    # 5) Load CSV outputs
    # --------------------------------------------------------
    print("=== 5) Loading & Normalizing Outputs ===")

    pnp_df = normalize_timestamp(pd.read_csv(pnp_out))
    dno_df = normalize_timestamp(pd.read_csv(dno_out))
    eeo_df = normalize_timestamp(pd.read_csv(eeo_out))
    nad_df = normalize_timestamp(pd.read_csv(nad_out))

    print("\nRows per module:")
    print("PNP:", len(pnp_df))
    print("DNO:", len(dno_df))
    print("EEO:", len(eeo_df))
    print("NAD:", len(nad_df))

    # Debug: Check for timestamp mismatches
    common_timestamps = (
        set(pnp_df["timestamp"])
        & set(dno_df["timestamp"])
        & set(eeo_df["timestamp"])
        & set(nad_df["timestamp"])
    )

    print("\nCommon timestamps across ALL modules:", len(common_timestamps))

    # --------------------------------------------------------
    # 6) Merge with LEFT JOIN from PNP
    # --------------------------------------------------------
    print("\n=== 6) Merging results ===")

    final_df = (
        pnp_df
        .merge(dno_df, on=["site_id", "timestamp"], how="left")
        .merge(eeo_df, on=["site_id", "timestamp"], how="left")
        .merge(nad_df, on=["site_id", "timestamp"], how="left")
    )

    final_csv = f"{out_dir}/final_results.csv"
    final_df.to_csv(final_csv, index=False)

    print(f"\n✅ Pipeline complete.")
    print(f"✅ Final merged output saved to: {final_csv}")
    print(f"✅ Final rows: {len(final_df)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input dataset CSV")
    parser.add_argument("--out_dir", default="final_results")
    args = parser.parse_args()

    run_pipeline(args.csv, args.out_dir)


'''
import argparse
from datetime import datetime
from utils import config
from utils.logger import Logger
from data_preparation.data_extraction import extract_data
from data_preparation.data_cleaning import clean_data
from data_preparation.data_transformation import transform_data
from models.predictive_network_planning.predict import make_predictions

def main(args):
    # Set up logger
    log_file = f"{config.LOGS_DIR}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logger = Logger(log_file)
    
    # Extract data
    logger.log("Extracting data...")
    raw_data = extract_data(args.data_file)
    
    # Clean data
    logger.log("Cleaning data...")
    clean_data = clean_data(raw_data)
    
    # Transform data
    logger.log("Transforming data...")
    transformed_data = transform_data(clean_data)
    
    # Make predictions
    logger.log("Making predictions...")
    predictions = make_predictions(transformed_data)
    
    # Save predictions to file
    logger.log("Saving predictions to file...")
    predictions_file = f"{config.PREDICTIONS_DIR}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
    predictions.to_csv(predictions_file, index=False)
    
    logger.log("Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main program.")
    parser.add_argument("data_file", type=str, help="Path to the raw data file.")
    args = parser.parse_args()
    
    main(args)

'''