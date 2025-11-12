# tests/test_data_preparation.py

import pandas as pd
import pytest

def test_data_integrity():
    """Check if dataset has all expected columns."""
    expected_columns = [
        "site_id", "timestamp", "traffic_load", "num_users",
        "average_throughput", "peak_throughput", "signal_strength_dBm",
        "interference_level_dBm", "energy_consumption_kW",
        "sleep_mode_enabled", "handover_failures", "latency_ms", "optimization_action"
    ]

    df = pd.read_csv("data/network_ran_dataset.csv")
    missing = [col for col in expected_columns if col not in df.columns]

    assert not missing, f"Missing columns in dataset: {missing}"
    assert len(df) > 0, "Dataset should not be empty"

'''import unittest
import pandas as pd
from src.utils.data_preparation.data_cleaning import clean_data
from src.utils.data_preparation.data_extraction import extract_data
from src.utils.data_preparation.data_transformation import transform_data


class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.raw_data = pd.read_csv("tests/test_data/raw_data.csv")

    def test_clean_data(self):
        # Test data cleaning function
        cleaned_data = clean_data(self.raw_data)
        self.assertIsInstance(cleaned_data, pd.DataFrame)
        self.assertEqual(len(cleaned_data), 4)
        self.assertEqual(cleaned_data.isna().sum().sum(), 0)

    def test_extract_data(self):
        # Test data extraction function
        extracted_data = extract_data(self.raw_data)
        self.assertIsInstance(extracted_data, pd.DataFrame)
        self.assertEqual(len(extracted_data), 4)
        self.assertEqual(len(extracted_data.columns), 3)

    def test_transform_data(self):
        # Test data transformation function
        transformed_data = transform_data(self.raw_data)
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertEqual(len(transformed_data), 4)
        self.assertEqual(len(transformed_data.columns), 2)

if __name__ == '__main__':
    unittest.main()

'''