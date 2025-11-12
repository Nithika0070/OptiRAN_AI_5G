# tests/test_network_anomaly_detection.py

import torch
import pytest
from src.models.network_anomaly_detection.model import AE, reconstruction_error

def test_nad_autoencoder():
    """Test that Autoencoder encodes and decodes correctly."""
    input_dim = 12
    model = AE(input_dim=input_dim)
    x = torch.randn(5, input_dim)

    x_hat = model(x)
    assert x_hat.shape == x.shape, "Reconstructed output shape should match input"

    # Reconstruction error check
    error = reconstruction_error(x, x_hat)
    assert error.shape == (5,), "Reconstruction error should have 1 value per sample"
    assert (error >= 0).all(), "Reconstruction error should be non-negative"


'''import unittest
from unittest.mock import Mock, patch
from src.utils.logger import Logger
from src.utils.data_loader import DataLoader
from src.utils.data_preprocessing import DataPreprocessor
from src.models.network_anomaly_detection import NetworkAnomalyDetection


class TestNetworkAnomalyDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = Logger()
        cls.logger.disable_logging()
        cls.data_loader = DataLoader()
        cls.preprocessor = DataPreprocessor()
        cls.nad_model = NetworkAnomalyDetection()

    @patch('src.models.network_anomaly_detection.NetworkAnomalyDetection.detect_anomaly')
    def test_detect_anomaly(self, mock_detect_anomaly):
        # Define test data
        test_data = self.data_loader.load_data('test_data.csv')
        preprocessed_data = self.preprocessor.preprocess_data(test_data)
        test_features = preprocessed_data.drop(columns=['timestamp'])
        
        # Mock the predict method and return a dummy prediction
        mock_detect_anomaly.return_value = [0, 0, 1, 1, 0]
        
        # Test the predict method
        predictions = self.nad_model.detect_anomaly(test_features)
        self.assertEqual(len(predictions), len(test_data))
        self.assertListEqual(predictions, mock_detect_anomaly.return_value)

if __name__ == '__main__':
    unittest.main()

'''