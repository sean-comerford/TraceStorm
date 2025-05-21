import unittest
from datetime import datetime
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

from tracestorm.constants import AZURE_DATASET_PATHS
from tracestorm.trace_generator import (
    AzureTraceGenerator,
    SyntheticTraceGenerator,
)


class TestSyntheticTraceGenerator(unittest.TestCase):
    def test_uniform_distribution(self):
        """Test uniform distribution pattern."""
        generator = SyntheticTraceGenerator(
            rps=2, pattern="uniform", duration=3
        )
        result = generator.generate()
        expected = [0, 500, 1000, 1500, 2000, 2500]
        self.assertEqual(result, expected)

    def test_uniform_distribution_float_rps(self):
        """Test uniform distribution pattern with float RPS value."""
        generator = SyntheticTraceGenerator(
            rps=1.5, pattern="uniform", duration=4
        )
        # Let's get the actual result and use direct value comparison
        result = generator.generate()
        # 1.5 RPS for 4 seconds = 6 requests
        self.assertEqual(len(result), 6)
        # First and last timestamps should be consistent
        self.assertEqual(result[0], 0)
        self.assertTrue(result[-1] < 4000)  # Should be less than duration in ms

    def test_invalid_rps(self):
        """Test invalid RPS value."""
        with self.assertRaises(ValueError) as context:
            SyntheticTraceGenerator(rps=-1, pattern="uniform", duration=10)
        self.assertEqual(
            str(context.exception), "rps must be a non-negative number"
        )

    def test_invalid_rps_float(self):
        """Test invalid RPS float value."""
        with self.assertRaises(ValueError) as context:
            SyntheticTraceGenerator(rps=-0.5, pattern="uniform", duration=10)
        self.assertEqual(
            str(context.exception), "rps must be a non-negative number"
        )

    def test_valid_float_rps(self):
        """Test valid float RPS value."""
        generator = SyntheticTraceGenerator(
            rps=0.5, pattern="uniform", duration=10
        )
        result = generator.generate()
        # 0.5 RPS for 10 seconds = 5 total requests
        self.assertEqual(len(result), 5)

    def test_invalid_duration(self):
        """Test invalid duration value."""
        with self.assertRaises(ValueError) as context:
            SyntheticTraceGenerator(rps=1, pattern="uniform", duration=-5)
        self.assertEqual(
            str(context.exception), "duration must be a non-negative integer"
        )

    def test_zero_requests(self):
        """Test zero requests case."""
        generator = SyntheticTraceGenerator(
            rps=0, pattern="uniform", duration=10
        )
        result = generator.generate()
        self.assertEqual(result, [])

    def test_invalid_pattern(self):
        """Test invalid pattern."""
        generator = SyntheticTraceGenerator(
            rps=1, pattern="invalid", duration=10
        )
        with self.assertRaises(ValueError) as context:
            generator.generate()
        self.assertEqual(str(context.exception), "Unknown pattern: invalid")


class TestAzureTraceGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = AzureTraceGenerator(dataset_type="code")

    def test_invalid_dataset_type(self):
        """Test initialization with invalid dataset type."""
        with self.assertRaises(ValueError) as context:
            AzureTraceGenerator(dataset_type="invalid")
        self.assertEqual(
            str(context.exception),
            "Invalid dataset type. Please choose 'code' or 'conv'.",
        )

    @patch("tracestorm.trace_generator.requests.get")
    def test_download_file_success(self, mock_requests):
        """Test successful file download."""
        mock_requests.return_value.status_code = 200
        mock_requests.return_value.content = b"Test content"

        with patch("builtins.open", mock_open()) as mock_file:
            self.generator._download_file("test/path.csv", "local.csv")

        mock_requests.assert_called_once()
        mock_file.assert_called_once_with("local.csv", "wb")
        mock_file().write.assert_called_once_with(b"Test content")

    @patch("tracestorm.trace_generator.requests.get")
    def test_download_file_failure(self, mock_requests):
        """Test file download failure."""
        mock_requests.return_value.status_code = 404
        mock_requests.return_value.text = "Not Found"

        with self.assertRaises(Exception) as context:
            self.generator._download_file("test/path.csv", "local.csv")
        self.assertIn("Failed to download file: 404", str(context.exception))

    def test_process_dataset(self):
        """Test dataset processing."""
        # Create test DataFrame
        dates = [
            datetime(2023, 1, 1, 12, 0, 0),
            datetime(2023, 1, 1, 12, 0, 1),
            datetime(2023, 1, 1, 12, 0, 2),
        ]
        test_df = pd.DataFrame({"TIMESTAMP": dates})

        with patch("pandas.read_csv", return_value=test_df):
            result = self.generator._process_dataset("dummy_path.csv")

        expected = [0, 1000, 2000]  # timestamps in milliseconds
        self.assertEqual(result, expected)

    @patch("tracestorm.trace_generator.AzureTraceGenerator._download_file")
    @patch("tracestorm.trace_generator.AzureTraceGenerator._process_dataset")
    def test_generate(self, mock_process, mock_download):
        """Test the generate method."""
        mock_process.return_value = [0, 1000, 2000]

        result = self.generator.generate()

        self.assertEqual(result, [0, 1000, 2000])
        mock_download.assert_called_once()
        mock_process.assert_called_once()


if __name__ == "__main__":
    unittest.main()
