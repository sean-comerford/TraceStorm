import unittest
from unittest import TestCase, mock
from unittest.mock import mock_open, patch

import pandas as pd

from tracestorm.constants import AZURE_DATASET_PATHS
from tracestorm.trace_loader import (
    download_file_from_github,
    load_azure_inference_dataset,
    process_dataset,
)


class TestTraceLoader(TestCase):
    @patch("tracestorm.trace_loader.requests.get")
    @patch("builtins.open", new_callable=mock_open)
    @patch("tracestorm.trace_loader.logger")
    def test_download_file_from_github_success(
        self, mock_logger, mock_open, mock_requests
    ):
        """Test successful file download from GitHub."""
        mock_requests.return_value.status_code = 200
        mock_requests.return_value.content = b"Test content"

        download_file_from_github("path/to/file.csv", "local_file.csv")

        mock_requests.assert_called_once()
        mock_open.assert_called_once_with("local_file.csv", "wb")
        mock_open().write.assert_called_once_with(b"Test content")
        mock_logger.info.assert_called_once_with(
            "File downloaded successfully as 'local_file.csv'"
        )

    @patch("tracestorm.trace_loader.requests.get")
    @patch("tracestorm.trace_loader.logger")
    def test_download_file_from_github_failure(
        self, mock_logger, mock_requests
    ):
        """Test failed file download from GitHub."""
        mock_requests.return_value.status_code = 404
        mock_requests.return_value.text = "Not Found"

        with self.assertRaises(Exception):
            download_file_from_github("path/to/file.csv", "local_file.csv")

        mock_logger.error.assert_called_once_with(
            "Failed to download file: 404 - Not Found"
        )

    @patch("tracestorm.trace_loader.download_file_from_github")
    @patch("tracestorm.trace_loader.process_dataset")
    @patch("tracestorm.trace_loader.logger")
    def test_load_azure_inference_dataset(
        self, mock_logger, mock_process, mock_download
    ):
        """Test loading the Azure inference dataset."""
        mock_process.return_value = [0, 1000, 2000]

        AZURE_DATASET_PATHS["code"] = "path/to/code.csv"
        timestamps = load_azure_inference_dataset("code")

        self.assertEqual(timestamps, [0, 1000, 2000])
        mock_download.assert_called_once_with("path/to/code.csv", mock.ANY)
        mock_process.assert_called_once()

    @patch("pandas.read_csv")
    @patch("tracestorm.trace_loader.logger")
    def test_process_dataset(self, mock_logger, mock_read_csv):
        """Test processing the dataset."""
        # Create test data with timestamp strings
        mock_read_csv.return_value = pd.DataFrame(
            {
                "TIMESTAMP": [
                    "2023-11-16 19:14:14.100000",
                    "2023-11-16 19:14:14.200000",
                    "2023-11-16 19:14:14.300000",
                ]
            }
        )

        result = process_dataset("dummy_path.csv")

        # Verify relative timestamps starting from 0
        expected = [
            0,
            100,
            200,
        ]  # Relative milliseconds from the first timestamp
        self.assertEqual(result, expected)
        mock_logger.info.assert_called_once_with(
            "Processed dataset and extracted 3 relative timestamps."
        )


if __name__ == "__main__":
    unittest.main()
