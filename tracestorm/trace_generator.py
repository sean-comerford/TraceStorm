import os
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from tracestorm.constants import AZURE_DATASET_PATHS, AZURE_REPO_URL
from tracestorm.logger import init_logger

logger = init_logger("trace_generator")


class TraceGenerator(ABC):
    """Abstract base class for trace generation."""

    @abstractmethod
    def generate(self) -> List[int]:
        """
        Generate a list of timestamps (ms) at which requests should be sent.

        Returns:
            List[int]: Sorted list of timestamps in milliseconds.

        Raises:
            ValueError: If inputs are invalid or generation fails.
        """
        pass


class SyntheticTraceGenerator(TraceGenerator):
    """Generate synthetic traces based on patterns."""

    def __init__(
        self, rps: int, pattern: str, duration: int, seed: Optional[int] = None
    ):
        """
        Initialize synthetic trace generator.

        Args:
            rps (int): Requests per second. Must be non-negative.
            pattern (str): Distribution pattern ('uniform', 'random', 'poisson', etc.).
            duration (int): Total duration in seconds. Must be non-negative.
            seed (int): Seed for reproducibility of 'poisson' and 'random' patterns
        """
        if not isinstance(rps, int) or rps < 0:
            raise ValueError("rps must be a non-negative integer")
        if not isinstance(duration, int) or duration < 0:
            raise ValueError("duration must be a non-negative integer")

        self.rps = rps
        self.pattern = pattern
        self.duration = duration
        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> List[int]:
        total_requests = self.rps * self.duration
        total_duration_ms = self.duration * 1000
        timestamps = []

        if total_requests == 0:
            return timestamps

        if self.pattern == "uniform":
            # Distribute requests evenly across the duration
            interval = total_duration_ms / total_requests
            current_time = 0.0
            for _ in range(total_requests):
                timestamp = int(round(current_time))
                timestamp = min(timestamp, total_duration_ms - 1)
                timestamps.append(timestamp)
                current_time += interval
        elif self.pattern == "poisson":
            # Exponential distribution for intervals
            rate_ms = self.rps / 1000
            intervals = np.random.exponential(1 / rate_ms, total_requests)
            current_time = 0.0
            for i in range(total_requests):
                timestamp = int(round(current_time))
                timestamps.append(timestamp)
                current_time += intervals[i]
        elif self.pattern == "random":
            timestamps = np.random.randint(
                0, total_duration_ms, size=total_requests
            ).tolist()
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

        return sorted(timestamps)


class AzureTraceGenerator(TraceGenerator):
    """Generate traces from Azure Function datasets."""

    def __init__(self, dataset_type: str):
        """
        Initialize Azure trace generator.

        Args:
            dataset_type (str): Type of dataset to load ('code' or 'conv').
        """
        if dataset_type not in AZURE_DATASET_PATHS:
            raise ValueError(
                "Invalid dataset type. Please choose 'code' or 'conv'."
            )

        self.dataset_type = dataset_type

    def generate(self) -> List[int]:
        """Generate timestamp trace from Azure dataset."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = AZURE_DATASET_PATHS[self.dataset_type]
            local_file = os.path.join(tmpdirname, os.path.basename(file_path))

            try:
                self._download_file(file_path, local_file)
                return self._process_dataset(local_file)
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                raise

    def _download_file(self, file_path: str, save_as: str) -> None:
        """
        Download a file from Azure GitHub repository.

        Args:
            file_path (str): The path to the file in the GitHub repository.
            save_as (str): The local filename to save the downloaded file as.

        Raises:
            Exception: If the file download fails.
        """
        raw_url = f"https://raw.githubusercontent.com/{AZURE_REPO_URL}/master/{file_path}"
        response = requests.get(raw_url)

        if response.status_code == 200:
            with open(save_as, "wb") as file:
                file.write(response.content)
            logger.info(f"File downloaded successfully as '{save_as}'")
        else:
            error_msg = f"Failed to download file: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _process_dataset(self, file_path: str) -> List[int]:
        """
        Process the downloaded Azure dataset.

        Args:
            file_path (str): The path to the downloaded dataset file.

        Returns:
            List[int]: A sorted list of relative timestamps in milliseconds.

        Raises:
            Exception: If there is an error processing the dataset.
        """
        try:
            df = pd.read_csv(file_path)

            # Convert 'TIMESTAMP' column to datetime objects
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

            # Find the earliest timestamp
            earliest_timestamp = df["TIMESTAMP"].min()

            # Calculate the difference in milliseconds from the earliest timestamp
            df["RELATIVE_TIMESTAMP_MS"] = (
                df["TIMESTAMP"] - earliest_timestamp
            ).dt.total_seconds() * 1000

            # Convert to integers and ensure they start from 0
            timestamps = df["RELATIVE_TIMESTAMP_MS"].astype(int).tolist()

            logger.info(
                f"Processed dataset and extracted {len(timestamps)} relative timestamps."
            )
            return sorted(timestamps)
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise
