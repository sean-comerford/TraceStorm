import os
import tempfile
from abc import ABC, abstractmethod
from typing import List, Optional, Union

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
        self,
        rps: Union[int, float],
        pattern: str,
        duration: int,
        seed: Optional[int] = None,
    ):
        """
        Initialize synthetic trace generator.

        Args:
            rps (Union[int, float]): Requests per second. Must be non-negative.
            pattern (str): Distribution pattern ('uniform', 'random', 'poisson', etc.).
            duration (int): Total duration in seconds. Must be non-negative.
            seed (int): Seed for reproducibility of 'poisson' and 'random' patterns
        """
        if not isinstance(rps, (int, float)) or rps < 0:
            raise ValueError("rps must be a non-negative number")
        if not isinstance(duration, int) or duration < 0:
            raise ValueError("duration must be a non-negative integer")

        self.rps = rps
        self.pattern = pattern
        self.duration = duration
        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> List[int]:
        total_requests = int(round(self.rps * self.duration))
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

    def __init__(self, dataset_type: str, target_rps: Optional[Union[int, float]] = None, window_duration: Optional[int] = None, seed: Optional[int] = None):
        """
        Initialize Azure trace generator.

        Args:
            dataset_type (str): Type of dataset to load ('code' or 'conv').
        """
        if dataset_type not in AZURE_DATASET_PATHS:
            raise ValueError(
                "Invalid dataset type. Please choose 'code' or 'conv'."
            )

        if target_rps is not None:
            if not isinstance(target_rps, (int, float)) or target_rps <= 0:
                raise ValueError("target_rps must be a positive number")
        
        if window_duration is not None and window_duration <= 0:
            raise ValueError("window_duration must be a positive integer")

        self.dataset_type = dataset_type
        self.target_rps: Optional[float] = float(target_rps) if target_rps else None
        self.window_duration_s = window_duration
        self.seed = seed
        self._window_start_ms: Optional[int] = None

    def generate(self) -> List[int]:
        """Generate timestamp trace from Azure dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = AZURE_DATASET_PATHS[self.dataset_type]
            local_file = os.path.join(tmpdir, os.path.basename(file_path))

            self._download_file(file_path, local_file)          # may raise
            ts: List[int] = self._process_dataset(local_file)          # native trace
            

        if self.target_rps is not None:
            ts = self._scale_to_target_rps(ts)

        if self.window_duration_s is not None:
            ts = self._select_random_window(ts)

        self.timestamps: List[int] = ts
        return ts

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
    
    def _scale_to_target_rps(self, ts: List[int]) -> List[int]:
        """Return a new timestamp list with avg RPS ≈ self.target_rps."""
        orig_rps = self._calc_rps(ts)
        scale = orig_rps / self.target_rps           # > 1 ⇒ stretch, < 1 ⇒ squeeze

        if abs(scale - 1.0) < 1e-6:                  # already matches
            logger.info("Target RPS matches original; no scaling applied.")
            return ts

        scaled = [int(round(t * scale)) for t in ts]

        # Ensure *strictly* increasing to avoid zero-interval duplicates
        for i in range(1, len(scaled)):
            if scaled[i] <= scaled[i - 1]:
                scaled[i] = scaled[i - 1] + 1

        new_rps = self._calc_rps(scaled)
        logger.info(
            f"Scaled trace: orig RPS {orig_rps:.2f} → target {self.target_rps} "
            f"(actual {new_rps:.2f}), scale factor {scale:.3f}"
        )
        return scaled

    @staticmethod
    def _calc_rps(ts: List[int]) -> float:
        """Compute average requests-per-second for a timestamp list."""
        if not ts:
            return 0.0
        duration_ms = ts[-1] or 1  # avoid division by zero
        return len(ts) / (duration_ms / 1_000)
    
    
    def _select_random_window(self, ts: List[int]) -> List[int]:
        """Return a random contiguous sub-trace of length *window_duration_s*."""
        win_ms = self.window_duration_s * 1_000
        total  = ts[-1]

        if win_ms >= total:          # trace shorter than requested → just return it
            logger.warning(
                "Requested window (%ds) longer than trace (%0.1fs) – using full trace",
                self.window_duration_s, total / 1000,
            )
            return ts

        rng = np.random.default_rng(self.seed)
        start_ms = int(rng.integers(0, total - win_ms))
        end_ms   = start_ms + win_ms

        sub = [t - start_ms for t in ts if start_ms <= t < end_ms]

        # Always keep at least one request – extremely unlikely to be empty, but safe
        if not sub:
            sub = [0]

        self._window_start_ms = start_ms               # so callers can inspect/plot
        logger.info(
            "Selected random %ds window: [%0.3f – %0.3f] s (%d requests)",
            self.window_duration_s, start_ms/1000, end_ms/1000, len(sub)
        )
        return sub
