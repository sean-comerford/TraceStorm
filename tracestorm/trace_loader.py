import os
import tempfile
from typing import List

import pandas as pd
import requests

from tracestorm.constants import AZURE_DATASET_PATHS, AZURE_REPO_URL
from tracestorm.logger import init_logger

# Initialize logger
logger = init_logger("trace_loader")


def download_file_from_github(file_path, save_as):
    """
    Download a file from GitHub and save it locally.

    Parameters:
    file_path (str): The path to the file in the GitHub repository.
    save_as (str): The local filename to save the downloaded file as.

    Raises:
    Exception: If the file download fails.
    """
    raw_url = (
        f"https://raw.githubusercontent.com/{AZURE_REPO_URL}/master/{file_path}"
    )
    response = requests.get(raw_url)

    if response.status_code == 200:
        with open(save_as, "wb") as file:
            file.write(response.content)
        logger.info(f"File downloaded successfully as '{save_as}'")
    else:
        error_msg = (
            f"Failed to download file: {response.status_code} - {response.text}"
        )
        logger.error(error_msg)
        raise Exception(error_msg)


def load_azure_inference_dataset(dataset_type: str) -> List[int]:
    """
    Load the Azure inference dataset based on the specified type and return timestamps.

    Parameters:
    dataset_type (str): The type of dataset to load ('code' or 'conv').

    Returns:
    List[int]: A list of relative timestamps in milliseconds.

    Raises:
    ValueError: If the dataset type is invalid.
    """
    if dataset_type not in AZURE_DATASET_PATHS:
        error_msg = "Invalid dataset type. Please choose 'code' or 'conv'."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Create a temporary directory to store the downloaded file
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = AZURE_DATASET_PATHS[dataset_type]
        local_file = os.path.join(tmpdirname, os.path.basename(file_path))

        try:
            download_file_from_github(file_path, local_file)
            return process_dataset(local_file)
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise


def process_dataset(file_path: str) -> List[int]:
    """
    Process the downloaded dataset and extract relative timestamps in milliseconds.

    Parameters:
    file_path (str): The path to the downloaded dataset file.

    Returns:
    List[int]: A sorted list of relative timestamps in milliseconds, starting from 0.

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


# Example usage
# load_azure_inference_dataset("code")
# load_azure_inference_dataset("conv")
