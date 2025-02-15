import os
import json
from typing import List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from datasets import load_dataset

from tracestorm.constants import DEFAULT_DATASET_FOLDER
from tracestorm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class Dataset:
    """
    Each Dataset object contains name of the dataset, a list of prompts, 
    the select ratio among all datasets, and the total number of prompts  
    """
    file_name: str
    prompts: List[str]
    select_ratio: float
    length: int
    

def normalize_prompts(row) -> List[str]:
    """
    Convert one row to a list of prompts based on the format.
    """
    prompts = []
    if isinstance(row, list): # if the row contains a list of prompts
        for item in row:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict) and item.get("role") == "user":
                prompts.append(item.get("content", ""))
    elif isinstance(row, str): # if the row is already a prompt
        prompts.append(row)
    elif isinstance(row, dict) and row.get("role") == "user": # if the row is a template, retrieve user prompt
        prompts.append(row.get("content", ""))
    else:
        logger.error(f"Unrecognized row format: {row}")
    return [p for p in prompts if p]  # Remove empty prompts    
            
def load_datasets(
    datasets_config_file: Optional[str] = None,
) -> Tuple[List[Dataset], Optional[str]]:
    """
    Load datasets from local files or Hugging Face datasets.
    
    Args:
        datasets_config_file: A dataset configuration file containing file paths, 
        prompt fields, selection ratios, and sorting strategies.
        A customized data loading logic needs to be implemented if no 
        datasets_config_file is provided.
        
    Return:
        (List[Dataset], str): A list of Dataset objects and the sorting strategy.
    """
    # Load datasets configuration file
    if datasets_config_file:
        try:
            with open(datasets_config_file, "r") as f:
                datasets_config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file '{datasets_config_file}' not found")
            return [], None
        except Exception as e:
            logger.error(f"Error reading '{datasets_config_file}': {e}")
            return [], None
        
        # Strategy to sort the provided datasets
        sort_strategy = datasets_config.get("sort", "random")

        # List to store each Dataset
        datasets = []

        for name, config in datasets_config.items():
            file_name = config.get("file_name")
            prompt_field = config.get("prompt_field")
            
            try:
                ratio = float(config.get("select_ratio", 1.0))
            except ValueError:
                logger.error(f"Invalid 'select_ratio' for dataset '{name}', using default 1")
                ratio = 1.0

            if not file_name or not prompt_field:
                logger.error(
                    f"Missing required 'file_name' or 'prompt_field' for dataset '{name}'"
                )
                continue
            if os.path.isfile(file_name):
                file_path = os.path.abspath(file_name)
            else:
                file_path = os.path.join(DEFAULT_DATASET_FOLDER, file_name)
               
            # Load dataset from local files
            if os.path.exists(file_path):
                prompts = []
                # CSV files
                if file_name.endswith(".csv"):
                    data = pd.read_csv(file_path)

                    if prompt_field not in set(data.column_names):
                        logger.error(f"Field '{prompt_field}' not found in '{file_path}'.")
                        continue
                    prompts = data[prompt_field].dropna().astype(str).tolist()
                # JSON files
                elif file_name.endswith(".json"):
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    
                    if isinstance(data, dict):
                        prompts = data.get(prompt_field, [])
                        if not isinstance(prompts, list):
                            logger.error(f"Field '{prompt_field}' in '{file_path}' is not a list.")
                            continue
                else:
                    logger.error(f"Unsupported file format for '{file_name}'")
                    continue    
            else: # Load HF datasets
                # data = load_dataset("lmsys/lmsys-chat-1m")
                data = load_dataset(file_name)["train"]
                if prompt_field not in data.column_names:
                    logger.error(f"'{prompt_field}' not found in dataset '{file_name}'")
                    continue
                
                prompts = []
                for row in data[prompt_field]:
                    prompts.extend(normalize_prompts(row))
                    
            # Add the dataset information (file name, a list of prompts, select ratio among all datasets, total number of prompts)
            dataset_obj = Dataset(file_name, prompts, ratio, len(prompts))
            datasets.append(dataset_obj)
            
            logger.info(
                f"loaded {file_path} with {len(prompts)} prompts, selection ratio = {ratio}"
            )
            
        return datasets, sort_strategy
    
    else:
        logger.error("Customized data loading logic needs to be implemented!")
        return [], None
