import unittest
import pandas as pd
import os

from tracestorm.data_loader import Dataset, load_datasets
from tracestorm.constants import DEFAULT_DATASET_FOLDER


class TestDataLoader(unittest.TestCase):

    def test_remote_files(self):
        """
        Test loading datasets from hugging face.
        There are 2 datasets, testing for:
        1. loading with datasets.load_dataset
        2. loading csv format with pandas
        """
        datasets, sort = load_datasets(
            "examples/datasets_config_hf.json"
        )
        assert isinstance(datasets, list)
        assert isinstance(datasets[0], Dataset) and isinstance(
            datasets[1], Dataset
        )
        assert sort == "original"
        assert len(datasets) == 2
        assert datasets[0].select_ratio == 2 and datasets[1].select_ratio == 8
        assert datasets[0].length > 0 and datasets[1].length > 0

     
    def test_local_files(self):
        """Test loading from local files"""
        
        os.makedirs(DEFAULT_DATASET_FOLDER, exist_ok=True)
        # testing datasets
        df1 = pd.read_json("hf://datasets/MAsad789565/Coding_GPT4_Data/Data/GPT_4_Coding.json")
        df2 = pd.read_json("hf://datasets/olathepavilion/Conversational-datasets-json/Validation.jsonl", lines=True)

       # test with different file formats
        path1 = os.path.join(DEFAULT_DATASET_FOLDER, "GPT4_coding_sample.csv")
        path2 = os.path.join(DEFAULT_DATASET_FOLDER, "Conversational_dataset.jsonl")

        # save the pre-processed dataset to the default folder for test
        df1.to_csv(path1, index=False)
        df2.to_json(path2, orient="records", lines=True)
        
        datasets, sort = load_datasets(os.
            "examples/datasets_config_local.json"
        )
        assert isinstance(datasets, list)
        assert isinstance(datasets[0], Dataset) and isinstance(
            datasets[1], Dataset
        )
        assert sort == "random"
        assert len(datasets) == 2
        assert datasets[0].select_ratio == 6 and datasets[1].select_ratio == 4
        assert datasets[0].length > 0 and datasets[1].length > 0


if __name__ == "__main__":
    unittest.main()
