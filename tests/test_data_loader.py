import unittest

from tracestorm.data_loader import Dataset, load_datasets


class TestDataLoader(unittest.TestCase):
    def test_local_files(self):
        """Test loading from local files"""
        # datasets, sort = load_datasets(
        #     "../tracestorm/datasets_config/datasets_config_local.json"
        # )
        # assert isinstance(datasets, list)
        # assert isinstance(datasets[0], Dataset) and isinstance(
        #     datasets[1], Dataset
        # )
        # assert sort == "random"
        # assert len(datasets) == 2
        # assert datasets[0].select_ratio == 6 and datasets[1].select_ratio == 4
        # assert datasets[0].length > 0 and datasets[1].length > 0
        pass

    # separate
    def test_remote_files(self):
        """
        Test loading datasets from hugging face.
        There are 2 datasets, testing for:
        1. loading with datasets.load_dataset
        2. loading csv format with pandas
        """
        datasets, sort = load_datasets(
            "../tracestorm/datasets_config/datasets_config_hf.json"
        )
        assert isinstance(datasets, list)
        assert isinstance(datasets[0], Dataset) and isinstance(
            datasets[1], Dataset
        )
        assert sort == "original"
        assert len(datasets) == 2
        assert datasets[0].select_ratio == 2 and datasets[1].select_ratio == 8
        assert datasets[0].length > 0 and datasets[1].length > 0

    def test_missing_fields(self):
        """Test loading with missing sort strategy and selection ratio in the config file"""
        datasets, sort = load_datasets(
            "../tracestorm/datasets_config/datasets_config_missing.json"
        )
        assert isinstance(datasets, list) and len(datasets) == 2
        assert sort == "random"
        assert datasets[0].select_ratio == 1
        assert datasets[1].select_ratio == 1
        assert datasets[0].length > 0 and datasets[1].length > 0


if __name__ == "__main__":
    unittest.main()
