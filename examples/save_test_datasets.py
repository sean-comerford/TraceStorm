import os

import pandas as pd

from tracestorm.constants import DEFAULT_DATASET_FOLDER


def prepare_test_datasets():
    df1 = pd.read_json(
        "hf://datasets/MAsad789565/Coding_GPT4_Data/Data/GPT_4_Coding.json"
    )
    df2 = pd.read_json(
        "hf://datasets/olathepavilion/Conversational-datasets-json/Validation.jsonl",
        lines=True,
    )

    # save the pre-processed dataset to the default folder for test
    os.makedirs(DEFAULT_DATASET_FOLDER, exist_ok=True)
    path1 = os.path.join(DEFAULT_DATASET_FOLDER, "GPT4_coding_sample.csv")
    path2 = os.path.join(DEFAULT_DATASET_FOLDER, "Conversational_dataset.jsonl")

    # test with different file formats
    df1.to_csv(path1, index=False)
    df2.to_json(path2, orient="records", lines=True)


if __name__ == "__main__":
    prepare_test_datasets()
