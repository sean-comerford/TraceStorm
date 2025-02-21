# TraceStorm

TraceStorm is a tool for generating and replaying traces of requests to OpenAI API endpoints. It allows users to simulate load testing by generating requests based on specified patterns and configurations.

## Features

- Generate synthetic traces using various patterns (e.g., uniform, poisson, etc.)
- Load traces from public datasets (e.g., [Azure LLM Inference dataset](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md))
- Replay requests to any OpenAI-compatible LLM service
- Analyze results and visualize performance metrics

## Installation

```bash
pip install tracestorm
```

## Usage

### Start an OpenAI-Compatible Server

Before running the load test, ensure you have an OpenAI-compatible server running. If you haven't already installed `vllm`, you can do so with the following command:

```bash
# Install vllm if you haven't already
pip install vllm

# Start the server with the desired model
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

### Run the Load Test

Once the server is running, you can execute the load test using the `tracestorm` command. Here's how to run it with different options:

#### Example Command for Synthetic Trace

```bash
tracestorm --model "Qwen/Qwen2.5-1.5B-Instruct" --rps 5 --pattern uniform --duration 10
```

#### Example Command for Azure Trace

To load a trace from the Azure dataset, use:

```bash
tracestorm --model "Qwen/Qwen2.5-1.5B-Instruct" --pattern azure_code
```

#### Example Command for Loading Prompts from Datasets

```bash
tracestorm --model "Qwen/Qwen2.5-1.5B-Instruct" --duration 30 --datasets-config-file ./examples/datasets_config_hf.json
```


**Supported Dataset Sources**:

1. Locally stored, pre-processed datasets
- Refer to `./examples/datasets_config_local.json` for an example configuration.
- If you want to test loading from local files, please run `./examples/test_data_loader.py` first to download and save two datasets.

2. Remote datasets from Hugging Face 
- Refer to `./examples/datasets_config_hf.json` for an example configuration.

**Sorting Strategy**: Defines how prompts from multiple datasets are ordered
- random (default): Shuffles prompts randomly.
- original: Maintains the original order of prompts.

Please check `./examples/datasets_config_default.json` for required fields in `datasets-config-file`. This file contains placeholders for necessary configurations.


### Command Options

- `--model`: Required. The name of the model to use.
- `--rps`: Optional. Requests per second (default is 1, only used for synthetic patterns).
- `--pattern`: Optional. Pattern for generating trace. Valid patterns include:
  - `uniform`: Distributes requests evenly across the duration.
  - `poisson`: Generates request timings based on a Poisson process.
  - `random`: Generates requests at random intervals within the duration.
  - `azure_code`: Loads the Azure inference dataset for code.
  - `azure_conv`: Loads the Azure inference dataset for conversation.
- `--duration`: Optional. Duration in seconds (default is 10, only used for synthetic patterns).
- `--subprocesses`: Optional. Number of subprocesses to use (default is 1).
- `--base-url`: Optional. OpenAI Base URL (default is `http://localhost:8000/v1`).
- `--api-key`: Optional. OpenAI API Key (default is `none`).
- `--seed`: Optional. Random seed for trace pattern reproducibility (default is `none`).
- `--datasets-config-file`: Optional. Configuration file for loading prompt messages from provided datasets. Uses `DEFAULT_MESSAGES` is not specified.

Make sure to adjust the parameters according to your testing needs!
