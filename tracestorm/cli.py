import datetime
import os
from typing import Optional, Tuple, Union
import csv

import click

from tracestorm.core import run_load_test
from tracestorm.data_loader import load_datasets
from tracestorm.logger import init_logger
from tracestorm.trace_generator import (
    AzureTraceGenerator,
    SyntheticTraceGenerator,
    TraceGenerator,
)

logger = init_logger(__name__)

# --- Add config.txt reading ---
def read_config(config_path="/home/sean/diss/virtualize_llm/config.txt"):
    config = {}
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return config
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            config[k.strip()] = v.strip()
    return config
# --- end config.txt reading ---

def clear_csv_files(duration, rps, memory_location, batch_size):
    with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/{batch_size}_batch_size/data/token_latency_{memory_location}_duration_{duration}_rps_{rps}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Request_ID", "Token_Index", "Latency (s)"])
    
    # Clear request metrics CSV file
    with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/{batch_size}_batch_size/data/request_metrics_{memory_location}_duration_{duration}_rps_{rps}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Request_ID", "Throughput (tokens/s)", "Average_Latency (s)", "First_Token_Latency (s)", "Max_Token_Latency (s)"])

    # Clear background sync access CSV file
    with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/{batch_size}_batch_size/data/background_synchronisation_{memory_location}_duration_{duration}_rps_{rps}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Operation", "Latency (us)", "Num Tokens", "Num Layers", "Request ID"])

    # with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/mem_free_{memory_location}_input_{input_len}_output_{output_len}.csv", 'w', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["Operation", "Latency (us)", "Num Tokens", "Num Layers"])

    with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/{batch_size}_batch_size/data/write_kv_{memory_location}_duration_{duration}_rps_{rps}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Operation", "Tokens Written", "Layer ID", "Latency (us)"])

    with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/{batch_size}_batch_size/data/read_kv_{memory_location}_duration_{duration}_rps_{rps}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Operation", "Layer ID", "Latency (us)"])

    with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/{batch_size}_batch_size/data/write_kernel_{memory_location}_duration_{duration}_rps_{rps}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Operation", "Tokens Written", "Layer ID", "Latency (us)"])

    with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/{batch_size}_batch_size/data/free_chunks_{memory_location}_duration_{duration}_rps_{rps}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Request ID", "Number of Free Chunks"])
    
    with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/{batch_size}_batch_size/data/prepare_access_{memory_location}_duration_{duration}_rps_{rps}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Operation", "Latency (us)", "Num Tokens", "Layer ID"])
    
    with open(f"/home/sean/diss/virtualize_llm/experiment_results/peer_access/{batch_size}_batch_size/data/non_contig_writes_{memory_location}_duration_{duration}_rps_{rps}.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Request ID", "Num non-contig writes", "Average time for converting to contig line (us)"])

# Valid patterns
SYNTHETIC_PATTERNS = {"uniform", "poisson", "random"}
AZURE_PATTERNS = {"azure_code", "azure_conv"}
VALID_PATTERNS = SYNTHETIC_PATTERNS | AZURE_PATTERNS


def create_trace_generator(
    pattern: str,
    rps: Union[int, float],
    duration: int,
    seed: Optional[int] = None,
) -> Tuple[TraceGenerator, str]:
    """
    Create appropriate trace generator based on pattern and validate parameters.

    Args:
        pattern: Pattern for trace generation
        rps: Requests per second (only for synthetic patterns)
        duration: Duration in seconds (only for synthetic patterns)
        seed: Random seed for reproducibility of trace patterns

    Returns:
        Tuple of (TraceGenerator instance, Warning message or empty string)

    Raises:
        ValueError: If pattern is invalid or parameter combination is incorrect
    """
    warning_msg = ""

    if pattern not in VALID_PATTERNS:
        raise ValueError(
            f"Invalid pattern: {pattern}. Valid patterns are: {sorted(VALID_PATTERNS)}"
        )

    if pattern in SYNTHETIC_PATTERNS:
        if rps < 0:
            raise ValueError("RPS must be non-negative for synthetic patterns")
        if duration < 0:
            raise ValueError(
                "Duration must be non-negative for synthetic patterns"
            )
        return SyntheticTraceGenerator(
            rps, pattern, duration, seed
        ), warning_msg

    # Azure patterns
    if rps != 1:
        warning_msg = (
            f"Warning: RPS parameter ({rps}) is ignored for Azure patterns"
        )
    if duration != 10:
        warning_msg += f"\nWarning: Duration parameter ({duration}) is ignored for Azure patterns"

    dataset_type = pattern.replace("azure_", "")
    return AzureTraceGenerator(dataset_type), warning_msg


@click.command()
@click.option("--model", required=True, help="Model name")
# NOTE: THIS IS NOW OVERWRITTEN BY THE RPS IN CONFIG.TXT
@click.option(
    "--rps",
    type=float,
    default=1.0,
    help="Requests per second (only used with synthetic patterns)",
)
@click.option(
    "--pattern",
    default="uniform",
    type=click.Choice(sorted(VALID_PATTERNS), case_sensitive=False),
    help=f"Pattern for generating trace. Valid patterns: {sorted(VALID_PATTERNS)}",
)
# NOTE: THIS IS NOW OVERWRITTEN BY THE DURATION IN CONFIG.TXT
@click.option(
    "--duration",
    type=int,
    default=10,
    help="Duration in seconds (only used with synthetic patterns)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility of trace patterns",
)
@click.option(
    "--subprocesses", type=int, default=1, help="Number of subprocesses"
)
@click.option(
    "--base-url",
    default=lambda: os.environ.get(
        "OPENAI_BASE_URL", "http://localhost:8000/v1"
    ),
    help="OpenAI Base URL",
)
@click.option(
    "--api-key",
    default=lambda: os.environ.get("OPENAI_API_KEY", "none"),
    help="OpenAI API Key",
)
@click.option(
    "--datasets-config", default=None, help="Config file for datasets"
)
@click.option(
    "--plot",
    is_flag=True,
    default=False,
    help="Generate performance plots",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory to save results (defaults to tracestorm_results/{timestamp})",
)
@click.option(
    "--include-raw-results",
    is_flag=True,
    default=False,
    help="Include raw results in the output",
)
def main(
    model,
    rps,
    pattern,
    duration,
    seed,
    subprocesses,
    base_url,
    api_key,
    datasets_config,
    plot,
    output_dir,
    include_raw_results,
):
    config = read_config()
    method = config["METHOD"]
    batch_size = config["BATCH_SIZE"]
    memory_location = config["MEMORY_LOCATION"]
    duration = int(config.get("DURATION", 10))  # Default to 10 seconds if not set
    rps = float(config.get("RPS", 1.0))  # Default to 1.0 if not set
    rps_str = str(int(rps)) if rps == int(rps) else str(rps)
    clear_csv_files(duration, rps_str, memory_location, batch_size)
    """Run trace-based load testing for OpenAI API endpoints."""
    try:
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("tracestorm_results", timestamp)
            output_dir2 = f"/home/sean/diss/virtualize_llm/experiment_results/{method}/{batch_size}_batch_size/data"

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir2, exist_ok=True)
        logger.info(f"Results will be saved to: {output_dir} and {output_dir2}")

        trace_generator, warning_msg = create_trace_generator(
            pattern, rps, duration, seed
        )
        if warning_msg:
            logger.warning(warning_msg)

        if datasets_config is None:
            datasets = []
            sort_strategy = None
        else:
            datasets, sort_strategy = load_datasets(datasets_config)

        _, result_analyzer = run_load_test(
            trace_generator=trace_generator,
            model=model,
            subprocesses=subprocesses,
            base_url=base_url,
            api_key=api_key,
            datasets=datasets,
            sort_strategy=sort_strategy,
            seed=seed,
        )

        print(result_analyzer)

        # Save raw results (always)
        results_file = os.path.join(output_dir, "results.json")
        results_file2 = os.path.join(output_dir2, f"results_{memory_location}.json")
        result_analyzer.export_json(
            results_file, include_raw=include_raw_results
        )
        result_analyzer.export_json(
            results_file2, include_raw=include_raw_results
        )
        logger.info(f"Raw results saved to: {results_file} and {results_file2}")

        # Only generate plots if requested
        if plot:
            ttft_file = os.path.join(output_dir, "ttft_cdf.png")
            ttft_file2 = os.path.join(output_dir2, f"ttft_cdf_{memory_location}.png")
            tpot_file = os.path.join(output_dir, "tpot_cdf.png")
            tpot_file2 = os.path.join(output_dir2, f"tpot_cdf_{memory_location}.png")
            result_analyzer.plot_cdf(ttft_file=ttft_file, tpot_file=tpot_file)
            logger.info("Performance plots generated")

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise click.UsageError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()