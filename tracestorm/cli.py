import datetime
import os
from typing import Optional, Tuple, Union
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
            k = k.strip()
            v = v.strip()
            if k == "RPS":
                try:
                    v_float = float(v)
                    if v_float.is_integer():
                        config[k] = int(v_float)
                    else:
                        config[k] = v_float
                except ValueError:
                    config[k] = v
            else:
                config[k] = v
    return config
# --- end config.txt reading ---

def clear_csv_files(duration, rps, memory_location, batch_size, method, dataset, base_data_dir):
    csv_files = [
        (
            f"/home/sean/diss/virtualize_llm/experiment_results/{method}/"
            f"{batch_size}_batch_size/{dataset}/data/token_latency_{memory_location}_duration_{duration}_rps_{rps}.csv",
            ["Request_ID", "Token_Index", "Latency (s)"]
        ),
        (
            f"/home/sean/diss/virtualize_llm/experiment_results/{method}/"
            f"{batch_size}_batch_size/{dataset}/data/write_kv_{memory_location}_duration_{duration}_rps_{rps}.csv",
            ["Operation", "Tokens Written", "Layer ID", "Latency (us)", "Average size of write per layer (bytes)"]
        ),
        (
            f"/home/sean/diss/virtualize_llm/experiment_results/{method}/"
            f"{batch_size}_batch_size/{dataset}/data/write_kernel_{memory_location}_duration_{duration}_rps_{rps}.csv",
            ["Operation", "Tokens Written", "Layer ID", "Latency (us)"]
        ),
        (
            f"/home/sean/diss/virtualize_llm/experiment_results/{method}/"
            f"{batch_size}_batch_size/{dataset}/data/prepare_access_{memory_location}_duration_{duration}_rps_{rps}.csv",
            ["Operation", "Latency (us)", "Num Tokens", "Layer ID"]
        ),
        (
            f"/home/sean/diss/virtualize_llm/experiment_results/{method}/"
            f"{batch_size}_batch_size/{dataset}/data/non_contig_writes_{memory_location}_duration_{duration}_rps_{rps}.csv",
            ["Request ID", "Num non-contig writes", "Average time for converting to contig line (us)"]
        ),
        (
            base_data_dir + f"/input_output_lengths_{memory_location}_duration_{duration}_rps_{rps}.csv",
            ["Request Timestamp", "Input Length", "Output Length"]
        ),
    ]
    for file_path, header in csv_files:
        try:
            with open(file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
        except Exception as e:
            logger.error(f"[TRACESTORM] Error clearing CSV file {file_path}: {e}")

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
    dataset_type = pattern.replace("azure_", "")

    target_rps: Optional[float]
    if rps in (None, 0):
        target_rps = None          # keep native tempo
    else:
        if rps <= 0:
            raise ValueError("RPS must be > 0 when scaling an Azure trace")
        target_rps = float(rps)

    window_duration = duration if duration else None   # 0 ⇒ full trace

    return (
        AzureTraceGenerator(
            dataset_type=dataset_type,
            target_rps=target_rps,
            window_duration=window_duration,
            seed=seed,
        ),
        warning_msg,
    )

def plot_arrival_distribution(
    timestamps_ms: list[int],
    save_path: str,
    rps,
    dataset
) -> None:
    """
    Parameters
    ----------
    timestamps_ms : list[int]
        Relative timestamps (ms) starting at 0, e.g. trace_generator.timestamps
    save_path : str
        Full filename for the PNG.
    """
    if not timestamps_ms:
        logger.warning("No timestamps – skipping arrival histogram.")
        return

    # One-second buckets
    max_s = int(np.ceil(timestamps_ms[-1] / 1000))
    hist, _ = np.histogram(timestamps_ms, bins=max_s, range=(0, max_s * 1000))

    plt.figure(figsize=(8, 4))
    plt.bar(range(max_s), hist, width=1.0, edgecolor="none")
    plt.xlabel("Seconds since window start")
    plt.ylabel("Requests")
    plt.title(f"Request-arrival distribution (random window) - RPS - {rps} - Dataset - {dataset}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info("Arrival-distribution plot saved to: %s", save_path)

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
@click.option(
    "--dataset",
    default=None,
    help="Dataset to use (overrides config.txt if provided)",
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
    dataset
):
    config = read_config()
    method = config["METHOD"]
    batch_size = config["BATCH_SIZE"]
    memory_location = config["MEMORY_LOCATION"]
    duration = int(config.get("DURATION", 10))  # Default to 10 seconds if not set
    rps = rps if rps is not None else config["RPS"]
    dataset = dataset if dataset is not None else config["DATASET"]

    
    """Run trace-based load testing for OpenAI API endpoints."""

    color_map = {
        "remote": "#E24A33",   # red/orange
        "local": "#348ABD",    # blue
        "original": "#F5A623", # orange
    }
    base_dir_data = f"/home/sean/diss/virtualize_llm/experiment_results/{method}/{batch_size}_batch_size/{dataset}/data"
    base_dir_plots = f"/home/sean/diss/virtualize_llm/experiment_results/{method}/{batch_size}_batch_size/{dataset}/plots"
    clear_csv_files(duration, rps, memory_location, batch_size, method, dataset, base_data_dir=base_dir_data)
    try:
        # Set up output directory for data
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("tracestorm_results", timestamp)
            output_dir2 = base_dir_data

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir2, exist_ok=True)
        logger.info(f"Results will be saved to: {output_dir} and {output_dir2}")

        trace_generator, warning_msg = create_trace_generator(
            pattern, rps, duration, seed
        )
        if warning_msg:
            logger.warning(warning_msg)
            
        
        # Always use dataset_config_{dataset}.json from the absolute path
        print(f"Loading {dataset} Dataset ...")
        dataset_config_file = f"/home/sean/diss/virtualize_llm/TraceStorm/examples/datasets_config_{dataset}.json"
        if not os.path.exists(dataset_config_file):
            logger.error(f"Dataset config file not found: {dataset_config_file}")
            raise click.UsageError(f"Dataset config file not found: {dataset_config_file}")
        datasets, sort_strategy = load_datasets(dataset_config_file)
        
        aggregated_results, result_analyzer = run_load_test(
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
        # Write input/output lengths to CSV
        input_output_csv = os.path.join(base_dir_data, f"input_output_lengths_{memory_location}_duration_{duration}_rps_{rps}.csv")
        with open(input_output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Request Timestamp", "Input Length", "Output Length"])
            for name, timestamp, resp in aggregated_results:
                # Use timestamp as Request ID, or change as needed
                input_length = resp.get("input_length", 0)
                output_length = resp.get("output_length", 0)
                writer.writerow([timestamp, input_length, output_length])
        logger.info(f"Input/output lengths CSV saved to: {input_output_csv}")

        # Save raw results (always)
        results_file = os.path.join(output_dir, "results.json")
        results_file2 = os.path.join(output_dir2, f"results_{memory_location}_duration_{duration}_rps_{rps}.json")
        result_analyzer.export_json(
            results_file, include_raw=include_raw_results, method=method, batch_size=batch_size, dataset=dataset, duration=duration, rps=rps, memory_location=memory_location
        )
        result_analyzer.export_json(
            results_file2, include_raw=include_raw_results, method=method, batch_size=batch_size, dataset=dataset, duration=duration, rps=rps, memory_location=memory_location
        )
        logger.info(f"Raw results saved to: {results_file} and {results_file2}")
        
        # Save tpot and ttft data to csv files
        result_analyzer.save_tpot_ttft(base_dir_data, memory_location, duration, rps)
        

        # Generate plots
        if plot:
            # Plot cdf of ttft and tpot for local, remote and original
            local_tpot_file = os.path.join(
                base_dir_data, f"tpot_local_duration_{duration}_rps_{rps}.csv"
            )
            remote_tpot_file = os.path.join(
                base_dir_data, f"tpot_remote_duration_{duration}_rps_{rps}.csv"
            )
            original_tpot_file = os.path.join(
                base_dir_data, f"tpot_original_duration_{duration}_rps_{rps}.csv"
            )
            local_ttft_file = os.path.join(
                base_dir_data, f"ttft_local_duration_{duration}_rps_{rps}.csv"
            )
            remote_ttft_file = os.path.join(
                base_dir_data, f"ttft_remote_duration_{duration}_rps_{rps}.csv"
            )
            original_ttft_file = os.path.join(
                base_dir_data, f"ttft_original_duration_{duration}_rps_{rps}.csv"
            )
            result_analyzer.plot_cdf_comparison(base_dir_plots, color_map, duration, rps, 
                local_tpot_file, remote_tpot_file, original_tpot_file,
                local_ttft_file, remote_ttft_file, original_ttft_file
            )
            logger.info("Plotted CDF of TTFT and TPOT for local, remote and original.")
            
            # ttft_file = os.path.join(output_dir, "ttft_cdf.png")
            # ttft_file2 = os.path.join(output_dir2, f"ttft_cdf_{memory_location}.png")
            # tpot_file = os.path.join(output_dir, "tpot_cdf.png")
            # tpot_file2 = os.path.join(output_dir2, f"tpot_cdf_{memory_location}.png")
        #     result_analyzer.plot_cdf(ttft_file=ttft_file, tpot_file=tpot_file,
        # method=method, batch_size=batch_size, dataset=dataset, duration=duration, rps=rps, memory_location=memory_location)
            # Plot cdf of write latency for local and remote
            local_write_csv = os.path.join(
                base_dir_data,
                f"write_kv_local_duration_{duration}_rps_{rps}.csv"
            )
            remote_write_csv = os.path.join(
                base_dir_data,
                f"write_kv_remote_duration_{duration}_rps_{rps}.csv"
            )
            write_cdf_output_name = f"write_kv_latency_cdf_duration_{duration}_rps_{rps}.png"
            write_plot_path = os.path.join(base_dir_plots, write_cdf_output_name)
            result_analyzer.plot_write_latency_cdf(local_write_csv, remote_write_csv, write_plot_path, color_map)
            logger.info(f"Write latency CDF plot saved to {write_plot_path}")
            
            # Plot graph of timeline of batch size of write
            batch_write_timeline_output_name = f"write_kv_batch_size_timeline_duration_{duration}_rps_{rps}.png"
            batch_write_timeline_plot_path = os.path.join(base_dir_plots, batch_write_timeline_output_name)
            result_analyzer.plot_batch_size_timeline(local_write_csv, remote_write_csv, batch_write_timeline_plot_path, color_map)
            
            # Plot the cdf of prepare_access latency for remote and local
            local_prepare_csv = os.path.join(
                base_dir_data,
                f"prepare_access_local_duration_{duration}_rps_{rps}.csv"
            )
            remote_prepare_csv = os.path.join(
                base_dir_data,
                f"prepare_access_remote_duration_{duration}_rps_{rps}.csv"
            )
            prepare_cdf_output_name = f"prepare_access_latency_cdf_duration_{duration}_rps_{rps}.png"
            prepare_plot_path = os.path.join(base_dir_plots, prepare_cdf_output_name)
            result_analyzer.plot_prepare_access_cdf(local_prepare_csv, remote_prepare_csv, prepare_plot_path, color_map)
            logger.info(f"Prepare access CDF plot saved to {prepare_plot_path}")

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise click.UsageError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise
    
    # ensure plots directory exists
    plots_dir = f"/home/sean/diss/virtualize_llm/experiment_results/{method}/{batch_size}_batch_size/{dataset}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    arrival_plot = os.path.join(plots_dir, f"arrival_hist_{memory_location}_duration_{duration}_rps_{rps}.png")
    if hasattr(trace_generator, "timestamps"):
        plot_arrival_distribution(trace_generator.timestamps, arrival_plot, rps, dataset)
        print(f"Request arrival distribution plot saved to: {arrival_plot}")
    
    input_output_csv = base_dir_data + f"/input_output_lengths_{memory_location}_duration_{duration}_rps_{rps}.csv"
    df = pd.read_csv(input_output_csv)
    # Input Length Histogram
    plt.figure(figsize=(8, 4))
    min_val = 0
    max_val = int(df["Input Length"].max())
    bins = np.arange(min_val, max_val + 25, 25)
    plt.hist(df["Input Length"], bins=bins, color="#348ABD", edgecolor="black")
    plt.xlabel("Input Prompt Length (tokens)")
    plt.xticks(np.arange(0, max_val + 101, 100))
    plt.xlim(left=0)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Input Prompt Lengths - Dataset: {dataset}")
    plt.tight_layout()
    input_hist_path = os.path.join(plots_dir, f"input_length_hist_{memory_location}_duration_{duration}_rps_{rps}.png")
    plt.savefig(input_hist_path, dpi=150)
    plt.close()
    logger.info(f"Input prompt length histogram saved to: {input_hist_path}")

    plt.figure(figsize=(8, 4))
    # Set bin edges for width 25 bins, starting at 0
    min_val = 0
    max_val = int(df["Output Length"].max())
    bins = np.arange(min_val, max_val + 25, 25)
    plt.hist(df["Output Length"], bins=bins, color="#E24A33", edgecolor="black")
    plt.xlabel("Output Length (tokens)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Output Lengths - Dataset: {dataset}")
    plt.tight_layout()
    output_hist_path = os.path.join(plots_dir, f"output_length_hist_{memory_location}_duration_{duration}_rps_{rps}.png")
    # Set x-axis ticks to increments of 25
    plt.xticks(np.arange(0, max_val + 101, 100))
    plt.xlim(left=0)
    plt.savefig(output_hist_path, dpi=150)
    plt.close()
    logger.info(f"Output length histogram saved to: {output_hist_path}")


if __name__ == "__main__":
    main()