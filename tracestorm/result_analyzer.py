import json
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tracestorm.logger import init_logger
from tracestorm.utils import get_unique_file_path
import csv
import os

logger = init_logger(__name__)


class ResultAnalyzer:
    def __init__(self):
        """
        Initializes the ResultAnalyzer with an empty raw_results list.
        """
        self.raw_results: List[Dict[str, Any]] = []
        self.ttft: List[float] = []  # time to first token
        self.tpot: List[List[float]] = []  # time per output token
        self.throughputs: List[float] = []  # throughput for each request

    def add_result(
        self, name: str, timestamp: int, resp: Dict[str, Any]
    ) -> None:
        self.raw_results.append(
            {"name": name, "timestamp": timestamp, "response": resp}
        )
        ttft, tpot = self.process_time_records(resp.get("time_records", []))
        throughput = resp.get("throughput")
        if throughput is not None:
            self.throughputs.append(throughput)
        if not ttft or not tpot:
            logger.error(
                f"Error processing time_records for {name} at timestamp {timestamp}"
            )
            return
        self.ttft.append(ttft)
        self.tpot.append(tpot)

    def store_raw_results(
        self, aggregated_raw_results: List[Tuple[str, int, Dict[str, Any]]]
    ) -> None:
        for name, timestamp, resp in aggregated_raw_results:
            self.add_result(name, timestamp, resp)

    def process_time_records(
        self, time_records: List[int]
    ) -> Tuple[float, List[float]]:
        """
        Returns the time to first token and time per output token.
        """
        if not time_records or len(time_records) == 1:
            logger.error("No time_records to process.")
            return None, None

        # ttot = time_records[1] - time_records[0]
        # tpot = time_records[i+1] - time_records[i], i > 1
        # use numpy for faster computation
        ts1 = np.array(time_records[:-1])
        ts2 = np.array(time_records[1:])
        latency = ts2 - ts1
        ttft = latency[0]
        tpot = latency[1:].tolist()
        
        # Debugging, log the first requests time records and latencies for debugging
        if not hasattr(self, 'first_request_logged'):
            logger.info(f"First request time_records: {time_records}")
            logger.info(f"First request latencies: {latency}")
            self.first_request_logged = True
        return ttft, tpot

    def summarize(
        self, percentiles: List[float] = [0.25, 0.5, 0.75, 0.99]
    ) -> Dict[str, Any]:
        """
        Returns a comprehensive summary of the results, including various statistics.

        Returns:
            Dict[str, Any]: A dictionary containing statistical summaries for ttft and tpot.
        """
        if not self.ttft or not self.tpot:
            logger.error("No results to summarize.")
            return {}

        # Convert ttft and tpot to Pandas Series
        ttft_series = pd.Series(self.ttft)

        # Flatten tpot list of lists to a single Series
        tpot_flat = [item for sublist in self.tpot for item in sublist]
        tpot_series = pd.Series(tpot_flat)
        
        throughput_series = pd.Series(self.throughputs)

        # Compute statistics for ttft
        ttft_stats = ttft_series.describe(percentiles=percentiles).to_dict()
        logger.debug(f"ttft_stats: {ttft_stats}")
        # Compute statistics for tpot
        tpot_stats = tpot_series.describe(percentiles=percentiles).to_dict()
        logger.debug(f"tpot_stats: {tpot_stats}")
        
        throughput_stats = throughput_series.describe(percentiles=percentiles).to_dict()
        logger.debug(f"throughput_stats: {throughput_stats}")

        # Organize the summary
        summary = {"ttft": ttft_stats, "tpot": tpot_stats, "throughput": throughput_stats}

        return summary

    def __str__(self) -> str:
        """
        Returns a string representation of the summary in an elegant format.

        Returns:
            str: Formatted summary of ttft and tpot statistics.
        """
        summary = self.summarize()
        if not summary:
            return "No results to summarize."

        ttft_stats = summary.get("ttft", {})
        tpot_stats = summary.get("tpot", {})
        throughput_stats = summary.get("throughput", {})

        # Helper function to format statistics
        def format_stats(title: str, stats: Dict[str, float]) -> str:
            lines = [f"{title.upper()} Statistics:"]
            for key, value in stats.items():
                lines.append(f"  {key}: {value:.4f}")
            return "\n".join(lines)

        ttft_str = format_stats("ttft", ttft_stats)
        tpot_str = format_stats("tpot", tpot_stats)
        throughput_str = format_stats("throughput", throughput_stats)

        return f"{ttft_str}\n\n{tpot_str}\n\n{throughput_str}"  

    def export_json(self, file_path: str, include_raw: bool = False, method=None, batch_size=None, dataset=None, duration=None, rps=None, memory_location=None, runtime=None) -> None:
        """
        Exports the summary statistics and optionally the raw results to a JSON file.

        Args:
            file_path (str): The path to the JSON file to be created.
            include_raw (bool, optional): Whether to include raw results in the export. Defaults to False.

        Raises:
            ValueError: If there are no results to export.
            IOError: If there is an error writing to the file.
        """
        summary = self.summarize()
        if not summary:
            logger.error("No results to export.")
            raise ValueError("No results to export.")

        # --- Add write statistics ---
        base_dir = f"/home/sean/diss/virtualize_llm/experiment_results/{method}/{batch_size}_batch_size/{dataset}/data"
        write_csv = f"{base_dir}/write_kv_{memory_location}_duration_{duration}_rps_{rps}.csv"

        # Read Latency (us) and Average size of write per layer (bytes)
        latencies_us = []
        write_sizes = []
        try:
            with open(write_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lat = row.get("Latency (us)")
                    size = row.get("Average size of write per layer (bytes)")
                    if lat:
                        latencies_us.append(float(lat))
                    if size:
                        write_sizes.append(float(size))
        except Exception as e:
            logger.warning(f"Could not read write stats from {write_csv}: {e}")

        # Convert latencies to s
        latencies_s = np.array(latencies_us) / 1_000_000.0 if latencies_us else np.array([])

        write_stats = {}
        if latencies_s.size > 0:
            write_stats = {
                "count": float(len(latencies_s)),
                "mean": float(np.mean(latencies_s)),
                "std": float(np.std(latencies_s)),
                "min": float(np.min(latencies_s)),
                "25%": float(np.percentile(latencies_s, 25)),
                "50%": float(np.percentile(latencies_s, 50)),
                "75%": float(np.percentile(latencies_s, 75)),
                "99%": float(np.percentile(latencies_s, 99)),
                "max": float(np.max(latencies_s)),
                "avg_bytes": float(np.mean(write_sizes)) if write_sizes else None,
            }
        else:
            write_stats = {k: None for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "99%", "max", "avg_bytes"]}

        # Add write section to summary
        summary["write"] = write_stats

        data_to_export = {"summary": summary}
        if include_raw:
            data_to_export["raw_results"] = self.raw_results

        file_path = get_unique_file_path(file_path, runtime=runtime)
        try:
            with open(file_path, "w") as f:
                json.dump(data_to_export, f, indent=4)
            logger.info(f"Exported data to {file_path} successfully.")
        except IOError as e:
            logger.error(f"Failed to export data to {file_path}: {e}")
            raise
    
    # Helper to compute average from a CSV column
    def compute_avg_size(self, csv_path, col_name):
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                sizes = [float(row[col_name]) for row in reader if row.get(col_name)]
            return sum(sizes) / len(sizes) if sizes else None
        except Exception as e:
            logger.warning(f"Could not compute average from {csv_path}: {e}")
            return None

    def plot_cdf(
        self, ttft_file: str = "ttft_cdf.png", tpot_file: str = "tpot_cdf.png",
        method=None, batch_size=None, dataset=None, duration=None, rps=None, memory_location=None
    ) -> None:
        """
        Plots the Cumulative Distribution Function (CDF) of ttft and tpot and saves the figures to files.

        Args:
            ttft_file (str, optional): File path to save the ttft CDF plot. Defaults to "ttft_cdf.png".
            tpot_file (str, optional): File path to save the tpot CDF plot. Defaults to "tpot_cdf.png".

        Raises:
            ValueError: If there is no data to plot.
            IOError: If there is an error saving the plot files.
        """
        if not self.ttft and not self.tpot:
            logger.error("No data available to plot.")
            raise ValueError("No data available to plot.")

        # Set Seaborn style for enhanced aesthetics
        sns.set(style="whitegrid")

        # Plot CDF for TTFT
        if self.ttft:
            print(f"[DEBUG] Plotting TTFT CDF with {len(self.ttft)} data points.")
            try:
                plt.figure(figsize=(8, 6))
                sns.ecdfplot(self.ttft, color="blue")
                plt.title("CDF of Time to First Token (TTFT)")
                plt.xlabel("TTFT")
                plt.ylabel("Cumulative Probability")
                plt.tight_layout()
                ttft_file = get_unique_file_path(ttft_file)
                plt.savefig(ttft_file)
                logger.info(f"TTFT CDF plot saved to {ttft_file}.")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to plot TTFT CDF: {e}")
                raise

        # Flatten tpot list of lists to a single list for plotting
        tpot_flat = [item for sublist in self.tpot for item in sublist]
        if tpot_flat:
            print(f"[DEBUG] Plotting TPOT CDF with {len(tpot_flat)} data points.")
            try:
                plt.figure(figsize=(8, 6))
                sns.ecdfplot(tpot_flat, color="green")
                plt.title("CDF of Time per Output Token (TPOT)")
                plt.xlabel("TPOT")
                plt.ylabel("Cumulative Probability")
                plt.tight_layout()
                tpot_file = get_unique_file_path(tpot_file)
                plt.savefig(tpot_file)
                logger.info(f"TPOT CDF plot saved to {tpot_file}.")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to plot TPOT CDF: {e}")
                raise
            

    # Save self.tpot and self.ttft to CSV files (convert seconds to milliseconds)
    def save_tpot_ttft(self, base_dir_data, memory_location, duration, rps, runtime):
        """
        Save the tpot and ttft data to CSV files for further analysis.
        Times are converted from seconds to milliseconds.
        """
        if not os.path.exists(base_dir_data):
            logger.error(f"Base directory does not exist: {base_dir_data}")
            return

        # Prepare file paths
        tpot_file = os.path.join(base_dir_data, f"tpot_{memory_location}_duration_{duration}_rps_{rps}_runtime_{runtime}.csv")
        ttft_file = os.path.join(base_dir_data, f"ttft_{memory_location}_duration_{duration}_rps_{rps}_runtime_{runtime}.csv")

        # Save tpot data (flatten and convert to ms)
        try:
            with open(tpot_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["tpot (ms)"])
                for tpot_list in self.tpot:
                    for tpot in tpot_list:
                        writer.writerow([tpot * 1000.0])
            logger.info(f"TPOT data saved to {tpot_file}")
        except Exception as e:
            logger.error(f"Failed to save TPOT data: {e}")

        # Save ttft data (convert to ms)
        try:
            with open(ttft_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ttft (ms)"])
                for ttft in self.ttft:
                    writer.writerow([ttft * 1000.0])
            logger.info(f"TTFT data saved to {ttft_file}")
        except Exception as e:
            logger.error(f"Failed to save TTFT data: {e}")

    # Plot the CDF of original, remote, and local TPOT and TTFT on separate plots
    def plot_cdf_comparison(self, base_dir_plots, color_map, duration, rps, local_tpot_file, remote_tpot_file, original_tpot_file, 
                            local_ttft_file, remote_ttft_file, original_ttft_file):
        """
        Plots CDF comparisons for TPOT and TTFT across original, remote, and local data.

        Args:
            base_dir_plots (str): Directory to save the plots.
            color_map (dict): Mapping of 'original', 'remote', 'local' to colors.
            duration, rps: Used for plot file naming.
            *_tpot_file, *_ttft_file (str): CSV file paths for each data source.
        """

        # Helper to read a single-column CSV and return the values as a numpy array
        def read_csv_column(file_path, col_name):
            try:
                df = pd.read_csv(file_path)
                return df[col_name].values
            except Exception as e:
                logger.warning(f"Could not read {col_name} from {file_path}: {e}")
                return []

        # Read TPOT data (ms)
        local_tpot = read_csv_column(local_tpot_file, "tpot (ms)")
        remote_tpot = read_csv_column(remote_tpot_file, "tpot (ms)")
        original_tpot = read_csv_column(original_tpot_file, "tpot (ms)")

        # Plot TPOT CDF
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        if len(original_tpot):
            sns.ecdfplot(original_tpot, label="Original", color=color_map.get("original", "black"))
        if len(remote_tpot):
            sns.ecdfplot(remote_tpot, label="Remote", color=color_map.get("remote", "red"))
        if len(local_tpot):
            sns.ecdfplot(local_tpot, label="Local", color=color_map.get("local", "blue"))
        plt.xlabel("TPOT (ms)")
        plt.ylabel("Cumulative Probability")
        plt.title("CDF of Time per Output Token (TPOT)")
        plt.legend()
        plt.tight_layout()
        tpot_plot_path = os.path.join(base_dir_plots, f"tpot_cdf_comparison_duration_{duration}_rps_{rps}.png")
        plt.savefig(tpot_plot_path)
        plt.close()
        logger.info(f"TPOT CDF comparison plot saved to {tpot_plot_path}")

        # Read TTFT data (ms)
        local_ttft = read_csv_column(local_ttft_file, "ttft (ms)")
        remote_ttft = read_csv_column(remote_ttft_file, "ttft (ms)")
        original_ttft = read_csv_column(original_ttft_file, "ttft (ms)")

        # Plot TTFT CDF
        plt.figure(figsize=(8, 6))
        if len(original_ttft):
            sns.ecdfplot(original_ttft, label="Original", color=color_map.get("original", "black"))
        if len(remote_ttft):
            sns.ecdfplot(remote_ttft, label="Remote", color=color_map.get("remote", "red"))
        if len(local_ttft):
            sns.ecdfplot(local_ttft, label="Local", color=color_map.get("local", "blue"))
        plt.xlabel("TTFT (ms)")
        plt.ylabel("Cumulative Probability")
        plt.title("CDF of Time to First Token (TTFT)")
        plt.legend()
        plt.tight_layout()
        ttft_plot_path = os.path.join(base_dir_plots, f"ttft_cdf_comparison_duration_{duration}_rps_{rps}.png")
        plt.savefig(ttft_plot_path)
        plt.close()
        logger.info(f"TTFT CDF comparison plot saved to {ttft_plot_path}")
        

    # Plot CDF of write latency for local and remote
    def plot_write_latency_cdf(self, local_csv, remote_csv, write_plot_path, color_map):
        """
        Plots the CDF of write latency for local and remote memory locations.

        Args:
            local_csv (str): Path to the CSV file containing local write latency data.
            remote_csv (str): Path to the CSV file containing remote write latency data.
            write_plot_path (str): Path to save the CDF plot.
            color_map (dict): Dictionary mapping memory locations to colors.
        """
        # Read data
        local_df = pd.read_csv(local_csv)
        remote_df = pd.read_csv(remote_csv)

        # Extract write latencies
        local_latencies = local_df["Latency (us)"].values / 1000.0  # Convert to ms
        remote_latencies = remote_df["Latency (us)"].values / 1000.0  # Convert to ms

        # Plot using seaborn for consistent style
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.ecdfplot(local_latencies, label="Local", color=color_map["local"])
        sns.ecdfplot(remote_latencies, label="Remote", color=color_map["remote"])
        plt.xlabel("Write Latency (ms)")
        plt.ylabel("Cumulative Probability")
        plt.title("CDF of Write Latency for Local and Remote")
        plt.legend()
        plt.tight_layout()
        plt.savefig(write_plot_path)
        plt.close()

    def plot_prepare_access_cdf(self, local_prepare_csv, remote_prepare_csv, prepare_plot_path, color_map):
        """
        Plots the CDF of prepare access time for local and remote memory locations.

        Args:
            local_prepare_csv (str): Path to the CSV file containing local prepare access data.
            remote_prepare_csv (str): Path to the CSV file containing remote prepare access data.
            prepare_plot_path (str): Path to save the CDF plot.
            color_map (dict): Dictionary mapping memory locations to colors.
        """
        # Read data
        local_df = pd.read_csv(local_prepare_csv)
        remote_df = pd.read_csv(remote_prepare_csv)

        # Extract prepare access times and convert from microseconds to milliseconds
        local_times = local_df["Latency (us)"].values / 1000.0
        remote_times = remote_df["Latency (us)"].values / 1000.0

        # Plot using seaborn for consistent style
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.ecdfplot(local_times, label="Local", color=color_map["local"])
        sns.ecdfplot(remote_times, label="Remote", color=color_map["remote"])
        plt.xlabel("Prepare Access Time (ms)")
        plt.ylabel("Cumulative Probability")
        plt.title("CDF of Prepare Access Time for Local and Remote")
        plt.legend()
        plt.tight_layout()
        plt.savefig(prepare_plot_path)
        plt.close()
        
    def plot_batch_size_timeline(self, local_write_csv, remote_write_csv, output_file, color_map):
        """
        Plots the batch size timeline for local and remote memory locations.

        Args:
            local_write_csv (str): Path to the CSV file containing local write data.
            remote_write_csv (str): Path to the CSV file containing remote write data.
            output_file (str): Path to save the batch size timeline plot.
            color_map (dict): Dictionary mapping memory locations to colors.
        """
        # Read data
        local_df = pd.read_csv(local_write_csv)
        remote_df = pd.read_csv(remote_write_csv)

        # Extract batch sizes
        local_batch_sizes = local_df["Tokens Written"].values
        remote_batch_sizes = remote_df["Tokens Written"].values

        # Plot using seaborn for consistent style
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=local_df.index, y=local_batch_sizes, label="Local", color=color_map["local"])
        sns.lineplot(x=remote_df.index, y=remote_batch_sizes, label="Remote", color=color_map["remote"])
        plt.xlabel("Index")
        plt.ylabel("Batch Size")
        plt.title("Batch Size Timeline for Local and Remote Memory Locations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
    def save_throughput_csv(self, file_path: str) -> None:
        """
        Saves the throughput for each request to a CSV file.
        Args:
            file_path (str): The path to the CSV file.
        """
        if not self.raw_results:
            logger.warning("No raw results to save throughput data from.")
            return

        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Throughput"])
                for result in self.raw_results:
                    timestamp = result.get("timestamp")
                    throughput = result.get("response", {}).get("throughput")
                    if timestamp is not None and throughput is not None:
                        writer.writerow([timestamp, throughput])
            logger.info(f"Throughput data saved to {file_path}")
        except IOError as e:
            logger.error(f"Failed to save throughput data to {file_path}: {e}")
            raise