import datetime
import os
from typing import Optional, Tuple

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

# Valid patterns
SYNTHETIC_PATTERNS = {"uniform", "poisson", "random"}
AZURE_PATTERNS = {"azure_code", "azure_conv"}
VALID_PATTERNS = SYNTHETIC_PATTERNS | AZURE_PATTERNS


def create_trace_generator(
    pattern: str, rps: int, duration: int, seed: Optional[int] = None
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
@click.option(
    "--rps",
    type=int,
    default=1,
    help="Requests per second (only used with synthetic patterns)",
)
@click.option(
    "--pattern",
    default="uniform",
    type=click.Choice(sorted(VALID_PATTERNS), case_sensitive=False),
    help=f"Pattern for generating trace. Valid patterns: {sorted(VALID_PATTERNS)}",
)
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
):
    """Run trace-based load testing for OpenAI API endpoints."""
    try:
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("tracestorm_results", timestamp)

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {output_dir}")

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
        result_analyzer.export_json(results_file, include_raw=True)
        logger.info(f"Raw results saved to: {results_file}")

        # Only generate plots if requested
        if plot:
            ttft_file = os.path.join(output_dir, "ttft_cdf.png")
            tpot_file = os.path.join(output_dir, "tpot_cdf.png")
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
