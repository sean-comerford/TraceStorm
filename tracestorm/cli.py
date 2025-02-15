import os
from typing import Tuple

import click

from tracestorm.core import run_load_test
from tracestorm.logger import init_logger
from tracestorm.trace_generator import (
    AzureTraceGenerator,
    SyntheticTraceGenerator,
    TraceGenerator,
)
from tracestorm.data_loader import load_datasets

logger = init_logger(__name__)

# Valid patterns
SYNTHETIC_PATTERNS = {"uniform", "poisson", "random"}
AZURE_PATTERNS = {"azure_code", "azure_conv"}
VALID_PATTERNS = SYNTHETIC_PATTERNS | AZURE_PATTERNS


def create_trace_generator(
    pattern: str, rps: int, duration: int, seed: int
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
        return SyntheticTraceGenerator(rps, pattern, duration, seed), warning_msg

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
    "--datasets-config-file", 
    default=None, 
    help="Config file for datasets"
)

def main(model, rps, pattern, duration, seed, subprocesses, base_url, api_key, datasets_config_file):
    """Run trace-based load testing for OpenAI API endpoints."""
    try:
        trace_generator, warning_msg = create_trace_generator(
            pattern, rps, duration, seed
        )
        if warning_msg:
            logger.warning(warning_msg)

        if datasets_config_file is None:
            datasets = []
            sort = None
        else: 
            datasets, sort = load_datasets(datasets_config_file)
            
        _, result_analyzer = run_load_test(
            trace_generator=trace_generator,
            model=model,
            subprocesses=subprocesses,
            base_url=base_url,
            api_key=api_key,
            datasets=datasets,
            sort=sort,
            seed=seed
        )

        print(result_analyzer)
        result_analyzer.plot_cdf()

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise click.UsageError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
