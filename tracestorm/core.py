import multiprocessing
from typing import List, Optional, Tuple

from tracestorm.logger import init_logger
from tracestorm.request_generator import generate_request
from tracestorm.result_analyzer import ResultAnalyzer
from tracestorm.trace_generator import TraceGenerator
from tracestorm.trace_player import play
from tracestorm.utils import round_robin_shard

logger = init_logger(__name__)


def run_load_test(
    trace_generator: TraceGenerator,
    model: str,
    subprocesses: int,
    base_url: str,
    api_key: str,
    datasets: List,
    sort: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple], ResultAnalyzer]:
    """
    Run load test with given configuration.

    Args:
        trace_generator: Generator for the trace
        model: Model name to test
        subprocesses: Number of subprocesses to use
        base_url: Base URL for API calls
        api_key: API key for authentication
        datasets: List of datasets to generate prompts
        sort: Sorting strategy for prompts in datasets.
        seed: Random seed for sorting.

    Returns:
        Tuple of (List of results, ResultAnalyzer instance)
    """
    raw_trace = trace_generator.generate()
    total_requests = len(raw_trace)

    if total_requests == 0:
        logger.warning("No requests to process. Trace is empty.")
        return [], ResultAnalyzer()

    requests = generate_request(
        model_name=model,
        nums=total_requests,
        datasets=datasets,
        sort=sort,
        seed=seed,
    )
    ipc_queue = multiprocessing.Queue()
    processes = []

    # Start processes
    for i, (partial_trace, partial_requests) in enumerate(
        round_robin_shard(raw_trace, requests, subprocesses), start=1
    ):
        p = multiprocessing.Process(
            target=play,
            args=(
                f"TracePlayer-{i}",
                partial_trace,
                partial_requests,
                base_url,
                api_key,
                ipc_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Collect results
    results_collected = 0
    aggregated_results = []
    while results_collected < total_requests:
        try:
            name, timestamp, resp = ipc_queue.get(timeout=30)
            results_collected += 1
            logger.info(
                f"Received result from {name} for timestamp {timestamp}: {resp['token_count']} tokens"
            )
            aggregated_results.append((name, timestamp, resp))
        except Exception as e:
            logger.error(f"Error collecting results: {str(e)}", exc_info=True)
            break

    # Wait for all processes
    for p in processes:
        p.join()

    # Analyze results
    result_analyzer = ResultAnalyzer()
    result_analyzer.store_raw_results(aggregated_results)

    return aggregated_results, result_analyzer
