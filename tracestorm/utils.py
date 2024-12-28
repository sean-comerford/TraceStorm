import os
from typing import Any, Dict, List, Tuple


def round_robin_shard(
    trace: List[int], requests: List[Dict[str, Any]], num_shards: int
) -> List[Tuple[List[int], List[Dict[str, Any]]]]:
    """
    Round-robin sharding of the trace and requests.

    Args:
        trace (List[int]): The original trace, a list of sorted timestamps in ms.
        requests (List[Dict[str, Any]]): The list of request objects corresponding to each timestamp.
        num_shards (int): The number of shards to divide the trace and requests into.

    Returns:
        List[Tuple[List[int], List[Dict[str, Any]]]]: A list containing tuples of (trace_shard, requests_shard) for each shard.

    Raises:
        ValueError: If `num_shards` is less than 1.
        ValueError: If `trace` and `requests` do not have the same length.
        ValueError: If `num_shards` exceeds the length of `trace` and `requests`.
        ValueError: If both `trace` and `requests` are empty.
    """
    if num_shards < 1:
        raise ValueError("Number of shards must be at least 1")

    if len(trace) != len(requests):
        raise ValueError("Trace and requests must have the same length")

    if len(trace) == 0:
        raise ValueError("Trace and requests are empty")

    if num_shards > len(trace):
        raise ValueError(
            "Number of shards cannot exceed the number of trace/request items"
        )

    shards = []
    for i in range(num_shards):
        trace_shard = trace[i::num_shards]
        requests_shard = requests[i::num_shards]
        shards.append((trace_shard, requests_shard))

    return shards


def get_unique_file_path(file_path: str) -> str:
    """
    Generates a unique file path by appending an index if the file already exists.

    Args:
        file_path (str): The desired file path.

    Returns:
        str: A unique file path with an appended index if necessary.
    """
    if not os.path.exists(file_path):
        return file_path

    base, extension = os.path.splitext(file_path)
    index = 1
    new_file_path = f"{base}_{index}{extension}"
    while os.path.exists(new_file_path):
        index += 1
        new_file_path = f"{base}_{index}{extension}"
    return new_file_path
