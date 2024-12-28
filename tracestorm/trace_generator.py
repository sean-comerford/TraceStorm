import math
import random
from typing import List


def generate_trace(rps: int, pattern: str, duration: int) -> List[int]:
    """
    Generates a list of timestamps (ms) at which requests should be sent.

    Args:
        rps (int): Requests per second. Must be non-negative.
        pattern (str): Distribution pattern ('uniform', 'random', 'poisson', etc.).
        duration (int): Total duration in seconds. Must be non-negative.

    Returns:
        List[int]: Sorted list of timestamps in milliseconds.

    Raises:
        ValueError: If an unknown pattern is provided or if inputs are invalid.
    """
    if not isinstance(rps, int) or rps < 0:
        raise ValueError("rps must be a non-negative integer")

    if not isinstance(duration, int) or duration < 0:
        raise ValueError("duration must be a non-negative integer")

    total_requests = rps * duration
    total_duration_ms = duration * 1000
    timestamps = []

    if total_requests == 0:
        return timestamps

    if pattern == "uniform":
        # Distribute requests evenly across the duration
        interval = total_duration_ms / total_requests
        current_time = 0.0
        for _ in range(total_requests):
            timestamp = int(round(current_time))
            timestamp = min(timestamp, total_duration_ms - 1)
            timestamps.append(timestamp)
            current_time += interval
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return sorted(timestamps)
