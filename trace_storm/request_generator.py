from typing import Any, Dict, List

from trace_storm.constants import DEFAULT_MESSAGES


def generate_request(
    model_name: str, nums: int, messages: str = DEFAULT_MESSAGES
) -> List[Dict[str, Any]]:
    requests = []
    for _ in range(nums):
        requests.append(
            {
                "model": model_name,
                "messages": [{"role": "user", "content": messages}],
                "stream": True,
            }
        )
    return requests
