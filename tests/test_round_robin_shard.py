import unittest
from typing import Any, Dict, List, Tuple

from trace_storm.utils import round_robin_shard


class TestRoundRobinShard(unittest.TestCase):
    def test_even_sharding(self):
        trace = [1, 2, 3, 4, 5, 6]
        requests = [
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
            {"id": 5},
            {"id": 6},
        ]
        num_shards = 2
        expected = [
            ([1, 3, 5], [{"id": 1}, {"id": 3}, {"id": 5}]),
            ([2, 4, 6], [{"id": 2}, {"id": 4}, {"id": 6}]),
        ]
        result = round_robin_shard(trace, requests, num_shards)
        self.assertEqual(result, expected)

    def test_uneven_sharding(self):
        trace = [1, 2, 3, 4, 5]
        requests = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
        num_shards = 2
        expected = [
            ([1, 3, 5], [{"id": 1}, {"id": 3}, {"id": 5}]),
            ([2, 4], [{"id": 2}, {"id": 4}]),
        ]
        result = round_robin_shard(trace, requests, num_shards)
        self.assertEqual(result, expected)

    def test_single_shard(self):
        trace = [10, 20, 30]
        requests = [{"a": 1}, {"a": 2}, {"a": 3}]
        num_shards = 1
        expected = [([10, 20, 30], [{"a": 1}, {"a": 2}, {"a": 3}])]
        result = round_robin_shard(trace, requests, num_shards)
        self.assertEqual(result, expected)

    def test_num_shards_exceeds_length(self):
        trace = [100, 200]
        requests = [{"x": "a"}, {"x": "b"}]
        num_shards = 5
        with self.assertRaises(ValueError) as context:
            round_robin_shard(trace, requests, num_shards)
        self.assertEqual(
            str(context.exception),
            "Number of shards cannot exceed the number of trace/request items",
        )

    def test_empty_trace_and_requests(self):
        trace: List[int] = []
        requests: List[Dict[str, Any]] = []
        num_shards = 3
        with self.assertRaises(ValueError) as context:
            round_robin_shard(trace, requests, num_shards)
        self.assertEqual(str(context.exception), "Trace and requests are empty")

    def test_different_lengths_raises_error(self):
        trace = [1, 2, 3]
        requests = [{"id": 1}, {"id": 2}]
        num_shards = 2
        with self.assertRaises(ValueError) as context:
            round_robin_shard(trace, requests, num_shards)
        self.assertEqual(
            str(context.exception),
            "Trace and requests must have the same length",
        )

    def test_num_shards_zero_raises_error(self):
        trace = [1, 2, 3]
        requests = [{"id": 1}, {"id": 2}, {"id": 3}]
        num_shards = 0
        with self.assertRaises(ValueError) as context:
            round_robin_shard(trace, requests, num_shards)
        self.assertEqual(
            str(context.exception), "Number of shards must be at least 1"
        )

    def test_num_shards_negative_raises_error(self):
        trace = [1, 2, 3]
        requests = [{"id": 1}, {"id": 2}, {"id": 3}]
        num_shards = -2
        with self.assertRaises(ValueError) as context:
            round_robin_shard(trace, requests, num_shards)
        self.assertEqual(
            str(context.exception), "Number of shards must be at least 1"
        )

    def test_large_num_shards(self):
        trace = list(range(1, 11))  # [1,2,3,4,5,6,7,8,9,10]
        requests = [{"id": i} for i in range(1, 11)]
        num_shards = 10
        expected = [
            ([1], [{"id": 1}]),
            ([2], [{"id": 2}]),
            ([3], [{"id": 3}]),
            ([4], [{"id": 4}]),
            ([5], [{"id": 5}]),
            ([6], [{"id": 6}]),
            ([7], [{"id": 7}]),
            ([8], [{"id": 8}]),
            ([9], [{"id": 9}]),
            ([10], [{"id": 10}]),
        ]
        result = round_robin_shard(trace, requests, num_shards)
        self.assertEqual(result, expected)

    def test_non_integer_timestamps(self):
        trace = [0.5, 1.5, 2.5, 3.5]
        requests = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
        num_shards = 2
        expected = [
            ([0.5, 2.5], [{"id": "a"}, {"id": "c"}]),
            ([1.5, 3.5], [{"id": "b"}, {"id": "d"}]),
        ]
        result = round_robin_shard(trace, requests, num_shards)
        self.assertEqual(result, expected)

    def test_requests_with_additional_fields(self):
        trace = [100, 200, 300, 400]
        requests = [
            {"id": 1, "payload": "data1"},
            {"id": 2, "payload": "data2"},
            {"id": 3, "payload": "data3"},
            {"id": 4, "payload": "data4"},
        ]
        num_shards = 2
        expected = [
            (
                [100, 300],
                [{"id": 1, "payload": "data1"}, {"id": 3, "payload": "data3"}],
            ),
            (
                [200, 400],
                [{"id": 2, "payload": "data2"}, {"id": 4, "payload": "data4"}],
            ),
        ]
        result = round_robin_shard(trace, requests, num_shards)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
