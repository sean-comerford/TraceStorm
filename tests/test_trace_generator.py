import random
import unittest
from math import floor
from typing import List

from trace_storm.trace_generator import generate_trace


class TestGenerateTrace(unittest.TestCase):
    def setUp(self):
        # Set a fixed seed before each test for reproducibility
        random.seed(42)

    def test_uniform_distribution(self):
        rps = 2
        duration = 3  # seconds
        pattern = "uniform"
        expected = [0, 500, 1000, 1500, 2000, 2500]
        result = generate_trace(rps, pattern, duration)
        self.assertEqual(result, expected)

    def test_zero_requests(self):
        rps = 0
        duration = 10  # seconds
        pattern = "uniform"
        expected = []
        result = generate_trace(rps, pattern, duration)
        self.assertEqual(result, expected)

    def test_zero_duration(self):
        rps = 5
        duration = 0  # seconds
        pattern = "uniform"
        expected = []
        result = generate_trace(rps, pattern, duration)
        self.assertEqual(result, expected)

    def test_unknown_pattern(self):
        rps = 1
        duration = 1
        pattern = "invalid_pattern"
        with self.assertRaises(ValueError) as context:
            generate_trace(rps, pattern, duration)
        self.assertEqual(
            str(context.exception), "Unknown pattern: invalid_pattern"
        )

    def test_negative_rps(self):
        rps = -1
        duration = 10
        pattern = "uniform"
        with self.assertRaises(ValueError) as context:
            generate_trace(rps, pattern, duration)
        self.assertEqual(
            str(context.exception), "rps must be a non-negative integer"
        )

    def test_negative_duration(self):
        rps = 1
        duration = -5
        pattern = "uniform"
        with self.assertRaises(ValueError) as context:
            generate_trace(rps, pattern, duration)
        self.assertEqual(
            str(context.exception), "duration must be a non-negative integer"
        )

    def test_non_integer_rps(self):
        rps = 2.5
        duration = 3
        pattern = "uniform"
        with self.assertRaises(ValueError) as context:
            generate_trace(rps, pattern, duration)
        self.assertEqual(
            str(context.exception), "rps must be a non-negative integer"
        )

    def test_non_integer_duration(self):
        rps = 2
        duration = 3.5
        pattern = "uniform"
        with self.assertRaises(ValueError) as context:
            generate_trace(rps, pattern, duration)
        self.assertEqual(
            str(context.exception), "duration must be a non-negative integer"
        )

    def test_uniform_distribution_edge_case(self):
        rps = 1
        duration = 1  # seconds
        pattern = "uniform"
        expected = [0]
        result = generate_trace(rps, pattern, duration)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
