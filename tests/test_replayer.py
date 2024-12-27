import asyncio
import multiprocessing
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from trace_storm.trace_player import RequestError, TracePlayer


class TestTracePlayer(unittest.TestCase):
    def setUp(self):
        # Common test data
        self.trace = [0, 500, 1000]  # ms
        self.requests = [
            {"prompt": "Test 1"},
            {"prompt": "Test 2"},
            {"prompt": "Test 3"},
        ]
        self.ipc_queue = multiprocessing.Queue()
        self.name = "test-player"
        self.base_url = "http://mock.api"
        self.api_key = "mock-api-key"

    def test_basic_run(self):
        """Test that the TracePlayer processes all items."""
        player = TracePlayer(
            self.name,
            self.trace,
            self.requests,
            self.base_url,
            self.api_key,
            self.ipc_queue,
        )

        # Mock the `request` method to avoid real network calls
        async def mock_request(_):
            return {
                "result": "ok",
                "token_count": 1,
                "time_records": [time.time()],
                "error": None,
            }

        player.request = mock_request

        # Run the player asynchronously
        asyncio.run(player.run(concurrency=2))

        # Verify all messages are consumed
        results = []
        while not self.ipc_queue.empty():
            results.append(self.ipc_queue.get_nowait())

        # We expect exactly 3 results (since trace has length 3)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(r, tuple) for r in results))

    def test_concurrency_zero_raises_error(self):
        """Test that concurrency=0 raises a ValueError."""
        player = TracePlayer(
            self.name,
            self.trace,
            self.requests,
            self.base_url,
            self.api_key,
            self.ipc_queue,
        )
        with self.assertRaises(ValueError):
            asyncio.run(player.run(concurrency=0))

    def test_mismatched_trace_and_requests_raises(self):
        """Test that mismatched lists in the constructor raises ValueError."""
        with self.assertRaises(ValueError):
            TracePlayer(
                self.name,
                [0, 100],  # 2 items
                [{"prompt": "Test"}],  # 1 item
                self.base_url,
                self.api_key,
                self.ipc_queue,
            )

    def test_request_error_handling(self):
        """Test that exceptions during request are caught and an error is returned in the result."""
        # Setup
        player = TracePlayer(
            self.name,
            self.trace,
            self.requests,
            self.base_url,
            self.api_key,
            self.ipc_queue,
        )

        # Make a mock that raises an exception to simulate a request error
        async def mock_request_raising(*args, **kwargs):
            raise RequestError("Simulated error")

        player.request = (
            mock_request_raising  # Override the real request method
        )

        # Run
        asyncio.run(player.run(concurrency=2))

        # Collect results
        results = []
        while not self.ipc_queue.empty():
            results.append(self.ipc_queue.get_nowait())

        # We had 3 items in trace, so we expect 3 results, each with an error
        self.assertEqual(len(results), 3, "Expected 3 results in the queue.")

        for r in results:
            # Each result is a tuple: (self.name, timestamp, response_dict)
            response_dict = r[2]
            # 'error' should match our simulated exception message
            self.assertIn(
                "error",
                response_dict,
                "Response dict should contain an 'error' field.",
            )
            self.assertEqual(
                response_dict["error"],
                "Simulated error",
                "The 'error' field should match our simulated exception message.",
            )


if __name__ == "__main__":
    unittest.main()
