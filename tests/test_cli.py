import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tracestorm.cli import create_trace_generator, main
from tracestorm.trace_base import AzureTraceGenerator, SyntheticTraceGenerator


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_create_trace_generator_synthetic(self):
        """Test creating synthetic trace generator."""
        generator, warning = create_trace_generator("uniform", 10, 60)

        self.assertIsInstance(generator, SyntheticTraceGenerator)
        self.assertEqual(warning, "")

    def test_create_trace_generator_azure(self):
        """Test creating Azure trace generator."""
        generator, warning = create_trace_generator("azure_code", 10, 60)

        self.assertIsInstance(generator, AzureTraceGenerator)
        self.assertIn("RPS parameter (10) is ignored", warning)
        self.assertIn("Duration parameter (60) is ignored", warning)

    def test_create_trace_generator_invalid(self):
        """Test creating generator with invalid pattern."""
        with self.assertRaises(ValueError):
            create_trace_generator("invalid_pattern", 10, 60)

    @patch("tracestorm.cli.run_load_test")
    def test_cli_basic(self, mock_run_load_test):
        """Test basic CLI functionality."""
        mock_analyzer = MagicMock()
        mock_run_load_test.return_value = ([], mock_analyzer)

        result = self.runner.invoke(main, ["--model", "gpt-3.5-turbo"])

        self.assertEqual(result.exit_code, 0)
        mock_run_load_test.assert_called_once()

    @patch("tracestorm.cli.run_load_test")
    def test_cli_with_options(self, mock_run_load_test):
        """Test CLI with various options."""
        mock_analyzer = MagicMock()
        mock_run_load_test.return_value = ([], mock_analyzer)

        result = self.runner.invoke(
            main,
            [
                "--model",
                "gpt-3.5-turbo",
                "--rps",
                "5",
                "--pattern",
                "uniform",
                "--duration",
                "30",
                "--subprocesses",
                "2",
                "--base-url",
                "http://test.com",
                "--api-key",
                "test-key",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        mock_run_load_test.assert_called_once()

    def test_cli_invalid_pattern(self):
        """Test CLI with invalid pattern."""
        result = self.runner.invoke(
            main, ["--model", "gpt-3.5-turbo", "--pattern", "invalid"]
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid pattern", result.output)


if __name__ == "__main__":
    unittest.main()
