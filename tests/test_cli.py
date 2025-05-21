import os
import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tracestorm.cli import create_trace_generator, main
from tracestorm.trace_generator import (
    AzureTraceGenerator,
    SyntheticTraceGenerator,
)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_create_trace_generator_synthetic(self):
        """Test creating synthetic trace generator."""
        generator, warning = create_trace_generator("uniform", 10, 60)

        self.assertIsInstance(generator, SyntheticTraceGenerator)
        self.assertEqual(warning, "")

    def test_create_trace_generator_synthetic_float_rps(self):
        """Test creating synthetic trace generator with float RPS."""
        generator, warning = create_trace_generator("uniform", 2.5, 60)

        self.assertIsInstance(generator, SyntheticTraceGenerator)
        self.assertEqual(warning, "")
        self.assertEqual(generator.rps, 2.5)

    def test_create_trace_generator_azure(self):
        """Test creating Azure trace generator."""
        generator, warning = create_trace_generator("azure_code", 10, 60)

        self.assertIsInstance(generator, AzureTraceGenerator)
        self.assertIn("RPS parameter (10) is ignored", warning)
        self.assertIn("Duration parameter (60) is ignored", warning)

    def test_create_trace_generator_azure_float_rps(self):
        """Test creating Azure trace generator with float RPS."""
        generator, warning = create_trace_generator("azure_code", 0.5, 60)

        self.assertIsInstance(generator, AzureTraceGenerator)
        self.assertIn("RPS parameter (0.5) is ignored", warning)
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

    @patch("tracestorm.cli.run_load_test")
    def test_cli_with_float_rps(self, mock_run_load_test):
        """Test CLI with float RPS value."""
        mock_analyzer = MagicMock()
        mock_run_load_test.return_value = ([], mock_analyzer)

        result = self.runner.invoke(
            main,
            [
                "--model",
                "gpt-3.5-turbo",
                "--rps",
                "0.5",
                "--pattern",
                "uniform",
                "--duration",
                "30",
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
        self.assertIn("Invalid value for '--pattern'", result.output)

    @patch("tracestorm.cli.run_load_test")
    @patch("tracestorm.cli.os.makedirs")
    @patch("tracestorm.cli.datetime")
    def test_cli_with_output_dir(
        self, mock_datetime, mock_makedirs, mock_run_load_test
    ):
        """Test CLI with output directory option."""
        mock_analyzer = MagicMock()
        mock_run_load_test.return_value = ([], mock_analyzer)
        mock_datetime.datetime.now.return_value.strftime.return_value = (
            "20240101_120000"
        )

        # Test with explicit output dir
        result = self.runner.invoke(
            main,
            [
                "--model",
                "gpt-3.5-turbo",
                "--output-dir",
                "custom_output_dir",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        mock_makedirs.assert_called_with("custom_output_dir", exist_ok=True)
        mock_analyzer.export_json.assert_called_once()

        # Reset mocks
        mock_makedirs.reset_mock()
        mock_analyzer.reset_mock()

        # Test with default output dir
        result = self.runner.invoke(
            main,
            [
                "--model",
                "gpt-3.5-turbo",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        mock_makedirs.assert_called_with(
            os.path.join("tracestorm_results", "20240101_120000"), exist_ok=True
        )
        mock_analyzer.export_json.assert_called_once()

    @patch("tracestorm.cli.run_load_test")
    @patch("tracestorm.cli.os.makedirs")
    def test_cli_with_plot_option(self, mock_makedirs, mock_run_load_test):
        """Test CLI with plot option."""
        mock_analyzer = MagicMock()
        mock_run_load_test.return_value = ([], mock_analyzer)

        # Test with plot enabled
        result = self.runner.invoke(
            main,
            [
                "--model",
                "gpt-3.5-turbo",
                "--plot",
                "--output-dir",
                "test_dir",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        mock_analyzer.plot_cdf.assert_called_once()

        # Reset mock
        mock_analyzer.reset_mock()

        # Test with plot disabled (default)
        result = self.runner.invoke(
            main,
            [
                "--model",
                "gpt-3.5-turbo",
                "--output-dir",
                "test_dir",
            ],
        )

        self.assertEqual(result.exit_code, 0)
        mock_analyzer.plot_cdf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
