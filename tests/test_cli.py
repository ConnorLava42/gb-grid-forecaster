"""The command-line tool loads and lists every command."""

from typer.testing import CliRunner

from gridforecast.cli import app


def test_help_lists_every_command():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "build", "backtest", "compare", "train", "forecast", "score"):
        assert command in result.output
