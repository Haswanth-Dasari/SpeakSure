"""CLI default checks for the answer-end silence duration."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script_name", ["live_interview.py", "demos/capture_demo.py"])
def test_end_silence_default_and_help(script_name):
    """Both command-line tools default to 2.5 seconds and explain why."""
    script_path = PROJECT_ROOT / "scripts" / script_name
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    option_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--end-silence"
    )
    keywords = {keyword.arg: keyword.value for keyword in option_call.keywords}
    assert isinstance(keywords["default"], ast.Constant)
    assert keywords["default"].value == 2.5
    assert isinstance(keywords["help"], ast.Constant)
    assert "default: 2.5 seconds" in keywords["help"].value
    assert "time to think" in keywords["help"].value

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    assert "default: 2.5 seconds" in help_text
    assert "time to think" in help_text
