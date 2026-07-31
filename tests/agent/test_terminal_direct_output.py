import json

import pytest

from agent.tool_executor import (
    extract_terminal_direct_output,
    terminal_direct_output_requested,
)
from tools.terminal_tool import TERMINAL_SCHEMA


TABLE = (
    "| Provider | Session | Weekly |\n"
    "| --- | ---: | ---: |\n"
    "| Codex | 100% | 43% |\n"
)


def _terminal_result(
    *,
    output=TABLE,
    exit_code=0,
    error=None,
):
    return json.dumps(
        {
            "output": output,
            "exit_code": exit_code,
            "error": error,
        }
    )


def test_terminal_schema_exposes_direct_response_contract():
    direct = TERMINAL_SCHEMA["parameters"]["properties"]["return_direct"]

    assert direct["type"] == "boolean"
    assert direct["default"] is False


def test_extract_terminal_direct_output_preserves_markdown_exactly():
    args = {"command": "usage --markdown", "return_direct": True}

    assert terminal_direct_output_requested("terminal", args) is True
    assert extract_terminal_direct_output(
        "terminal",
        args,
        _terminal_result(),
    ) == TABLE


@pytest.mark.parametrize(
    ("function_name", "args", "result"),
    [
        ("web_search", {"return_direct": True}, _terminal_result()),
        ("terminal", {"command": "usage --markdown"}, _terminal_result()),
        (
            "terminal",
            {"command": "usage --markdown", "return_direct": True, "background": True},
            _terminal_result(),
        ),
        (
            "terminal",
            {"command": "usage --markdown", "return_direct": True, "pty": True},
            _terminal_result(),
        ),
        (
            "terminal",
            {"command": "usage --markdown", "return_direct": True},
            _terminal_result(exit_code=1),
        ),
        (
            "terminal",
            {"command": "usage --markdown", "return_direct": True},
            _terminal_result(exit_code=False),
        ),
        (
            "terminal",
            {"command": "usage --markdown", "return_direct": True},
            _terminal_result(error="failed"),
        ),
        (
            "terminal",
            {"command": "usage --markdown", "return_direct": True},
            _terminal_result(output=" \n "),
        ),
        (
            "terminal",
            {"command": "usage --markdown", "return_direct": True},
            "not json",
        ),
    ],
)
def test_extract_terminal_direct_output_fails_closed(
    function_name,
    args,
    result,
):
    assert extract_terminal_direct_output(function_name, args, result) is None
