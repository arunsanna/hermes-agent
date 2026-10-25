"""Tests for acp_adapter.entry startup wiring."""

import acp
import pytest

from acp_adapter import entry


@pytest.mark.parametrize("failure", [None, KeyboardInterrupt, RuntimeError])
def test_main_enables_unstable_protocol(monkeypatch, failure):
    calls = {}
    ended = []
    from acp_adapter.session import SessionManager
    monkeypatch.setattr(SessionManager, "end_all_sessions", lambda self: ended.append(self))
    monkeypatch.setattr(SessionManager, "cleanup", lambda self: pytest.fail("entry must not delete sessions"))

    async def fake_run_agent(agent, **kwargs):
        from acp_adapter.server import HermesACPAgent

        assert isinstance(agent, HermesACPAgent)
        assert agent.session_manager is not None
        calls["kwargs"] = kwargs
        if failure is not None:
            raise failure("proof transport exit")

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    if failure is RuntimeError:
        with pytest.raises(SystemExit) as error:
            entry.main([])
        assert error.value.code == 1
    else:
        entry.main([])

    assert len(ended) == 1
    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_skips_configured_mcp_discovery_when_requested(monkeypatch):
    discovery_calls = []

    async def fake_run_agent(agent, **kwargs):
        pass

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setenv("HERMES_ACP_SKIP_CONFIGURED_MCP", "1")
    monkeypatch.setattr(
        "tools.mcp_tool_discovery.discover_mcp_tools",
        lambda: discovery_calls.append(True),
    )
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert discovery_calls == []










def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]










def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    import pm

    def refuse(name, **kwargs):
        raise pm.InstallError(name, "download failed")

    monkeypatch.setattr(pm, "ensure", refuse)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1


def test_setup_browser_is_one_explicit_package_request(monkeypatch):
    import pm

    calls = []
    monkeypatch.setattr(pm, "ensure", lambda name, **kwargs: calls.append((name, kwargs)))

    entry.main(["--setup-browser", "--yes"])

    assert calls == [("chromium", {"explicit": True})]
