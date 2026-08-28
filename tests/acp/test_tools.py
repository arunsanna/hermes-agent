"""Tests for acp_adapter.tools — tool kind mapping and ACP content building."""


from acp_adapter.edit_approval import EditProposal
from acp_adapter.tools import (
    TOOL_KIND_MAP,
    build_tool_complete,
    build_tool_start,
    build_tool_title,
    extract_locations,
    get_tool_kind,
    make_tool_call_id,
)
from acp.schema import (
    FileEditToolCallContent,
    ContentToolCallContent,
    ToolCallLocation,
    ToolCallStart,
    ToolCallProgress,
)


# ---------------------------------------------------------------------------
# TOOL_KIND_MAP coverage
# ---------------------------------------------------------------------------


COMMON_HERMES_TOOLS = ["read_file", "search_files", "terminal", "patch", "write_file", "process"]


class TestToolKindMap:
    def test_all_hermes_tools_have_kind(self):
        """Every common hermes tool should appear in TOOL_KIND_MAP."""
        for tool in COMMON_HERMES_TOOLS:
            assert tool in TOOL_KIND_MAP, f"{tool} missing from TOOL_KIND_MAP"

    def test_tool_kind_read_file(self):
        assert get_tool_kind("read_file") == "read"

    def test_tool_kind_terminal(self):
        assert get_tool_kind("terminal") == "execute"








    def test_unknown_tool_returns_other_kind(self):
        assert get_tool_kind("nonexistent_tool_xyz") == "other"


# ---------------------------------------------------------------------------
# make_tool_call_id
# ---------------------------------------------------------------------------


class TestMakeToolCallId:
    def test_returns_string(self):
        tc_id = make_tool_call_id()
        assert isinstance(tc_id, str)

    def test_starts_with_tc_prefix(self):
        tc_id = make_tool_call_id()
        assert tc_id.startswith("tc-")

    def test_ids_are_unique(self):
        ids = {make_tool_call_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# build_tool_title
# ---------------------------------------------------------------------------


class TestBuildToolTitle:
    def test_terminal_title_includes_command(self):
        title = build_tool_title("terminal", {"command": "ls -la /tmp"})
        assert "ls -la /tmp" in title

    def test_terminal_title_truncates_long_command(self):
        long_cmd = "x" * 200
        title = build_tool_title("terminal", {"command": long_cmd})
        assert len(title) < 120
        assert "..." in title

    def test_read_file_title(self):
        title = build_tool_title("read_file", {"path": "/etc/hosts"})
        assert "/etc/hosts" in title


    def test_search_title(self):
        title = build_tool_title("search_files", {"pattern": "TODO"})
        assert "TODO" in title




    def test_skill_view_title_includes_skill_name(self):
        title = build_tool_title("skill_view", {"name": "github-pitfalls"})
        assert title == "skill view (github-pitfalls)"


    def test_execute_code_title_includes_first_code_line(self):
        title = build_tool_title("execute_code", {"code": "\nfrom hermes_tools import terminal\nprint('done')"})
        assert title == "python: from hermes_tools import terminal"


    def test_unknown_tool_uses_name(self):
        title = build_tool_title("some_new_tool", {"foo": "bar"})
        assert title == "some_new_tool"


# ---------------------------------------------------------------------------
# build_tool_start
# ---------------------------------------------------------------------------


class TestBuildToolStart:
    def test_build_tool_start_for_patch(self):
        """patch start should not duplicate the edit-approval diff."""
        args = {
            "path": "src/main.py",
            "old_string": "print('hello')",
            "new_string": "print('world')",
        }
        result = build_tool_start("tc-1", "patch", args)
        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        assert len(result.content) >= 1
        item = result.content[0]
        assert isinstance(item, ContentToolCallContent)
        assert "Approval prompt shows the diff" in item.content.text
        assert "src/main.py" in item.content.text


    def test_auto_approved_edit_start_shows_diff_content(self):
        """Auto-approved edit starts need the diff because no approval card exists."""
        args = {"path": "/tmp/acp.txt", "old_string": "old", "new_string": "new"}
        result = build_tool_start(
            "tc-auto-edit",
            "patch",
            args,
            edit_diff=EditProposal("patch", "/tmp/acp.txt", "old\n", "new\n", args),
        )

        assert isinstance(result, ToolCallStart)
        assert result.kind == "edit"
        assert len(result.content) == 1
        item = result.content[0]
        assert isinstance(item, FileEditToolCallContent)
        assert item.path == "/tmp/acp.txt"
        assert item.old_text == "old\n"
        assert item.new_text == "new\n"







    def test_build_tool_start_for_browser_navigate(self):
        """browser_navigate should emit a polished start event."""
        args = {"url": "https://x.com"}
        result = build_tool_start("tc-browser-start", "browser_navigate", args)
        assert isinstance(result, ToolCallStart)
        assert result.title == "navigate: https://x.com"
        assert result.kind == "fetch"
        assert result.content[0].content.text == '{\n  "url": "https://x.com"\n}'
        assert result.raw_input is None








# ---------------------------------------------------------------------------
# build_tool_complete
# ---------------------------------------------------------------------------


class TestBuildToolComplete:
    def test_build_tool_complete_for_terminal(self):
        """Completed terminal call should include output text."""
        result = build_tool_complete("tc-2", "terminal", "total 42\ndrwxr-xr-x 2 root root 4096 ...")
        assert isinstance(result, ToolCallProgress)
        assert result.status == "completed"
        assert len(result.content) >= 1
        content_item = result.content[0]
        assert isinstance(content_item, ContentToolCallContent)
        assert "total 42" in content_item.content.text
        assert result.raw_output is None








    def test_build_tool_complete_marks_returncode_nonzero_as_failed(self):
        result = build_tool_complete("tc-fail", "execute_code", '{"output": "bad", "returncode": 2}')
        assert result.status == "failed"

    def test_build_tool_complete_preserves_delegation_attempt_raw_output(self):
        payload = {
            "error": "Task 0 output_schema invalid: expected object",
            "delegationAttempt": {
                "id": "attempt-preflight-1",
                "state": "preflight_rejected",
                "requestedCount": 1,
                "spawnedCount": 0,
                "failure": {
                    "phase": "preflight",
                    "code": "invalid_output_schema",
                    "retryable": False,
                },
            },
        }
        result = build_tool_complete(
            "tc-preflight",
            "delegate_task",
            '{"error":"Task 0 output_schema invalid: expected object",'
            '"delegationAttempt":{"id":"attempt-preflight-1",'
            '"state":"preflight_rejected","requestedCount":1,'
            '"spawnedCount":0,"failure":{"phase":"preflight",'
            '"code":"invalid_output_schema","retryable":false}}}',
        )

        assert result.status == "failed"
        assert result.raw_output == payload
        assert "Delegation failed" in result.content[0].content.text






    def test_build_tool_complete_keeps_background_dispatch_in_progress(self):
        """delegate_task(background=True) only *dispatched* the batch; the child
        agents keep running and re-enter later as their own frames. The dispatch
        tool_call must stay in_progress so clients do not show every subagent as
        Done ~1s after dispatch.
        """
        result = build_tool_complete(
            "tc-bg",
            "delegate_task",
            '{"status": "dispatched", "mode": "background", "count": 2, '
            '"delegation_id": "abc123", "goals": ["a", "b"]}',
        )
        assert result.status == "in_progress"

    def test_build_tool_complete_marks_synchronous_delegate_completed(self):
        """The sync / pool-at-capacity paths return aggregated child results
        (not status=dispatched), so they must complete normally.
        """
        result = build_tool_complete(
            "tc-sync",
            "delegate_task",
            '{"results": [{"goal": "a", "output": "done"}]}',
        )
        assert result.status == "completed"

    def test_build_tool_complete_ignores_dispatched_without_background_mode(self):
        """Only the accepted-async case (mode=background) stays in_progress; a
        foreground dispatch ack must still complete.
        """
        result = build_tool_complete(
            "tc-fg",
            "delegate_task",
            '{"status": "dispatched", "mode": "foreground"}',
        )
        assert result.status == "completed"

    def test_build_tool_complete_background_marker_scoped_to_delegate_task(self):
        """The in_progress carve-out is delegate_task-only; a same-shaped result
        from another tool must not be held open.
        """
        result = build_tool_complete(
            "tc-other",
            "terminal",
            '{"status": "dispatched", "mode": "background"}',
        )
        assert result.status == "completed"

    def test_build_tool_complete_for_skill_manage_summarizes_without_raw_json(self):
        result = build_tool_complete(
            "tc-skill-manage",
            "skill_manage",
            '{"success":true,"message":"Patched references/hermes-acp-zed-rendering.md in skill \'hermes-agent-operations\' (1 replacement)."}',
            function_args={
                "action": "patch",
                "name": "hermes-agent-operations",
                "file_path": "references/hermes-acp-zed-rendering.md",
            },
        )
        text = result.content[0].content.text
        assert "**✅ Skill updated**" in text
        assert "`patch`" in text
        assert "`hermes-agent-operations`" in text
        assert "references/hermes-acp-zed-rendering.md" in text
        assert "{\"success\"" not in text
        assert result.raw_output is None


    def test_build_tool_complete_for_search_files_formats_matches(self):
        result = build_tool_complete(
            "tc-search",
            "search_files",
            '{"total_count":2,"matches":[{"path":"README.md","line":3,"content":"TODO: fix this"},{"path":"src/app.py","line":9,"content":"needle"}],"truncated":true}\n\n[Hint: Results truncated. Use offset=12 to see more.]',
        )
        text = result.content[0].content.text
        assert "Search results" in text
        assert "Found 2 matches" in text
        assert "README.md:3" in text
        assert "TODO: fix this" in text
        assert "Results truncated" in text
        assert result.raw_output is None







    def test_build_tool_complete_generically_formats_unknown_json_dict_without_raw_output(self):
        result = build_tool_complete(
            "tc-recall-search",
            "memory_archive_search",
            '{"results":[{"id":"obs-1","status":"active","content":"Recall should render as a readable summary."}],"trust":"lower-trust archive evidence"}',
        )
        text = result.content[0].content.text
        assert "memory_archive_search result" in text
        assert "lower-trust archive evidence" in text
        assert "Recall should render as a readable summary" in text
        assert "{\"results\"" not in text
        assert result.raw_output is None









# ---------------------------------------------------------------------------
# extract_locations
# ---------------------------------------------------------------------------


class TestExtractLocations:
    def test_extract_locations_with_path(self):
        args = {"path": "src/app.py", "offset": 42}
        locs = extract_locations(args)
        assert len(locs) == 1
        assert isinstance(locs[0], ToolCallLocation)
        assert locs[0].path == "src/app.py"
        assert locs[0].line == 42

    def test_extract_locations_without_path(self):
        args = {"command": "echo hi"}
        locs = extract_locations(args)
        assert locs == []
