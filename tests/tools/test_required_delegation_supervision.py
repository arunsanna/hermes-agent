"""Behavior contracts for ACP required-delegation supervision."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _clean_registry():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _owner(session: str = "session-a", turn: str = "turn-a"):
    return SimpleNamespace(
        session_id=session,
        _current_turn_id=turn,
        _required_delegation_owner_token=f"owner:{session}:{turn}",
        platform="acp",
        _touch_activity=lambda *_args, **_kwargs: None,
    )


def _dispatch_required(
    *,
    owner=None,
    release: threading.Event | None = None,
    result=None,
    interrupt_fn=None,
    child_interrupt_fn=None,
    child_terminal_fn=None,
    child_ids=None,
    no_progress_timeout_seconds=300.0,
    in_flight_no_progress_timeout_seconds=None,
    start_children=True,
):
    owner = owner or _owner()
    release = release or threading.Event()
    result = result or {
        "results": [
            {"task_index": 0, "status": "completed", "summary": "done"}
        ],
        "total_duration_seconds": 1.0,
    }

    def runner():
        release.wait(timeout=10)
        return result

    dispatch = ad.dispatch_async_delegation_batch(
        goals=["required work"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=owner.session_id,
        parent_session_id=owner.session_id,
        parent_owner_token=owner._required_delegation_owner_token,
        parent_turn_id=owner._current_turn_id,
        runner=runner,
        interrupt_fn=interrupt_fn,
        child_interrupt_fn=child_interrupt_fn,
        child_terminal_fn=child_terminal_fn,
        child_ids=child_ids or ["sa-0-test"],
        required=True,
        max_async_children=3,
        no_progress_timeout_seconds=no_progress_timeout_seconds,
        in_flight_no_progress_timeout_seconds=in_flight_no_progress_timeout_seconds,
    )
    if start_children and dispatch.get("status") == "dispatched":
        for child_id in (child_ids or ["sa-0-test"]):
            ad.note_required_progress(
                dispatch["delegation_id"],
                child_id=child_id,
                current_tool=None,
                activity="child started",
                meaningful=False,
                state="running",
            )
    return owner, release, dispatch


def test_required_status_is_ownership_scoped_and_wait_consumes_terminal_result():
    owner, release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]

    pending = ad.required_status(owner, delegation_id)
    assert pending["status"] in {"queued", "running"}
    assert pending["required"] is True
    assert pending["consumed"] is False
    assert pending["child_ids"] == ["sa-0-test"]
    assert ad.has_unconsumed_required(owner) is True

    foreign = _owner(session="session-b", turn="turn-b")
    assert ad.required_status(foreign, delegation_id)["status"] == "not_found"
    assert ad.wait_required(foreign, delegation_id, timeout_seconds=0)["status"] == "not_found"

    still_pending = ad.wait_required(owner, delegation_id, timeout_seconds=0)
    assert still_pending["status"] in {"queued", "running"}
    assert still_pending["consumed"] is False

    release.set()
    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=5)
    assert terminal["status"] == "completed"
    assert terminal["consumed"] is False
    assert terminal["result"]["results"][0]["summary"] == "done"
    observed = []
    ad.observe_required(owner, delegation_id, observed.append)
    assert observed[0]["result"]["results"][0]["summary"] == "done"
    assert ad.has_unconsumed_required(owner) is False


# Fix 5: observe_required is a two-phase consume — append_observation (a
# real durable SQLite write) must run OUTSIDE _records_lock, never inside
# it, while still guaranteeing exactly-once consumption and a gate that
# cannot open before the write completes.


def test_observe_required_gate_stays_closed_until_append_observation_returns():
    owner, release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    release.set()
    assert ad.wait_required(owner, delegation_id, timeout_seconds=2)["terminal"] is True

    persist_started = threading.Event()
    allow_persist_finish = threading.Event()
    observed = []

    def _append(payload):
        persist_started.set()
        assert allow_persist_finish.wait(timeout=2)
        observed.append(payload)

    result_holder: dict = {}

    def _call_observe():
        result_holder["payload"] = ad.observe_required(owner, delegation_id, _append)

    t = threading.Thread(target=_call_observe)
    t.start()
    assert persist_started.wait(timeout=2)

    # Mid-persist (lock released, append_observation running): the gate
    # must read as still closed to every other caller.
    with ad._records_lock:
        record = ad._records[delegation_id]
        assert record["consumed_at"] is None
        assert record["consuming"] is True
    assert ad.has_unconsumed_required(owner) is True
    mid_persist_status = ad.required_status(owner, delegation_id)
    assert mid_persist_status["status"] == "completed"
    assert mid_persist_status["consumed"] is False

    allow_persist_finish.set()
    t.join(timeout=2)

    assert len(observed) == 1
    assert result_holder["payload"]["consumed"] is True
    with ad._records_lock:
        record = ad._records[delegation_id]
        assert record["consumed_at"] is not None
        assert record["consuming"] is False
    assert ad.has_unconsumed_required(owner) is False


def test_observe_required_concurrent_callers_consume_exactly_once():
    owner, release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    release.set()
    assert ad.wait_required(owner, delegation_id, timeout_seconds=2)["terminal"] is True

    append_calls: list = []
    append_lock = threading.Lock()

    def _append(payload):
        with append_lock:
            append_calls.append(payload)
        # Widen the race window so concurrent callers overlap.
        threading.Event().wait(0.05)

    results: list = []
    results_lock = threading.Lock()

    def _call():
        payload = ad.observe_required(owner, delegation_id, _append)
        with results_lock:
            results.append(payload)

    threads = [threading.Thread(target=_call) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=2)

    assert len(append_calls) == 1
    assert all(r.get("status") == "completed" for r in results)


def test_observe_required_append_failure_clears_claim_and_stays_consumable():
    owner, release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    release.set()
    assert ad.wait_required(owner, delegation_id, timeout_seconds=2)["terminal"] is True

    def _fail(_payload):
        raise RuntimeError("durable write failed")

    with pytest.raises(RuntimeError, match="durable write failed"):
        ad.observe_required(owner, delegation_id, _fail)

    with ad._records_lock:
        record = ad._records[delegation_id]
        assert record["consuming"] is False
        # The gate must not appear open for an observation that was never
        # durably persisted.
        assert record["consumed_at"] is None

    observed = []
    retried = ad.observe_required(owner, delegation_id, observed.append)
    assert retried["consumed"] is True
    assert len(observed) == 1


class _BoomNotAnException(BaseException):
    """A BaseException subclass that is NOT an Exception subclass — models
    SystemExit/KeyboardInterrupt delivered to the persisting thread."""


def test_observe_required_base_exception_still_clears_claim():
    """Concurrency-review LOW: the claim release must be a `finally`, not
    `except Exception` — a BaseException that isn't an Exception subclass
    (SystemExit/KeyboardInterrupt mid-write) must still release the claim,
    or the record is stuck forever (has_unconsumed_required reports it
    pending forever; stop_required_for_agent skips it forever)."""
    owner, release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    release.set()
    assert ad.wait_required(owner, delegation_id, timeout_seconds=2)["terminal"] is True

    def _fail(_payload):
        raise _BoomNotAnException("shutdown mid-write")

    with pytest.raises(_BoomNotAnException):
        ad.observe_required(owner, delegation_id, _fail)

    with ad._records_lock:
        record = ad._records[delegation_id]
        assert record["consuming"] is False
        assert record["consumed_at"] is None

    # The record must not be permanently stuck: has_unconsumed_required
    # still reports pending (correct — not yet durably persisted), and a
    # retry with a working callback succeeds and consumes exactly once.
    assert ad.has_unconsumed_required(owner) is True
    observed = []
    retried = ad.observe_required(owner, delegation_id, observed.append)
    assert retried["consumed"] is True
    assert len(observed) == 1
    assert ad.has_unconsumed_required(owner) is False


def test_stop_required_skips_a_record_with_an_in_flight_consuming_claim():
    owner, release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    release.set()
    assert ad.wait_required(owner, delegation_id, timeout_seconds=2)["terminal"] is True

    persist_started = threading.Event()
    allow_persist_finish = threading.Event()

    def _append(_payload):
        persist_started.set()
        assert allow_persist_finish.wait(timeout=2)

    t = threading.Thread(
        target=ad.observe_required, args=(owner, delegation_id, _append)
    )
    t.start()
    assert persist_started.wait(timeout=2)

    # STOP races the in-flight consuming claim. It must skip this record
    # entirely rather than stamp consumed_at ahead of the durable write.
    stopped = ad.stop_required_for_agent(owner, "STOP")
    assert stopped == 0
    with ad._records_lock:
        record = ad._records[delegation_id]
        assert record["consumed_at"] is None
        assert record["consuming"] is True

    allow_persist_finish.set()
    t.join(timeout=2)
    with ad._records_lock:
        assert ad._records[delegation_id]["consumed_at"] is not None
    # The in-flight consume owns terminal consumption; a second stop call
    # now sees a fully consumed record and does nothing further.
    assert ad.stop_required_for_agent(owner, "STOP") == 0


def test_required_cancel_is_idempotent_and_supports_owned_child_target():
    interrupts: list[str | None] = []
    child_ids = ["sa-0-test", "sa-1-test"]

    def interrupt(child_id=None):
        interrupts.append(child_id)

    owner, release, dispatch = _dispatch_required(
        interrupt_fn=lambda: interrupt(None),
        child_interrupt_fn=lambda child_id: interrupt(child_id),
        child_ids=child_ids,
    )
    delegation_id = dispatch["delegation_id"]

    first = ad.cancel_required(owner, delegation_id, child_id="sa-1-test")
    second = ad.cancel_required(owner, delegation_id, child_id="sa-1-test")
    assert first["status"] == "running"
    assert second["status"] == "running"
    assert {
        child["child_id"]: child["status"]
        for child in first["children"]
    }["sa-1-test"] == "cancelled"
    assert interrupts == ["sa-1-test"]

    assert ad.cancel_required(
        owner, delegation_id, child_id="sa-foreign"
    )["status"] == "not_found"

    whole = ad.cancel_required(owner, delegation_id)
    repeated = ad.cancel_required(owner, delegation_id)
    assert whole["status"] == "cancelled"
    assert whole["terminal"] is True
    assert repeated["status"] == "cancelled"
    assert interrupts == ["sa-1-test", None]

    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=5)
    assert terminal["status"] == "cancelled"
    release.set()


def test_required_owner_survives_same_turn_session_rotation():
    owner, release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]

    owner.session_id = "session-after-compression"
    assert ad.required_status(owner, delegation_id)["status"] in {
        "queued",
        "running",
    }
    cancelled = ad.cancel_required(owner, delegation_id)
    assert cancelled["status"] == "cancelled"
    assert cancelled["terminal"] is True
    assert ad.has_unconsumed_required(owner) is True
    observed = []
    ad.observe_required(owner, delegation_id, observed.append)
    assert observed[0]["status"] == "cancelled"
    assert ad.has_unconsumed_required(owner) is False
    release.set()


def test_targeted_cancel_preserves_completed_sibling_evidence():
    owner, release, dispatch = _dispatch_required(
        child_ids=["child-complete", "child-stuck"],
        child_interrupt_fn=lambda _child_id: None,
        interrupt_fn=lambda: None,
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_child_terminal(
        "child-complete",
        status="completed",
        activity="finished",
        result={
            "task_index": 0,
            "child_id": "child-complete",
            "status": "completed",
            "summary": "useful sibling evidence",
        },
    )

    cancelled = ad.cancel_required(
        owner, delegation_id, child_id="child-stuck"
    )
    assert cancelled["status"] == "cancelled"
    assert cancelled["terminal"] is True
    result_by_id = {
        item["child_id"]: item
        for item in cancelled["result"]["results"]
    }
    assert result_by_id["child-complete"]["summary"] == (
        "useful sibling evidence"
    )
    assert result_by_id["child-stuck"]["status"] == "cancelled"
    release.set()


def test_required_completion_never_enters_legacy_completion_queue(monkeypatch):
    published = []
    fake_registry = SimpleNamespace(
        completion_queue=SimpleNamespace(put=lambda event: published.append(event))
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry", fake_registry
    )

    owner, release, dispatch = _dispatch_required()
    release.set()
    terminal = ad.wait_required(
        owner, dispatch["delegation_id"], timeout_seconds=5
    )

    assert terminal["status"] == "completed"
    assert published == []
    assert ad.running_for_session(owner.session_id) == []


def test_required_progress_tracks_liveness_separately_from_meaningful_progress():
    owner, release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]

    start = ad.required_status(owner, delegation_id)
    generation = start["progress_generation"]
    meaningful_at = start["last_meaningful_at"]

    ad.note_required_progress(
        delegation_id,
        child_id="sa-0-test",
        current_tool="terminal",
        activity="still waiting",
        meaningful=False,
    )
    liveness = ad.required_status(owner, delegation_id)
    assert liveness["last_liveness_at"] >= start["last_liveness_at"]
    assert liveness["last_meaningful_at"] == meaningful_at
    assert liveness["progress_generation"] == generation

    ad.note_required_progress(
        delegation_id,
        child_id="sa-0-test",
        current_tool="read_file",
        activity="opened next file",
        meaningful=True,
    )
    progressed = ad.required_status(owner, delegation_id)
    assert progressed["last_meaningful_at"] >= meaningful_at
    assert progressed["progress_generation"] == generation + 1
    release.set()


def test_agent_wait_notices_advance_liveness_not_meaningful_generation():
    from run_agent import AIAgent

    now = __import__("time").time()
    agent = SimpleNamespace(
        _last_activity_ts=now,
        _last_activity_desc="initial",
        _last_meaningful_activity_ts=now,
        _last_meaningful_activity_desc="initial",
        _meaningful_progress_generation=7,
        thinking_callback=None,
    )
    agent._touch_activity = (
        lambda desc, meaningful=False: AIAgent._touch_activity(
            agent, desc, meaningful=meaningful
        )
    )

    AIAgent._emit_wait_notice(agent, "waiting on provider heartbeat")
    assert agent._last_activity_desc == "waiting on provider heartbeat"
    assert agent._meaningful_progress_generation == 7
    assert agent._last_meaningful_activity_desc == "initial"

    AIAgent._touch_activity(agent, "real response bytes", meaningful=True)
    assert agent._meaningful_progress_generation == 8
    assert agent._last_meaningful_activity_desc == "real response bytes"


def test_activity_touch_defaults_to_liveness_only():
    from run_agent import AIAgent

    now = __import__("time").time()
    agent = SimpleNamespace(
        _last_activity_ts=now,
        _last_activity_desc="initial",
        _last_meaningful_activity_ts=now,
        _last_meaningful_activity_desc="initial",
        _meaningful_progress_generation=4,
    )

    AIAgent._touch_activity(agent, "concurrent tools still running")
    assert agent._last_activity_desc == "concurrent tools still running"
    assert agent._meaningful_progress_generation == 4
    assert agent._last_meaningful_activity_desc == "initial"

    AIAgent._touch_activity(
        agent, "tool result received", meaningful=True
    )
    assert agent._meaningful_progress_generation == 5
    assert agent._last_meaningful_activity_desc == "tool result received"


def test_repeated_liveness_pulses_cannot_prevent_required_timeout():
    # current_tool="hung_tool" below puts this child under the wider
    # in-flight ceiling (state-aware timeouts) — set it tiny too so the
    # test still proves liveness pulses can't move the deadline.
    owner, release, dispatch = _dispatch_required(
        no_progress_timeout_seconds=0.03,
        in_flight_no_progress_timeout_seconds=0.03,
    )
    delegation_id = dispatch["delegation_id"]

    # Simulate the concurrent-tool 30s poll pulse at a much tighter cadence.
    # It keeps the session live but cannot move the semantic deadline.
    for _ in range(8):
        ad.note_required_progress(
            delegation_id,
            child_id="sa-0-test",
            current_tool="hung_tool",
            activity="concurrent tools still running",
            meaningful=False,
            state="running",
        )
        threading.Event().wait(0.01)
        if ad.required_status(owner, delegation_id)["terminal"]:
            break

    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=1)
    assert terminal["status"] == "timeout"
    assert terminal["children"][0]["status"] == "timeout"
    release.set()


# Fix 1: state-aware timeout ceilings. A child silently inside a tool call
# (current_tool set) or with a provider API call in flight (in_flight=True)
# earns the wider in_flight_no_progress_timeout_seconds ceiling instead of
# the tight idle no_progress_timeout_seconds one — see
# _required_child_effective_timeout_locked.

_IDLE_CEILING = 0.02
_IN_FLIGHT_CEILING = 0.08


def test_silent_in_tool_child_survives_past_idle_ceiling_under_in_flight_ceiling():
    owner, release, dispatch = _dispatch_required(
        no_progress_timeout_seconds=_IDLE_CEILING,
        in_flight_no_progress_timeout_seconds=_IN_FLIGHT_CEILING,
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_progress(
        delegation_id,
        child_id="sa-0-test",
        current_tool="slow_tool",
        activity="running slow_tool",
        meaningful=True,
    )
    with ad._records_lock:
        # Push the clock past the idle ceiling but still under the in-flight
        # ceiling.
        ad._records[delegation_id]["child_supervision"]["sa-0-test"][
            "last_meaningful_at"
        ] -= (_IDLE_CEILING * 2)
    status = ad.required_status(owner, delegation_id)
    assert status["terminal"] is False
    assert status["children"][0]["current_tool"] == "slow_tool"
    release.set()


def test_in_tool_child_dies_once_it_crosses_the_in_flight_ceiling():
    owner, release, dispatch = _dispatch_required(
        no_progress_timeout_seconds=_IDLE_CEILING,
        in_flight_no_progress_timeout_seconds=_IN_FLIGHT_CEILING,
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_progress(
        delegation_id,
        child_id="sa-0-test",
        current_tool="slow_tool",
        activity="running slow_tool",
        meaningful=True,
    )
    with ad._records_lock:
        # Push the clock past BOTH ceilings.
        ad._records[delegation_id]["child_supervision"]["sa-0-test"][
            "last_meaningful_at"
        ] -= (_IN_FLIGHT_CEILING * 2)
    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=1)
    assert terminal["status"] == "timeout"
    assert terminal["children"][0]["status"] == "timeout"
    release.set()


def test_idle_child_without_tool_or_api_call_still_dies_at_idle_ceiling():
    # in_flight_no_progress_timeout_seconds is huge here to prove it is NOT
    # applied to a genuinely idle child (no current_tool, no in_flight).
    owner, release, dispatch = _dispatch_required(
        no_progress_timeout_seconds=_IDLE_CEILING,
        in_flight_no_progress_timeout_seconds=1000.0,
    )
    delegation_id = dispatch["delegation_id"]
    with ad._records_lock:
        record = ad._records[delegation_id]
        child = record["child_supervision"]["sa-0-test"]
        assert not child.get("current_tool")
        assert not child.get("in_flight_sources")
        child["last_meaningful_at"] -= (_IDLE_CEILING * 2)
    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=1)
    assert terminal["status"] == "timeout"
    release.set()


def test_in_flight_api_call_marker_earns_the_wider_ceiling_without_a_tool():
    """The in_flight marker (not just current_tool) must widen the ceiling —
    this is the provider-API-call-in-flight case (no tool involved)."""
    owner, release, dispatch = _dispatch_required(
        no_progress_timeout_seconds=_IDLE_CEILING,
        in_flight_no_progress_timeout_seconds=_IN_FLIGHT_CEILING,
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_progress(
        delegation_id,
        child_id="sa-0-test",
        current_tool=None,
        activity="dispatching streaming API call",
        meaningful=False,
        in_flight=True,
    )
    with ad._records_lock:
        child = ad._records[delegation_id]["child_supervision"]["sa-0-test"]
        assert "sa-0-test" in child.get("in_flight_sources")
        child["last_meaningful_at"] -= (_IDLE_CEILING * 2)
    status = ad.required_status(owner, delegation_id)
    assert status["terminal"] is False
    release.set()


def test_ceiling_reverts_to_idle_when_tool_call_ends():
    owner, release, dispatch = _dispatch_required(
        no_progress_timeout_seconds=_IDLE_CEILING,
        in_flight_no_progress_timeout_seconds=_IN_FLIGHT_CEILING,
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_progress(
        delegation_id,
        child_id="sa-0-test",
        current_tool="slow_tool",
        activity="running slow_tool",
        meaningful=True,
    )
    with ad._records_lock:
        ad._records[delegation_id]["child_supervision"]["sa-0-test"][
            "last_meaningful_at"
        ] -= (_IDLE_CEILING * 2)
    # Survives under the wider in-flight ceiling while the tool is running.
    assert ad.required_status(owner, delegation_id)["terminal"] is False

    # Tool completes: current_tool clears via a meaningful touch, mirroring
    # tool_executor.py's real completion touch (agent._current_tool = None
    # then agent._touch_activity(..., meaningful=True)).
    ad.note_required_progress(
        delegation_id,
        child_id="sa-0-test",
        current_tool=None,
        activity="tool completed",
        meaningful=True,
    )
    with ad._records_lock:
        child = ad._records[delegation_id]["child_supervision"]["sa-0-test"]
        assert not child.get("current_tool")
        # Same margin that survived above must now kill it — the ceiling
        # reverted to the tight idle one the instant the tool call ended.
        child["last_meaningful_at"] -= (_IDLE_CEILING * 2)
    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=1)
    assert terminal["status"] == "timeout"
    release.set()


def test_nested_sibling_finish_does_not_clobber_still_in_flight_sibling():
    """Adversarial review follow-up (MEDIUM): with nested required delegation
    (max_spawn_depth >= 2), concurrent grandchildren under one direct child
    share the SAME child_supervision slot (they inherit the same frozen
    _required_delegation_ancestor_binding). in_flight tracking must be a
    per-source set, not one shared bool — otherwise a finishing sibling's
    in_flight=False clobbers a still-genuinely-in-flight sibling's marker,
    prematurely narrowing the ceiling out from under it."""
    owner, release, dispatch = _dispatch_required(
        child_ids=["child-a"],
        no_progress_timeout_seconds=_IDLE_CEILING,
        in_flight_no_progress_timeout_seconds=_IN_FLIGHT_CEILING,
    )
    delegation_id = dispatch["delegation_id"]

    # Two concurrent grandchildren both dispatch provider calls under the
    # same direct child slot.
    ad.note_required_progress(
        delegation_id,
        child_id="child-a",
        current_tool=None,
        activity="grandchild-1 dispatching",
        meaningful=False,
        in_flight=True,
        in_flight_source="grandchild-1",
    )
    ad.note_required_progress(
        delegation_id,
        child_id="child-a",
        current_tool=None,
        activity="grandchild-2 dispatching",
        meaningful=False,
        in_flight=True,
        in_flight_source="grandchild-2",
    )
    with ad._records_lock:
        child = ad._records[delegation_id]["child_supervision"]["child-a"]
        assert child.get("in_flight_sources") == {"grandchild-1", "grandchild-2"}

    # grandchild-1 finishes and clears ONLY its own source.
    ad.note_required_progress(
        delegation_id,
        child_id="child-a",
        current_tool=None,
        activity="grandchild-1 settled",
        meaningful=False,
        in_flight=False,
        in_flight_source="grandchild-1",
    )
    with ad._records_lock:
        child = ad._records[delegation_id]["child_supervision"]["child-a"]
        # grandchild-2 is still marked in flight — the shared slot must not
        # have been cleared to empty by grandchild-1's finish.
        assert child.get("in_flight_sources") == {"grandchild-2"}
        child["last_meaningful_at"] -= (_IDLE_CEILING * 2)

    # Push the clock past the idle ceiling but under the wider in-flight
    # ceiling: must survive, because grandchild-2 is still genuinely in
    # flight — pre-fix, grandchild-1's finish would have cleared the shared
    # bool and this same push would have killed the child.
    assert ad.required_status(owner, delegation_id)["terminal"] is False

    # Now grandchild-2 also finishes — the slot is genuinely empty, and the
    # ceiling must revert to idle.
    ad.note_required_progress(
        delegation_id,
        child_id="child-a",
        current_tool=None,
        activity="grandchild-2 settled",
        meaningful=False,
        in_flight=False,
        in_flight_source="grandchild-2",
    )
    with ad._records_lock:
        child = ad._records[delegation_id]["child_supervision"]["child-a"]
        assert not child.get("in_flight_sources")
        child["last_meaningful_at"] -= (_IDLE_CEILING * 2)
    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=1)
    assert terminal["status"] == "timeout"
    release.set()


def test_real_touch_activity_derives_distinct_in_flight_source_per_descendant():
    """Same defect as above, exercised through the REAL production wiring
    (run_agent.AIAgent._touch_activity's own_source_id derivation) instead
    of calling note_required_progress with an explicit in_flight_source —
    proves the wiring, not just the controller logic, keys off each
    descendant's own _subagent_id."""
    from run_agent import AIAgent

    owner, release, dispatch = _dispatch_required(
        child_ids=["child-a"],
        no_progress_timeout_seconds=_IDLE_CEILING,
        in_flight_no_progress_timeout_seconds=_IN_FLIGHT_CEILING,
    )
    delegation_id = dispatch["delegation_id"]
    binding = (delegation_id, "child-a")
    grandchild_1 = SimpleNamespace(
        _required_delegation_ancestor_binding=binding,
        _subagent_id="grandchild-1",
        _current_tool=None,
    )
    grandchild_2 = SimpleNamespace(
        _required_delegation_ancestor_binding=binding,
        _subagent_id="grandchild-2",
        _current_tool=None,
    )

    AIAgent._touch_activity(
        grandchild_1, "dispatching", meaningful=False, in_flight=True
    )
    AIAgent._touch_activity(
        grandchild_2, "dispatching", meaningful=False, in_flight=True
    )
    with ad._records_lock:
        child = ad._records[delegation_id]["child_supervision"]["child-a"]
        assert child.get("in_flight_sources") == {"grandchild-1", "grandchild-2"}

    AIAgent._touch_activity(
        grandchild_1, "settled", meaningful=False, in_flight=False
    )
    with ad._records_lock:
        child = ad._records[delegation_id]["child_supervision"]["child-a"]
        assert child.get("in_flight_sources") == {"grandchild-2"}
        child["last_meaningful_at"] -= (_IDLE_CEILING * 2)
    assert ad.required_status(owner, delegation_id)["terminal"] is False
    release.set()


def test_child_meaningful_touch_updates_controller_before_heartbeat_boundary(monkeypatch):
    from run_agent import AIAgent

    clock = [1_000.0]
    monkeypatch.setattr(ad.time, "time", lambda: clock[0])
    owner, release, dispatch = _dispatch_required(
        no_progress_timeout_seconds=1000.0,
    )
    delegation_id = dispatch["delegation_id"]
    now = clock[0]
    child = SimpleNamespace(
        _required_delegation_id=delegation_id,
        _subagent_id="sa-0-test",
        _current_tool=None,
        _last_activity_ts=now,
        _last_activity_desc="initial",
        _last_meaningful_activity_ts=now,
        _last_meaningful_activity_desc="initial",
        _meaningful_progress_generation=0,
    )
    with ad._records_lock:
        record = ad._records[delegation_id]
        record["no_progress_timeout_seconds"] = 0.05
        supervised = record["child_supervision"]["sa-0-test"]
        supervised["last_meaningful_at"] = now - 0.045

    # Real output lands just before the old deadline. Its canonical touch must
    # synchronously move the controller clock, without waiting for heartbeat.
    AIAgent._touch_activity(
        child, "receiving assistant reasoning", meaningful=True
    )
    clock[0] += 0.015
    still_running = ad.required_status(owner, delegation_id)
    assert still_running["terminal"] is False

    # Liveness alone must not buy another semantic window.
    with ad._records_lock:
        supervised = ad._records[delegation_id][
            "child_supervision"
        ]["sa-0-test"]
        supervised["last_meaningful_at"] = clock[0] - 0.045
    AIAgent._touch_activity(
        child, "concurrent tools still running", meaningful=False
    )
    clock[0] += 0.015
    terminal = ad.required_status(owner, delegation_id)
    assert terminal["status"] == "timeout"
    release.set()


def test_nested_descendant_touch_updates_direct_child_before_deadline_boundary(monkeypatch):
    from run_agent import AIAgent

    clock = [1_000.0]
    monkeypatch.setattr(ad.time, "time", lambda: clock[0])
    owner, release, dispatch = _dispatch_required(
        child_ids=["child-a"],
        no_progress_timeout_seconds=1000.0,
    )
    delegation_id = dispatch["delegation_id"]
    now = clock[0]
    descendant = SimpleNamespace(
        _required_delegation_ancestor_binding=(
            delegation_id,
            "child-a",
        ),
        _subagent_id="child-b",
        _current_tool="read_file",
        _last_activity_ts=now,
        _last_activity_desc="initial",
        _last_meaningful_activity_ts=now,
        _last_meaningful_activity_desc="initial",
        _meaningful_progress_generation=0,
    )
    with ad._records_lock:
        record = ad._records[delegation_id]
        record["no_progress_timeout_seconds"] = 0.05
        # The descendant's _current_tool="read_file" propagates onto child-a
        # from the very first touch below, which puts it under the wider
        # in-flight ceiling (state-aware timeouts). Keep that ceiling tiny
        # too so this test still proves what it's testing: meaningful vs.
        # liveness-only propagation, not ceiling selection.
        record["in_flight_no_progress_timeout_seconds"] = 0.05
        record["child_supervision"]["child-a"][
            "last_meaningful_at"
        ] = now - 0.045

    # B produces semantic output at the edge of A's deadline. The inherited
    # binding must advance A immediately, without waiting for B's heartbeat to
    # relay through A's progress callback.
    AIAgent._touch_activity(
        descendant, "receiving assistant reasoning", meaningful=True
    )
    clock[0] += 0.015
    still_running = ad.required_status(owner, delegation_id)
    assert still_running["terminal"] is False
    assert still_running["children"][0]["child_id"] == "child-a"

    # A descendant keepalive is liveness only and cannot reset A's semantic
    # clock.
    with ad._records_lock:
        ad._records[delegation_id]["child_supervision"]["child-a"][
            "last_meaningful_at"
        ] = clock[0] - 0.045
    AIAgent._touch_activity(
        descendant, "provider heartbeat", meaningful=False
    )
    clock[0] += 0.015
    terminal = ad.required_status(owner, delegation_id)
    assert terminal["status"] == "timeout"
    release.set()


def test_nested_descendant_progress_rolls_up_without_stealing_terminal_identity():
    from tools.delegate_tool import _build_child_progress_callback

    owner, release, dispatch = _dispatch_required(
        child_ids=["child-a"],
        no_progress_timeout_seconds=1000.0,
    )
    delegation_id = dispatch["delegation_id"]
    owner._delegate_depth = 0
    owner._delegate_spinner = None
    owner.tool_progress_callback = lambda *_args, **_kwargs: None
    callback = _build_child_progress_callback(
        task_index=0,
        goal="orchestrate nested work",
        parent_agent=owner,
        task_count=1,
        subagent_id="child-a",
        parent_id=None,
        depth=0,
        model="test/model",
        toolsets=["delegation"],
        session_ref={},
    )

    before = ad.required_status(owner, delegation_id)
    generation = before["children"][0]["progress_generation"]
    meaningful_at = before["children"][0]["last_meaningful_at"]

    callback(
        "subagent.heartbeat",
        preview="B transport alive",
        subagent_id="child-b",
        meaningful=False,
        current_tool="terminal",
    )
    liveness = ad.required_status(owner, delegation_id)
    assert liveness["children"][0]["progress_generation"] == generation
    assert liveness["children"][0]["last_meaningful_at"] == meaningful_at

    callback(
        "subagent.heartbeat",
        preview="B produced tool output",
        subagent_id="child-b",
        meaningful=True,
        current_tool="terminal",
    )
    progressed = ad.required_status(owner, delegation_id)
    assert progressed["children"][0]["progress_generation"] > generation

    callback(
        "subagent.complete",
        preview="B finished",
        subagent_id="child-b",
        status="completed",
        summary="grandchild evidence",
    )
    after_descendant = ad.required_status(owner, delegation_id)
    assert after_descendant["children"][0]["status"] == "running"
    with ad._records_lock:
        stored = ad._records[delegation_id]["child_supervision"][
            "child-a"
        ]["result"]
    assert stored is None

    callback(
        "subagent.complete",
        preview="A synthesized",
        status="completed",
        summary="orchestrator evidence",
    )
    after_owner = ad.required_status(owner, delegation_id)
    assert after_owner["children"][0]["status"] == "completed"
    with ad._records_lock:
        stored = dict(
            ad._records[delegation_id]["child_supervision"][
                "child-a"
            ]["result"]
        )
    assert stored["summary"] == "orchestrator evidence"
    release.set()


def test_runner_exception_fails_closed_with_every_child_and_clears_callbacks():
    owner = _owner()
    release = threading.Event()
    interrupt_calls = []
    terminal_events = []

    def _runner():
        release.wait(timeout=2)
        raise RuntimeError("aggregate exploded")

    dispatch = ad.dispatch_async_delegation_batch(
        goals=["a", "b"],
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key=owner.session_id,
        parent_session_id=owner.session_id,
        parent_owner_token=owner._required_delegation_owner_token,
        parent_turn_id=owner._current_turn_id,
        runner=_runner,
        interrupt_fn=lambda: interrupt_calls.append("interrupt"),
        child_terminal_fn=(
            lambda child_id, status, reason: terminal_events.append(
                (child_id, status, reason)
            )
        ),
        child_ids=["child-a", "child-b"],
        required=True,
        max_async_children=3,
        no_progress_timeout_seconds=1000,
    )
    for child_id in ("child-a", "child-b"):
        ad.note_required_progress(
            dispatch["delegation_id"],
            child_id=child_id,
            current_tool=None,
            activity="started",
            meaningful=False,
            state="running",
        )
    release.set()

    terminal = ad.wait_required(
        owner, dispatch["delegation_id"], timeout_seconds=1
    )
    assert terminal["status"] == "failed"
    assert {
        item["child_id"] for item in terminal["result"]["results"]
    } == {"child-a", "child-b"}
    assert {
        item["status"] for item in terminal["result"]["results"]
    } == {"failed"}
    assert sorted(
        (child_id, status)
        for child_id, status, _reason in terminal_events
    ) == [("child-a", "failed"), ("child-b", "failed")]
    assert interrupt_calls == ["interrupt"]
    with ad._records_lock:
        record = ad._records[dispatch["delegation_id"]]
        assert record["interrupt_fn"] is None
        assert record["child_interrupt_fn"] is None
        assert record["child_terminal_fn"] is None


def test_completed_children_cannot_leave_controller_finalizing_forever():
    interrupts = []
    owner, release, dispatch = _dispatch_required(
        child_ids=["child-complete"],
        interrupt_fn=lambda: interrupts.append("interrupt"),
        no_progress_timeout_seconds=1000.0,
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_child_terminal(
        "child-complete",
        status="completed",
        activity="child complete before cleanup",
        result={
            "child_id": "child-complete",
            "status": "completed",
            "summary": "preserved evidence",
        },
    )
    finalizing = ad.required_status(owner, delegation_id)
    assert finalizing["status"] == "finalizing"
    with ad._records_lock:
        record = ad._records[delegation_id]
        record["finalization_timeout_seconds"] = 0.01
        record["finalization_started_at"] -= 1

    terminal = ad.required_status(owner, delegation_id)
    assert terminal["status"] == "timeout"
    assert terminal["result"]["finalization_timeout"] is True
    assert terminal["result"]["results"][0]["status"] == "completed"
    assert terminal["result"]["results"][0]["summary"] == (
        "preserved evidence"
    )
    assert interrupts == ["interrupt"]
    with ad._records_lock:
        record = ad._records[delegation_id]
        assert record["interrupt_fn"] is None
        assert record["child_interrupt_fn"] is None
        assert record["child_terminal_fn"] is None
    release.set()


def test_targeted_cancel_wins_over_late_batch_completion():
    owner, release, dispatch = _dispatch_required(
        child_ids=["child-a", "child-b"],
        child_interrupt_fn=lambda _child_id: None,
        interrupt_fn=lambda: None,
    )
    delegation_id = dispatch["delegation_id"]

    cancelled = ad.cancel_required(
        owner, delegation_id, child_id="child-a"
    )
    assert cancelled["terminal"] is False
    assert {
        child["child_id"]: child["status"]
        for child in cancelled["children"]
    }["child-a"] == "cancelled"

    ad._finalize_batch(
        delegation_id,
        {
            "results": [
                {
                    "task_index": 0,
                    "child_id": "child-a",
                    "status": "completed",
                    "summary": "late stale success",
                },
                {
                    "task_index": 1,
                    "child_id": "child-b",
                    "status": "completed",
                    "summary": "real sibling success",
                },
            ],
            "total_duration_seconds": 1.0,
        },
        "completed",
    )

    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=1)
    assert terminal["status"] == "completed"
    result_by_id = {
        item["child_id"]: item
        for item in terminal["result"]["results"]
    }
    assert result_by_id["child-a"]["status"] == "cancelled"
    assert result_by_id["child-a"]["summary"] is None
    assert result_by_id["child-b"]["status"] == "completed"
    assert result_by_id["child-b"]["summary"] == "real sibling success"
    release.set()


def test_worker_terminal_callback_is_enriched_by_full_aggregate_result():
    owner, release, dispatch = _dispatch_required(
        child_ids=["child-worker"],
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_child_terminal(
        "child-worker",
        status="completed",
        activity="short UI summary",
        result={
            "task_index": 0,
            "child_id": "child-worker",
            "status": "completed",
            "summary": "short UI summary",
        },
    )
    ad._finalize_batch(
        delegation_id,
        {
            "results": [{
                "task_index": 0,
                "child_id": "child-worker",
                "status": "completed",
                "summary": "full budgeted worker summary",
                "tool_trace": [{"tool": "read_file"}],
            }],
            "total_duration_seconds": 1.0,
        },
        "completed",
    )
    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=1)
    assert terminal["status"] == "completed"
    assert terminal["result"]["results"][0]["summary"] == (
        "full budgeted worker summary"
    )
    assert terminal["result"]["results"][0]["tool_trace"] == [
        {"tool": "read_file"}
    ]
    release.set()


def test_consumed_required_records_are_pruned_to_retention_cap():
    owner = _owner()
    total = ad._MAX_RETAINED_COMPLETED + 7
    for index in range(total):
        delegation_id = f"retained-{index:03d}"
        with ad._records_lock:
            ad._records[delegation_id] = {
                "delegation_id": delegation_id,
                "required": True,
                "parent_session_id": owner.session_id,
                "parent_owner_token": (
                    owner._required_delegation_owner_token
                ),
                "parent_turn_id": owner._current_turn_id,
                "status": "completed",
                "dispatched_at": float(index),
                "completed_at": float(index),
                "consumed_at": None,
                "child_ids": [],
                "child_supervision": {},
                "result": {"results": []},
                "done_event": threading.Event(),
                "interrupt_fn": lambda: None,
                "child_interrupt_fn": lambda _child_id: None,
                "child_terminal_fn": lambda *_args: None,
            }
        observed = []
        ad.observe_required(owner, delegation_id, observed.append)
        assert len(observed) == 1

    with ad._records_lock:
        retained = [
            record for record in ad._records.values()
            if record.get("required")
            and record.get("status") in ad._REQUIRED_TERMINAL_STATES
            and record.get("consumed_at") is not None
        ]
        assert len(retained) <= ad._MAX_RETAINED_COMPLETED
        assert all(
            record.get("interrupt_fn") is None
            and record.get("child_interrupt_fn") is None
            and record.get("child_terminal_fn") is None
            for record in retained
        )


def test_terminal_environment_heartbeat_is_liveness_only(monkeypatch):
    from tools.environments import base

    observed = []
    base.set_activity_callback(
        lambda label, *, meaningful=True: observed.append((label, meaningful))
    )
    state = {
        "last_touch": 0.0,
        "start": 0.0,
        "interval": 0.0,
    }
    try:
        base.touch_activity_if_due(state, "terminal command running")
    finally:
        base.set_activity_callback(None)

    assert observed
    assert observed[-1][1] is False


def test_stream_and_interim_sinks_stay_closed_until_terminal_result_consumed():
    from run_agent import AIAgent

    owner, release, dispatch = _dispatch_required()
    delivered_stream = []
    delivered_interim = []
    owner.stream_delta_callback = delivered_stream.append
    owner._stream_callback = None
    owner.interim_assistant_callback = (
        lambda text, **kwargs: delivered_interim.append(text)
    )
    owner._current_streamed_assistant_text = ""
    owner._stream_needs_break = False
    owner._stream_think_scrubber = None
    owner._stream_context_scrubber = None
    owner._stream_writer_lock = threading.Lock()
    owner._stream_writer_token = 0
    owner._stream_writer_tls = threading.local()
    owner._delivered_interim_texts = set()
    owner._has_unconsumed_required_delegations = (
        lambda: AIAgent._has_unconsumed_required_delegations(owner)
    )
    owner._deliver_scrubbed_stream_delta = (
        lambda text: AIAgent._deliver_scrubbed_stream_delta(owner, text)
    )
    owner._stream_writer_superseded = lambda: False
    owner._strip_think_blocks = lambda text: text
    owner._record_streamed_assistant_text = lambda text: None
    owner._deliver_scrubbed_stream_delta = (
        lambda text: AIAgent._deliver_scrubbed_stream_delta(owner, text)
    )
    owner._interim_assistant_visible_text = (
        lambda message: message.get("content", "")
    )
    owner._interim_text_was_delivered = lambda text: False
    owner._interim_content_was_streamed = lambda text: False
    owner._record_delivered_interim_text = lambda text: None
    owner._extract_codex_interim_visible_parts = lambda message: []

    AIAgent._fire_stream_delta(owner, "premature")
    AIAgent._emit_interim_assistant_message(
        owner, {"role": "assistant", "content": "premature interim"}
    )
    assert delivered_stream == []
    assert delivered_interim == []

    release.set()
    terminal_unconsumed = ad.required_status(
        owner, dispatch["delegation_id"]
    )
    for _ in range(100):
        if terminal_unconsumed["status"] == "completed":
            break
        threading.Event().wait(0.01)
        terminal_unconsumed = ad.required_status(
            owner, dispatch["delegation_id"]
        )
    assert terminal_unconsumed["status"] == "completed"
    assert terminal_unconsumed["consumed"] is False

    AIAgent._fire_stream_delta(owner, "still premature")
    assert delivered_stream == []

    ad.observe_required(
        owner, dispatch["delegation_id"], lambda payload: None
    )
    AIAgent._fire_stream_delta(owner, "final")
    AIAgent._emit_interim_assistant_message(
        owner, {"role": "assistant", "content": "final interim"}
    )
    assert delivered_stream == ["final"]
    assert delivered_interim == ["final interim"]


def test_acp_provisional_stream_discards_required_candidate_prose():
    from run_agent import AIAgent

    delivered = []
    owner = _owner()
    owner.stream_delta_callback = delivered.append
    owner._stream_callback = None
    owner._current_streamed_assistant_text = ""
    owner._stream_needs_break = False
    owner._stream_think_scrubber = None
    owner._stream_context_scrubber = None
    owner._stream_writer_superseded = lambda: False
    owner._has_unconsumed_required_delegations = lambda: False
    owner._strip_think_blocks = lambda text: text
    owner._record_streamed_assistant_text = lambda text: None
    owner._deliver_scrubbed_stream_delta = (
        lambda text: AIAgent._deliver_scrubbed_stream_delta(owner, text)
    )
    owner._acp_provisional_stream_active = True
    owner._acp_provisional_stream_buffer = []

    AIAgent._fire_stream_delta(owner, "premature prose")
    assert delivered == []
    AIAgent._finish_acp_provisional_stream(owner, discard=True)
    assert delivered == []

    owner._acp_provisional_stream_active = True
    AIAgent._fire_stream_delta(owner, "safe final")
    AIAgent._finish_acp_provisional_stream(owner, discard=False)
    assert delivered == ["safe final"]


def test_acp_provisional_gate_buffers_commentary_and_nonstreaming_interim():
    from run_agent import AIAgent

    delivered = []
    owner = _owner()
    owner.interim_assistant_callback = (
        lambda text, **kwargs: delivered.append((text, kwargs))
    )
    owner._has_unconsumed_required_delegations = lambda: False
    owner._strip_think_blocks = lambda text: text
    owner._delivered_interim_texts = set()
    owner._normalize_interim_visible_text = lambda text: text.strip()
    owner._interim_text_was_delivered = (
        lambda text: AIAgent._interim_text_was_delivered(owner, text)
    )
    owner._record_delivered_interim_text = (
        lambda text: AIAgent._record_delivered_interim_text(owner, text)
    )
    owner._extract_codex_interim_visible_parts = lambda _message: []
    owner._interim_assistant_visible_text = (
        lambda message: message.get("content", "")
    )
    owner._interim_content_was_streamed = lambda _text: False
    owner._fire_streamed_codex_commentary = (
        lambda text: AIAgent._fire_streamed_codex_commentary(owner, text)
    )
    owner._emit_interim_assistant_message = (
        lambda message: AIAgent._emit_interim_assistant_message(owner, message)
    )
    owner._deliver_scrubbed_stream_delta = (
        lambda text: AIAgent._deliver_scrubbed_stream_delta(owner, text)
    )
    owner._acp_provisional_stream_active = True
    owner._acp_provisional_stream_buffer = []

    AIAgent._fire_streamed_codex_commentary(
        owner, "candidate streamed commentary"
    )
    AIAgent._emit_interim_assistant_message(
        owner,
        {"role": "assistant", "content": "candidate nonstreaming interim"},
    )
    assert delivered == []
    AIAgent._finish_acp_provisional_stream(owner, discard=True)
    assert delivered == []

    owner._acp_provisional_stream_active = True
    AIAgent._fire_streamed_codex_commentary(owner, "safe commentary")
    AIAgent._emit_interim_assistant_message(
        owner,
        {"role": "assistant", "content": "safe interim"},
    )
    AIAgent._finish_acp_provisional_stream(owner, discard=False)
    assert [text for text, _kwargs in delivered] == [
        "safe commentary",
        "safe interim",
    ]


def test_acp_reasoning_stays_live_during_required_supervision():
    from run_agent import AIAgent

    delivered = []
    owner = _owner()
    owner.reasoning_callback = delivered.append
    owner._stream_writer_superseded = lambda: False
    owner._has_unconsumed_required_delegations = lambda: True
    owner._acp_provisional_stream_active = True
    owner._acp_provisional_stream_buffer = []

    AIAgent._fire_reasoning_delta(owner, "child progress is still visible")

    assert delivered == ["child progress is still visible"]
    assert owner._acp_provisional_stream_buffer == []


def test_required_candidate_sanitizes_codex_and_anthropic_replay_sidecars():
    from agent.conversation_loop import _sanitize_required_assistant_candidate

    thinking = {"type": "thinking", "thinking": "signed", "signature": "sig"}
    tool_use = {"type": "tool_use", "id": "tool-1", "name": "delegate_task"}
    message = {
        "role": "assistant",
        "content": "premature answer",
        "codex_message_items": [
            {"type": "message", "content": "premature answer"}
        ],
        "codex_reasoning_items": [{"type": "reasoning", "id": "reason-1"}],
        "tool_calls": [{"id": "tool-1"}],
        "anthropic_content_blocks": [
            thinking,
            {"type": "text", "text": "premature answer"},
            tool_use,
        ],
    }

    _sanitize_required_assistant_candidate(message)

    assert message["content"] == ""
    assert "codex_message_items" not in message
    assert message["codex_reasoning_items"] == [
        {"type": "reasoning", "id": "reason-1"}
    ]
    assert message["tool_calls"] == [{"id": "tool-1"}]
    assert message["anthropic_content_blocks"] == [thinking, tool_use]


def test_required_candidate_sanitizer_drops_api_content_sidecar():
    """The api_content sidecar replays verbatim (byte-fidelity, no
    sanitize/strip) both in the live api_messages build and on a durable
    reload via get_messages_as_conversation. Clearing `content` alone does
    not stop either path from resurrecting a rejected candidate whose sidecar
    diverges from the cleared content — the sidecar must be dropped in the
    same step, matching the existing drop_stale_api_content contract used by
    every other content-rewrite site (compaction, stale-confirmation
    redaction, historical image strip)."""
    from agent.conversation_loop import _sanitize_required_assistant_candidate

    message = {
        "role": "assistant",
        "content": "premature answer",
        "api_content": "premature answer\n\n<memory-context>leaked</memory-context>",
        "tool_calls": [{"id": "tool-1"}],
    }

    _sanitize_required_assistant_candidate(message)

    assert message["content"] == ""
    assert "api_content" not in message
    assert message["tool_calls"] == [{"id": "tool-1"}]


def test_forced_compression_is_blocked_for_unobserved_required_owner():
    from agent.conversation_compression import (
        RequiredDelegationCompressionBlocked,
        compress_context,
    )

    owner = _owner()
    owner._delegate_depth = 0
    owner._required_delegation_launching = True
    owner._has_unconsumed_required_delegations = lambda: False

    with pytest.raises(RequiredDelegationCompressionBlocked):
        compress_context(
            owner,
            [{"role": "user", "content": "keep"}],
            "system",
            force=True,
        )


def test_scrubber_reset_tail_uses_provisional_stream_gate():
    from run_agent import AIAgent

    delivered = []
    owner = _owner()
    owner.stream_delta_callback = delivered.append
    owner._stream_callback = None
    owner._current_streamed_assistant_text = "already buffered"
    owner._stream_think_scrubber = None
    owner._stream_context_scrubber = SimpleNamespace(
        flush=lambda: "tail after tool delta"
    )
    owner._stream_writer_superseded = lambda: False
    owner._has_unconsumed_required_delegations = lambda: False
    owner._record_streamed_assistant_text = lambda text: None
    owner._acp_provisional_stream_active = True
    owner._acp_provisional_stream_buffer = []
    owner._deliver_scrubbed_stream_delta = (
        lambda text: AIAgent._deliver_scrubbed_stream_delta(owner, text)
    )

    AIAgent._reset_stream_delivery_tracking(owner)

    assert delivered == []
    assert owner._acp_provisional_stream_buffer == [
        ("stream_delta", "tail after tool delta")
    ]


def test_retry_reset_discards_prior_provisional_attempt():
    from run_agent import AIAgent

    delivered = []
    owner = _owner()
    owner.stream_delta_callback = delivered.append
    owner._stream_callback = None
    owner._current_streamed_assistant_text = "stale"
    owner._stream_think_scrubber = None
    owner._stream_context_scrubber = SimpleNamespace(
        flush=lambda: "stale scrubber tail",
        feed=lambda text: text,
    )
    owner._stream_writer_superseded = lambda: False
    owner._has_unconsumed_required_delegations = lambda: False
    owner._strip_think_blocks = lambda text: text
    owner._record_streamed_assistant_text = lambda _text: None
    owner._acp_provisional_stream_active = True
    owner._acp_provisional_stream_buffer = [
        ("stream_delta", "stale first attempt")
    ]
    owner._deliver_scrubbed_stream_delta = (
        lambda text: AIAgent._deliver_scrubbed_stream_delta(owner, text)
    )

    AIAgent._reset_stream_delivery_tracking(
        owner, discard_provisional_attempt=True
    )
    assert owner._acp_provisional_stream_buffer == []

    AIAgent._fire_stream_delta(owner, "safe retry")
    AIAgent._finish_acp_provisional_stream(owner, discard=False)
    assert delivered == ["safe retry"]


def test_model_wait_tool_result_never_consumes_controller():
    owner, release, dispatch = _dispatch_required()
    release.set()
    terminal = ad.wait_required(
        owner, dispatch["delegation_id"], timeout_seconds=5
    )
    assert terminal["terminal"] is True
    assert ad.has_unconsumed_required(owner) is True

    call = SimpleNamespace(
        id="wait-call",
        function=SimpleNamespace(name="delegation_wait"),
    )
    messages = [{
        "role": "tool",
        "tool_call_id": "wait-call",
        "content": __import__("json").dumps(terminal),
    }]
    assert call.function.name == "delegation_wait"
    assert messages[0]["tool_call_id"] == call.id
    assert ad.has_unconsumed_required(owner) is True


def test_no_progress_timeout_terminalizes_and_interrupts_once():
    calls = []
    owner, _release, dispatch = _dispatch_required(
        interrupt_fn=lambda: calls.append("stop")
    )
    delegation_id = dispatch["delegation_id"]
    with ad._records_lock:
        record = ad._records[delegation_id]
        record["no_progress_timeout_seconds"] = 0.01
        record["child_supervision"]["sa-0-test"]["last_meaningful_at"] -= 1

    terminal = ad.wait_required(owner, delegation_id, timeout_seconds=0)
    assert terminal["status"] == "timeout"
    assert terminal["terminal"] is True
    assert calls == ["stop"]
    ad.wait_required(owner, delegation_id, timeout_seconds=0)
    assert calls == ["stop"]


def test_timeout_preserves_pre_timeout_diagnostics_in_result_and_logs(caplog):
    """Fix 3: _claim_required_timeout_locked overwrites last_activity/
    current_tool with the generic timeout reason — capture what the child
    was actually doing BEFORE that happens, into the result payload, and
    log it. Otherwise the last real signal from a stuck child is lost."""
    import logging

    owner, _release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    ad.note_required_progress(
        delegation_id,
        child_id="sa-0-test",
        current_tool="reading_giant_file",
        activity="reading /var/log/giant.log (attempt 3)",
        meaningful=True,
    )
    with ad._records_lock:
        record = ad._records[delegation_id]
        record["no_progress_timeout_seconds"] = 0.01
        record["in_flight_no_progress_timeout_seconds"] = 0.01
        record["child_supervision"]["sa-0-test"]["last_meaningful_at"] -= 1
        pre_timeout_last_meaningful_at = record["child_supervision"][
            "sa-0-test"
        ]["last_meaningful_at"]

    with caplog.at_level(logging.WARNING, logger="tools.async_delegation"):
        terminal = ad.wait_required(owner, delegation_id, timeout_seconds=0)

    assert terminal["status"] == "timeout"
    child_result = terminal["result"]["results"][0]
    diagnostics = child_result["diagnostics"]
    assert diagnostics["current_tool"] == "reading_giant_file"
    assert diagnostics["last_activity"] == "reading /var/log/giant.log (attempt 3)"
    assert diagnostics["last_meaningful_at"] == pre_timeout_last_meaningful_at
    # The generic timeout reason still overwrites the LIVE child state (for
    # future status reads) — diagnostics is a snapshot, not a state change.
    assert terminal["children"][0]["current_tool"] is None
    assert "sa-0-test" in caplog.text
    assert "reading_giant_file" in caplog.text


def test_watchdog_survives_a_supervision_exception_and_keeps_enforcing():
    """Fix 4: _required_watchdog must not die on an unhandled exception —
    it is the ONLY deadline enforcement when the parent never calls
    wait_required/required_status."""
    owner, release, dispatch = _dispatch_required(no_progress_timeout_seconds=0.02)
    delegation_id = dispatch["delegation_id"]

    calls = {"n": 0}
    real_supervise = ad._supervise_required_delegation

    def _flaky_supervise(deleg_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected supervision failure")
        return real_supervise(deleg_id)

    import tools.async_delegation as ad_module
    original = ad_module._supervise_required_delegation
    ad_module._supervise_required_delegation = _flaky_supervise
    try:
        with ad._records_lock:
            ad._records[delegation_id]["child_supervision"]["sa-0-test"][
                "last_meaningful_at"
            ] -= 1
        # The watchdog thread started by dispatch_async_delegation_batch
        # polls on its own; give it enough time to hit the injected
        # exception at least once and still reach the real deadline check.
        done_event = ad._records[delegation_id]["done_event"]
        assert done_event.wait(timeout=3)
    finally:
        ad_module._supervise_required_delegation = original

    assert calls["n"] >= 2
    status = ad.required_status(owner, delegation_id)
    assert status["status"] == "timeout"
    release.set()


def test_stop_terminalizes_without_model_observation():
    calls = []
    owner, _release, dispatch = _dispatch_required(
        interrupt_fn=lambda: calls.append("stop")
    )
    assert ad.stop_required_for_agent(owner, "STOP") == 1
    status = ad.required_status(owner, dispatch["delegation_id"])
    assert status["status"] == "cancelled"
    assert status["consumed"] is True
    assert ad.has_unconsumed_required(owner) is False
    assert calls == ["stop"]
    assert ad.stop_required_for_agent(owner, "STOP") == 0


def test_required_terminalization_is_monotonic_stop_then_finalize():
    owner, _release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    assert ad.stop_required_for_agent(owner, "STOP") == 1
    ad._finalize_batch(
        delegation_id,
        {"results": [{"status": "completed", "summary": "late"}]},
        "completed",
    )
    status = ad.required_status(owner, delegation_id)
    assert status["status"] == "cancelled"


def test_required_terminalization_is_monotonic_finalize_then_stop():
    owner, _release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    ad._finalize_batch(
        delegation_id,
        {"results": [{"status": "completed", "summary": "won"}]},
        "completed",
    )
    assert ad.stop_required_for_agent(owner, "STOP") == 1
    status = ad.required_status(owner, delegation_id)
    assert status["status"] == "completed"


def test_required_terminalization_is_monotonic_timeout_then_finalize():
    owner, _release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    with ad._records_lock:
        record = ad._records[delegation_id]
        record["no_progress_timeout_seconds"] = 0.01
        record["child_supervision"]["sa-0-test"]["last_meaningful_at"] -= 1
    assert ad.wait_required(owner, delegation_id)["status"] == "timeout"
    ad._finalize_batch(
        delegation_id,
        {"results": [{"status": "completed", "summary": "late"}]},
        "completed",
    )
    assert ad.required_status(owner, delegation_id)["status"] == "timeout"


def test_required_records_are_hidden_from_legacy_global_listing():
    owner, _release, _dispatch = _dispatch_required()
    assert ad.list_async_delegations() == []


def test_required_supervision_reaches_slow_and_stalled_then_progress_resets():
    owner, _release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    with ad._records_lock:
        record = ad._records[delegation_id]
        record["no_progress_timeout_seconds"] = 100
        record["child_supervision"]["sa-0-test"]["last_meaningful_at"] -= 55
    assert ad.refresh_required_supervision(delegation_id)["status"] == "slow"

    with ad._records_lock:
        ad._records[delegation_id]["child_supervision"]["sa-0-test"][
            "last_meaningful_at"
        ] -= 30
    assert ad.refresh_required_supervision(delegation_id)["status"] == "stalled"

    progressed = ad.note_required_progress(
        delegation_id,
        child_id="sa-0-test",
        current_tool="terminal",
        activity="new terminal output",
        meaningful=True,
    )
    assert progressed["status"] == "running"
    assert progressed["progress_generation"] > 0


def test_productive_same_tool_activity_resets_300_second_clock():
    from tools.delegate_tool import _build_child_progress_callback

    owner, _release, dispatch = _dispatch_required()
    delegation_id = dispatch["delegation_id"]
    owner._delegate_depth = 0
    owner._delegate_spinner = None
    owner.tool_progress_callback = None
    callback = _build_child_progress_callback(
        0, "long terminal work", owner, subagent_id="sa-0-test"
    )
    assert callback is not None

    with ad._records_lock:
        ad._records[delegation_id]["child_supervision"]["sa-0-test"][
            "last_meaningful_at"
        ] -= 299
    callback(
        "subagent.text",
        "terminal",
        "new output from the same long-running command",
    )
    status = ad.required_status(owner, delegation_id)
    assert status["status"] == "running"
    assert status["progress_generation"] >= 1
    assert ad.wait_required(owner, delegation_id, timeout_seconds=0)["status"] != "timeout"


def test_progressing_sibling_cannot_mask_stuck_child_timeout():
    terminalized = []
    interrupts = []
    owner, release, dispatch = _dispatch_required(
        child_ids=["child-a", "child-b"],
        interrupt_fn=lambda: interrupts.append("batch"),
        child_terminal_fn=lambda child_id, status, reason: terminalized.append(
            (child_id, status, reason)
        ),
    )
    delegation_id = dispatch["delegation_id"]

    with ad._records_lock:
        record = ad._records[delegation_id]
        record["no_progress_timeout_seconds"] = 100
        record["child_supervision"]["child-b"]["last_meaningful_at"] -= 55
    ad.note_required_progress(
        delegation_id,
        child_id="child-a",
        current_tool="read_file",
        activity="productive sibling advanced",
        meaningful=True,
    )
    slow = ad.required_status(owner, delegation_id)
    by_id = {child["child_id"]: child for child in slow["children"]}
    assert by_id["child-a"]["status"] == "running"
    assert by_id["child-b"]["status"] == "slow"

    with ad._records_lock:
        ad._records[delegation_id]["child_supervision"]["child-b"][
            "last_meaningful_at"
        ] -= 30
    stalled = ad.required_status(owner, delegation_id)
    by_id = {child["child_id"]: child for child in stalled["children"]}
    assert by_id["child-a"]["status"] == "running"
    assert by_id["child-b"]["status"] == "stalled"

    # More real work from A must not move B's independent deadline.
    ad.note_required_progress(
        delegation_id,
        child_id="child-a",
        current_tool="read_file",
        activity="productive sibling advanced again",
        meaningful=True,
    )
    with ad._records_lock:
        ad._records[delegation_id]["child_supervision"]["child-b"][
            "last_meaningful_at"
        ] -= 20
    timed_out = ad.required_status(owner, delegation_id)
    assert timed_out["status"] == "timeout"
    assert timed_out["terminal"] is True
    assert interrupts == ["batch"]
    assert [item[:2] for item in terminalized] == [
        ("child-a", "interrupted"),
        ("child-b", "timeout"),
    ]
    result_by_id = {
        item["child_id"]: item
        for item in ad._records[delegation_id]["result"]["results"]
    }
    assert result_by_id["child-a"]["status"] == "interrupted"
    assert result_by_id["child-b"]["status"] == "timeout"
    release.set()


def test_queued_sibling_does_not_accrue_running_no_progress_timeout():
    owner, release, dispatch = _dispatch_required(
        child_ids=["child-running", "child-queued"],
        start_children=False,
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_progress(
        delegation_id,
        child_id="child-running",
        current_tool="terminal",
        activity="first child started",
        meaningful=False,
        state="running",
    )
    with ad._records_lock:
        record = ad._records[delegation_id]
        record["no_progress_timeout_seconds"] = 10
        record["start_timeout_seconds"] = 1_000
        record["dispatched_at"] -= 100
        record["child_supervision"]["child-queued"][
            "last_meaningful_at"
        ] -= 100

    ad.note_required_progress(
        delegation_id,
        child_id="child-running",
        current_tool="terminal",
        activity="first child made real progress",
        meaningful=True,
    )
    snapshot = ad.required_status(owner, delegation_id)
    by_id = {
        child["child_id"]: child for child in snapshot["children"]
    }
    assert snapshot["terminal"] is False
    assert by_id["child-running"]["status"] == "running"
    assert by_id["child-queued"]["status"] == "queued"
    release.set()


def test_queued_child_has_own_start_deadline_after_sibling_completes():
    interrupts = []
    owner, release, dispatch = _dispatch_required(
        child_ids=["child-finished", "child-never-started"],
        start_children=False,
        interrupt_fn=lambda: interrupts.append("batch"),
    )
    delegation_id = dispatch["delegation_id"]
    ad.note_required_progress(
        delegation_id,
        child_id="child-finished",
        current_tool=None,
        activity="first child started",
        meaningful=False,
        state="running",
    )
    ad.note_required_child_terminal(
        "child-finished",
        status="completed",
        activity="first child completed",
        result={
            "child_id": "child-finished",
            "status": "completed",
            "summary": "useful evidence",
        },
    )
    with ad._records_lock:
        record = ad._records[delegation_id]
        record["start_timeout_seconds"] = 10
        record["child_supervision"]["child-never-started"][
            "dispatched_at"
        ] -= 100

    snapshot = ad.required_status(owner, delegation_id)
    by_id = {
        child["child_id"]: child for child in snapshot["children"]
    }
    assert snapshot["status"] == "timeout"
    assert by_id["child-finished"]["status"] == "completed"
    assert by_id["child-never-started"]["status"] == "timeout"
    assert interrupts == ["batch"]
    result_by_id = {
        item["child_id"]: item
        for item in ad._records[delegation_id]["result"]["results"]
    }
    assert result_by_id["child-finished"]["summary"] == "useful evidence"
    assert result_by_id["child-never-started"]["status"] == "timeout"
    release.set()


def test_required_watchdog_times_out_without_wait_or_status_poll():
    terminal = threading.Event()
    interrupts = []
    callbacks = []

    owner, release, dispatch = _dispatch_required(
        interrupt_fn=lambda: interrupts.append("ignored"),
        child_terminal_fn=lambda child_id, status, reason: (
            callbacks.append((child_id, status, reason)),
            terminal.set(),
        ),
        no_progress_timeout_seconds=0.05,
        start_children=False,
    )
    delegation_id = dispatch["delegation_id"]

    assert terminal.wait(timeout=1), "watchdog never enforced no-progress deadline"
    with ad._records_lock:
        record = ad._records[delegation_id]
        assert record["status"] == "timeout"
        assert record["done_event"].is_set()
    assert interrupts == ["ignored"]
    assert callbacks[0][0:2] == ("sa-0-test", "timeout")
    release.set()


def test_timeout_child_terminal_callback_precedes_done_and_runs_once():
    observed = []
    holder = {}

    def terminalize(child_id, status, reason):
        observed.append(
            (
                child_id,
                status,
                holder["done_event"].is_set(),
                reason,
            )
        )

    owner, release, dispatch = _dispatch_required(
        interrupt_fn=lambda: None,
        child_terminal_fn=terminalize,
    )
    delegation_id = dispatch["delegation_id"]
    with ad._records_lock:
        record = ad._records[delegation_id]
        holder["done_event"] = record["done_event"]
        record["no_progress_timeout_seconds"] = 0.01
        record["child_supervision"]["sa-0-test"]["last_meaningful_at"] -= 1

    assert ad.wait_required(owner, delegation_id)["status"] == "timeout"
    assert len(observed) == 1
    assert observed[0][0:3] == ("sa-0-test", "timeout", False)
    assert "timed out" in observed[0][3].lower()
    assert holder["done_event"].is_set()

    # A late worker return/finalizer cannot overwrite timeout or re-fire the
    # controller-owned terminal callback.
    ad._finalize_batch(
        delegation_id,
        {"results": [{"task_index": 0, "status": "completed"}]},
        "completed",
    )
    assert len(observed) == 1
    assert ad.required_status(owner, delegation_id)["status"] == "timeout"
    release.set()
