"""A managed worker's agent gets a RuntimeSessionStore as its session_db. Every SessionDB method
the agent core calls must exist on the store, accept the same keywords, and answer like the
owner's SessionDB (Adolanium reg-1 was one missing keyword; the sweep found two missing reads:
the failed-turn boundary's tail role and in-place compaction's held-row liveness)."""
import ast
import copy
import inspect
from pathlib import Path
import re

import pytest

from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, mutate_worker_execution, register_worker_execution

ROOT = Path(__file__).resolve().parents[2]
# The agent core: everything a worker's AIAgent runs (tools open their own handles, or refuse).
CORE = [ROOT / 'run_agent.py', *sorted((ROOT / 'agent').glob('*.py'))]


def _core_session_db_calls():
    """Names called on the agent's store: ``x._session_db`` / ``x.db`` / ``session_db``, and a local
    ``db`` where the module binds it from ``_session_db`` (a module opening its own SessionDB is not)."""
    names = set()
    for path in CORE:
        text = path.read_text(encoding='utf-8')
        local_db = re.search(r'\bdb\b[^=\n]*=\s*getattr\([^)\n]*_session_db', text) is not None
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if ((isinstance(owner, ast.Attribute) and owner.attr in ('_session_db', 'db'))
                        or (isinstance(owner, ast.Name) and (owner.id == 'session_db' or (owner.id == 'db' and local_db)))):
                    names.add(node.func.attr)
    return sorted(n for n in names if hasattr(SessionDB, n))


@pytest.mark.parametrize('name', _core_session_db_calls())
def test_store_offers_every_core_session_db_call_with_its_keywords(name):
    assert hasattr(RuntimeSessionStore, name), f'RuntimeSessionStore lacks {name}'
    store = inspect.signature(getattr(RuntimeSessionStore, name)).parameters
    if any(p.kind is p.VAR_KEYWORD for p in store.values()):
        return
    owner = inspect.signature(getattr(SessionDB, name)).parameters
    missing = [k for k, p in owner.items() if k not in store and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    assert missing == [], f'RuntimeSessionStore.{name} rejects {missing}'


def test_tail_reads_answer_like_the_owner(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('s', 'cli')
        db.append_messages_batch('s', [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'},
                                       {'role': 'user', 'content': 'failed turn'}])
        db.create_session('other', 'cli')
        db.append_messages_batch('other', [{'role': 'assistant', 'content': 'x'}])
        rows = [r[0] for r in db._read_all('SELECT id FROM messages ORDER BY id')]
        epoch = begin_runtime_epoch(db, instance_id='fixture')
        scope = dict(epoch=epoch, execution_id='w', session_id='s', generation=0)
        register_worker_execution(db, **scope, kind='compute', adoption_secret='secret')
        store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope, tmp_path / 'outbox')
        try:
            assert store.latest_conversation_role('s') == db.latest_conversation_role('s') == 'user'
            for row_id in [*rows, 10**6]:  # another transcript's row is not this session's
                assert store.get_message_role('s', row_id) == db.get_message_role('s', row_id)
        finally:
            store.close()
    finally:
        db.close()


@pytest.fixture
def worker_store(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('s', 'cli')
    db.create_session('other', 'cli')
    epoch = begin_runtime_epoch(db, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='s', generation=0)
    register_worker_execution(db, **scope, kind='compute', adoption_secret='fixture')
    calls = []

    def rpc(method, **params):
        calls.append(copy.deepcopy(params))
        return mutate_worker_execution(db, **params)

    store = RuntimeSessionStore(rpc, scope, tmp_path / 'outbox')
    try:
        yield db, store, calls
    finally:
        try:
            if store.failure:
                with pytest.raises(WorkerPersistenceError, match=re.escape(store.failure)):
                    store.close()
            else:
                store.close()
        finally:
            db.close()


def test_atomic_observation_rolls_back_and_replays_without_mutating_input(worker_store, monkeypatch):
    db, store, calls = worker_store
    db.append_messages_batch('s', [{'role': 'user', 'content': 'question'}])
    pair = [
        {'role': 'assistant', 'content': None, '_required_delegation_observation': True, 'tool_calls': [
            {'id': 'required', 'type': 'function', 'function': {'name': 'observe', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'required', 'content': 'observed', '_required_delegation_observation': True},
    ]
    original = copy.deepcopy(pair)
    before = db.get_session('s')

    def fail_after_insert(*args, **kwargs):
        raise RuntimeError('injected before owner commit')

    with monkeypatch.context() as patch:
        patch.setattr(db, '_bump_session_counters', fail_after_insert)
        with pytest.raises(RuntimeError, match='injected before owner commit'):
            store.append_messages_atomic('s', pair)
    assert [m['role'] for m in db.get_messages('s')] == ['user']
    assert pair == original
    after = db.get_session('s')
    assert (after['message_count'], after['runtime_revision']) == (before['message_count'], before['runtime_revision'])
    assert db._read_all('SELECT sequence FROM worker_receipts') == []
    assert store.retry_pending()[0]['count'] == 2
    assert [m['role'] for m in db.get_messages('s')] == ['user', 'assistant', 'tool']
    assert pair == original
    assert mutate_worker_execution(db, **calls[-1])['count'] == 2
    assert len(db.get_messages('s')) == 3
    with pytest.raises(WorkerPersistenceError, match='permission_denied'):
        store.append_messages_atomic('other', pair)


def test_tool_result_cas_is_owned_active_and_receipt_idempotent(worker_store):
    db, store, calls = worker_store
    for sid in ['s', 'other']:
        db.append_messages_batch(sid, [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'tool', 'type': 'function', 'function': {'name': 'observe', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'tool', 'content': 'pixels', 'api_content': 'sidecar'},
        ])
    assert store.rewrite_message_content('s', 'tool', 'stub', expected_content='wrong') is False
    assert db.get_messages('s')[-1]['content'] == 'pixels'
    assert store.rewrite_message_content('s', 'tool', 'stub', expected_content='pixels') is True
    receipt = calls[-1]
    assert mutate_worker_execution(db, **receipt) == {'value': True}
    assert store.rewrite_message_content('s', 'tool', 'again', expected_content='pixels') is False
    assert db.get_messages('s')[-1]['content'] == 'stub'
    assert db.get_messages('other')[-1]['content'] == 'pixels'
    with db._read_ctx() as conn:
        row = conn.execute("SELECT api_content FROM messages WHERE session_id='s' AND role='tool'").fetchone()
    assert row[0] is None
    with pytest.raises(WorkerPersistenceError, match='permission_denied'):
        store.rewrite_message_content('other', 'tool', 'stolen', expected_content='pixels')
    with pytest.raises(RuntimeStoreError):
        mutate_worker_execution(db, **dict(receipt, session_id='other'))
    db._execute_write(lambda conn: conn.execute("UPDATE messages SET active=0 WHERE session_id='s' AND role='tool'"))
    assert store.rewrite_message_content('s', 'tool', 'inactive', expected_content='stub') is False
    db.append_message('s', 'user', 'pixels', tool_call_id='not-tool')
    assert store.rewrite_message_content('s', 'not-tool', 'stub', expected_content='pixels') is False
    assert db.get_messages('s')[-1]['content'] == 'pixels'


def test_atomic_append_inserts_owned_row_snapshots_and_replays_lost_ack(worker_store):
    db, store, calls = worker_store
    db.append_messages_atomic('s', [{'role': 'user', 'content': 'original'}])
    row_id = db._read_all("SELECT id FROM messages WHERE session_id='s'")[0][0]
    group = [{'role': 'user', 'content': 'original', '_row_id': row_id,
              '_required_delegation_observation': True}]
    original = copy.deepcopy(group)
    rpc = store.rpc
    lost = []

    def lose_first_ack(method, **params):
        result = rpc(method, **params)
        if not lost:
            lost.append(True)
            raise TimeoutError('lost committed acknowledgement')
        return result

    store.rpc = lose_first_ack
    with pytest.raises(TimeoutError, match='lost committed acknowledgement'):
        store.append_messages_atomic('s', group)
    assert len(db.get_messages('s')) == 2
    assert group == original
    assert store.retry_pending()[0]['count'] == 1
    assert len(db.get_messages('s')) == 2
    assert group == original
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        mutate_worker_execution(db, **dict(calls[-1], payload={'messages': []}))
    with pytest.raises(RuntimeStoreError):
        mutate_worker_execution(db, **dict(calls[-1], generation=99))


def test_cas_preserves_collisions_and_replays_success_after_a_later_writer(worker_store):
    db, store, calls = worker_store
    db.append_messages_atomic('s', [
        {'role': 'tool', 'tool_call_id': 'collision', 'content': content, 'api_content': 'pixels'}
        for content in ['same', 'different', 'same']
    ])
    before = db.get_session('s')['runtime_revision']
    assert store.rewrite_message_content('s', 'collision', 'stub', expected_content='wrong') is False
    assert db.get_session('s')['runtime_revision'] == before
    assert store.rewrite_message_content('s', 'collision', 'stub', expected_content='same') is True
    receipt = calls[-1]
    rows = db._read_all("SELECT content,api_content FROM messages WHERE session_id='s' ORDER BY id")
    assert [tuple(row) for row in rows] == [('stub', None), ('different', 'pixels'), ('stub', None)]
    db._execute_write(lambda conn: conn.execute("UPDATE messages SET content='later' WHERE content='stub'"))
    assert mutate_worker_execution(db, **receipt) == {'value': True}
    assert [row[0] for row in db._read_all("SELECT content FROM messages WHERE session_id='s' ORDER BY id")] == [
        'later', 'different', 'later']
    before = db.get_session('s')['runtime_revision']
    assert store.rewrite_message_content('s', 'collision', 'stale', expected_content='same') is False
    assert db.get_session('s')['runtime_revision'] == before
    bad = dict(calls[-1], sequence=calls[-1]['sequence'] + 1,
               payload=dict(calls[-1]['payload'], session_id='other'))
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        mutate_worker_execution(db, **bad)


@pytest.mark.parametrize('bad_row', [
    {'role': 'invalid', 'content': 'bad'},
    {'role': 'tool', 'content': 'foreign'},
])
def test_atomic_append_rejects_the_whole_invalid_or_foreign_group(worker_store, bad_row):
    db, store, _calls = worker_store
    if bad_row['role'] == 'tool':
        db.append_message('other', 'tool', 'foreign')
        bad_row = dict(bad_row, _row_id=db._read_all("SELECT id FROM messages WHERE session_id='other'")[0][0])
    with pytest.raises(RuntimeStoreError):
        store.append_messages_atomic('s', [{'role': 'user', 'content': 'first'}, bad_row])
    assert db.get_messages('s') == []
    assert db.get_session('s')['message_count'] == 0
    assert db._read_all('SELECT sequence FROM worker_receipts') == []
