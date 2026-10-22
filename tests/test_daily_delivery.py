"""Offline regression tests for acknowledged, ambiguous and interrupted sends."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.core import daily_delivery as delivery


@pytest.fixture
def identity():
    return {"repository": "owner/repo", "workflow_id": 123,
            "workflow_path": ".github/workflows/00-daily-analysis.yml", "branch": "main",
            "run_id": "20", "run_attempt": 1, "event_name": "workflow_dispatch", "as_of": "2026-10-21",
            "target_fingerprint": delivery.feishu_target_fingerprint("https://example.test/noncredential")}


@pytest.fixture
def receipt(tmp_path, monkeypatch, identity):
    delivery._GUARDS.clear()
    for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_REPOSITORY"):
        monkeypatch.delenv(key, raising=False)
    _, path = delivery.create_delivery_files(tmp_path, identity, delivery.COMPONENTS)
    monkeypatch.setenv("DAILY_DELIVERY_FILE", str(path))
    return path


def acknowledged():
    delivery.record_daily_post("ack")
    return True


def load(path):
    return json.loads(Path(path).read_text())


def test_opt_in_keeps_other_notifications_unchanged(monkeypatch):
    monkeypatch.delenv("DAILY_DELIVERY_FILE", raising=False)
    assert delivery.daily_delivery_kwargs(("stocks",)) == {}
    assert delivery.run_daily_delivery("unused", ("stocks",), lambda: True)
    delivery.record_daily_post("unknown")
    assert not delivery.daily_post_is_uncertain()


def test_ack_is_not_complete_until_phase_finishes(receipt, identity):
    target = identity["target_fingerprint"]
    assert delivery.run_daily_delivery(target, ("stocks",), acknowledged)
    state = load(receipt)["components"]["stocks"]
    assert state["posts"] == {"ack": 1, "rejected": 0, "unknown": 0}
    assert not state["finalized"]
    delivery.finish_daily_components("stocks")
    assert delivery.finalize_daily_delivery()
    assert delivery.run_daily_delivery(target, ("stocks",), lambda: pytest.fail("must skip confirmed report"))


def test_single_stock_calls_share_component_until_finished(receipt, identity):
    for _ in range(3):
        assert delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), acknowledged)
    delivery.finish_daily_components("stocks")
    assert load(receipt)["components"]["stocks"]["posts"]["ack"] == 3


@pytest.mark.parametrize("fail_first", [False, True])
def test_pre_dispatch_failure_and_other_stock_ack_is_partial(receipt, identity, fail_first):
    if fail_first:
        delivery.record_daily_component_failure("stocks")
    assert delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), acknowledged) is (not fail_first)
    if not fail_first:
        delivery.record_daily_component_failure("stocks")
    delivery.finish_daily_components("stocks")
    state = load(receipt)["components"]["stocks"]
    assert state["status"] == "partial" and state["failed_calls"] == 1 and state["calls"] == 2
    assert state["posts"] == {"ack": 1, "rejected": 0, "unknown": 0}
    assert not delivery.finalize_daily_delivery()


def test_pre_dispatch_failure_without_post_is_retryable_but_not_green(receipt):
    delivery.record_daily_component_failure("stocks")
    delivery.finish_daily_components("stocks")
    state = load(receipt)["components"]["stocks"]
    assert state["status"] == "not_attempted" and state["failed_calls"] == 1
    assert not any(state["posts"].values())
    assert not delivery.finalize_daily_delivery()
    from scripts.daily_delivery import _component_evidence
    assert _component_evidence(state) == "allow"


def test_phase_without_post_can_reopen_for_merged_notification(receipt, identity):
    delivery.finish_daily_components("stocks", "market")
    assert delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks", "market"), acknowledged)
    delivery.finish_daily_components("stocks", "market")
    assert delivery.finalize_daily_delivery()


def test_partial_chunk_send_blocks_automatic_replay(receipt, identity):
    def partial():
        delivery.record_daily_post("ack")
        delivery.record_daily_post("rejected")
        return False

    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), partial)
    assert load(receipt)["components"]["stocks"]["status"] == "partial"
    delivery.finish_daily_components("stocks")
    assert not delivery.finalize_daily_delivery()
    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), lambda: pytest.fail("no replay"))


def test_timeout_stays_unknown_even_if_a_fallback_claims_success(receipt, identity):
    def ambiguous():
        delivery.record_daily_post("unknown")
        assert delivery.daily_post_is_uncertain()
        delivery.record_daily_post("ack")
        return True

    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("market",), ambiguous)
    delivery.finish_daily_components("market")
    assert load(receipt)["components"]["market"]["status"] == "unknown"
    assert not delivery.daily_post_is_uncertain()


def test_explicit_rejection_then_successful_fallback_is_acknowledged(receipt, identity):
    def fallback():
        delivery.record_daily_post("rejected")
        assert not delivery.daily_post_is_uncertain()
        return acknowledged()

    assert delivery.run_daily_delivery(identity["target_fingerprint"], ("market",), fallback)
    delivery.finish_daily_components("market")
    assert delivery.finalize_daily_delivery()
    assert load(receipt)["components"]["market"]["failed_calls"] == 0


@pytest.mark.parametrize("first_ack", [False, True])
def test_failed_logical_call_is_not_erased_by_another_stock_ack(receipt, identity, first_ack):
    def rejected_stock():
        delivery.record_daily_post("rejected")  # Card rejected.
        delivery.record_daily_post("rejected")  # Text also rejected.
        return False

    callbacks = (acknowledged, rejected_stock) if first_ack else (rejected_stock, acknowledged)
    results = [delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), callback)
               for callback in callbacks]
    assert not results[-1]
    delivery.finish_daily_components("stocks")
    state = load(receipt)["components"]["stocks"]
    assert state["status"] == "partial" and state["calls"] == 2 and state["failed_calls"] == 1
    assert not delivery.finalize_daily_delivery()


def test_all_rejected_remains_retryable_but_reports_failure(receipt, identity):
    def rejected():
        delivery.record_daily_post("rejected")
        return False

    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), rejected)
    delivery.finish_daily_components("stocks")
    assert not delivery.finalize_daily_delivery()
    # A different logical call may be the next stock, not a retry of the first.
    # It must not erase the earlier missing delivery in this phase.
    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), acknowledged)
    delivery.finish_daily_components("stocks")
    assert not delivery.finalize_daily_delivery()
    assert load(receipt)["components"]["stocks"]["status"] == "partial"


def test_callback_exception_is_unknown_and_red(receipt, identity):
    def crash():
        raise RuntimeError("a private provider message must not enter receipt")

    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), crash)
    assert "private" not in receipt.read_text()
    delivery.finish_daily_components("stocks")
    assert not delivery.finalize_daily_delivery()


def test_callback_boolean_does_not_prove_webhook_acceptance(receipt, identity):
    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), lambda: True)
    delivery.finish_daily_components("stocks")
    assert not delivery.finalize_daily_delivery()


def test_no_post_failure_can_retry_but_failed_attempt_is_visible(receipt, identity):
    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), lambda: False)
    assert load(receipt)["components"]["stocks"]["status"] == "not_attempted"
    assert not delivery.finalize_daily_delivery()
    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), acknowledged)


@pytest.mark.parametrize("change", ["target", "undeclared", "mixed"])
def test_refusals_cannot_leave_a_green_terminal_receipt(receipt, identity, change):
    target, components = identity["target_fingerprint"], ("stocks",)
    if change == "target":
        target = "0" * 64
    elif change == "undeclared":
        data = load(receipt)
        del data["components"]["stocks"]
        delivery.atomic_write_json(receipt, data)
    else:
        data = load(receipt)
        data["components"]["stocks"]["prior"] = "delivered"
        delivery.atomic_write_json(receipt, data)
        components = ("stocks", "market")
    assert not delivery.run_daily_delivery(target, components, lambda: pytest.fail("refused send"))
    assert load(receipt)["refused"]
    assert not delivery.finalize_daily_delivery()


def test_finalizer_does_not_seal_an_interrupted_ack_as_delivered(receipt, identity):
    assert delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), acknowledged)
    assert not delivery.finalize_daily_delivery()
    assert load(receipt)["components"]["stocks"]["status"] == "unknown"


def test_persisted_active_call_stays_unknown_after_restart(receipt):
    data = load(receipt)
    data["components"]["stocks"].update(status="active_call", active_call=True, calls=1)
    delivery.atomic_write_json(receipt, data)
    assert not delivery.finalize_daily_delivery()
    assert load(receipt)["components"]["stocks"]["status"] == "unknown"


def test_other_run_cannot_use_receipt(receipt, identity, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "21")
    assert not delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), lambda: pytest.fail("wrong owner"))


def test_thread_local_outcomes_and_shared_guard_serialize_calls(receipt, identity):
    active = 0
    peak = 0
    lock = threading.Lock()

    def send():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        delivery.record_daily_post("ack")
        with lock:
            active -= 1
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), send), range(8)))
    assert all(results) and peak == 1
    assert load(receipt)["components"]["stocks"]["posts"]["ack"] == 8
    assert not delivery.daily_post_is_uncertain()


@pytest.mark.parametrize("mutation", [
    lambda data: data["identity"].update(as_of="20261021"),
    lambda data: data["identity"].update(run_attempt=True),
    lambda data: data["components"]["stocks"].update(calls=-1),
    lambda data: data["components"]["stocks"].update(status="acknowledged"),
    lambda data: data["components"]["stocks"].update(failed_calls=1),
    lambda data: data["components"]["stocks"].update(status="active_call", active_call=True, finalized=True),
])
def test_invalid_schema_cannot_be_sending_evidence(receipt, mutation):
    data = load(receipt)
    mutation(data)
    with pytest.raises(ValueError):
        delivery.validate_receipt(data)


def test_target_hash_uses_effective_destination_only():
    one = delivery.feishu_target_fingerprint("https://example.test/a", "unused-app", "open_id", "unused-chat", "lark")
    assert one == delivery.feishu_target_fingerprint("https://example.test/a")
    assert one != delivery.feishu_target_fingerprint("https://example.test/b")
    assert delivery.feishu_target_fingerprint("", "app", "chat_id", "group", "lark") != delivery.feishu_target_fingerprint(
        "", "app", "chat_id", "group", "feishu")
