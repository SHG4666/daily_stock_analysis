"""Exercise durable daily receipts through the real Feishu transport boundary.

All outbound requests are mocked; acknowledgements come from HTTP/SDK response
objects rather than a mocked notification-service success flag.
"""
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

from src.core import daily_delivery
from src.notification_sender.feishu_sender import FeishuSender


SENDER_MODULE = "src.notification_sender.feishu_sender"


def config(**overrides):
    values = {"feishu_webhook_url": "https://feishu.example/hook", "feishu_max_bytes": 20000}
    values.update(overrides)
    return SimpleNamespace(**values)


def response(status=200, body=None):
    result = mock.Mock(status_code=status, text="response body")
    result.json.return_value = {"code": 0} if body is None else body
    return result


def sdk_response(status=200, code=0):
    return SimpleNamespace(
        raw=SimpleNamespace(status_code=status), code=code, msg="response",
        success=lambda: code == 0, get_log_id=lambda: "test-log-id",
    )


def app_sender(**overrides):
    return FeishuSender(config(
        feishu_webhook_url=None, feishu_app_id="app", feishu_app_secret="secret",
        feishu_chat_id="chat", **overrides,
    ))


def app_client(*results):
    create = mock.Mock(side_effect=results)
    return SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        message=SimpleNamespace(create=create),
    ))), create


@pytest.fixture(autouse=True)
def isolate_daily_environment(monkeypatch):
    monkeypatch.delenv("DAILY_DELIVERY_FILE", raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setattr(daily_delivery, "_GUARDS", {})


@pytest.fixture
def prepare(tmp_path, monkeypatch):
    files = []

    def create(cfg=None, *, as_of="2026-10-21", prior=None):
        cfg = cfg or config()
        identity = {
            "repository": "owner/repo", "workflow_id": 123,
            "workflow_path": ".github/workflows/00-daily-analysis.yml", "branch": "main",
            "run_id": "1", "run_attempt": 1, "event_name": "workflow_dispatch", "as_of": as_of,
            "target_fingerprint": daily_delivery.feishu_target_fingerprint(
                getattr(cfg, "feishu_webhook_url", None), getattr(cfg, "feishu_app_id", ""),
                getattr(cfg, "feishu_receive_id_type", "chat_id"), getattr(cfg, "feishu_chat_id", ""),
                getattr(cfg, "feishu_domain", "feishu"),
            ),
        }
        _, receipt = daily_delivery.create_delivery_files(
            tmp_path / str(len(files)), identity, ("stocks", "market", "document"), prior=prior,
        )
        files.append(receipt)
        monkeypatch.setenv("DAILY_DELIVERY_FILE", str(receipt))
        return receipt

    return create


def state(path, component="stocks"):
    return json.loads(path.read_text(encoding="utf-8"))["components"][component]


def send_daily(sender, content="daily report", **kwargs):
    return sender.send_to_feishu(content, daily_components=("stocks",), **kwargs)


def test_no_components_preserves_legacy_fallback_even_if_receipt_file_is_configured(tmp_path, monkeypatch):
    path = tmp_path / "must-not-open.json"
    monkeypatch.setenv("DAILY_DELIVERY_FILE", str(path))
    with mock.patch(f"{SENDER_MODULE}.requests.post", side_effect=[response(503), response()]) as post:
        assert FeishuSender(config()).send_to_feishu("ad-hoc report")
    assert post.call_count == 2
    assert not path.exists()


def test_disabled_guard_keeps_the_legacy_webhook_call_arguments(monkeypatch):
    monkeypatch.delenv("DAILY_DELIVERY_FILE", raising=False)
    sender = FeishuSender(config())
    with mock.patch.object(sender, "_send_via_webhook", return_value=True) as send:
        assert sender.send_to_feishu("daily report", timeout_seconds=7, daily_components=("stocks",))
    send.assert_called_once_with("daily report", timeout_seconds=7)


def test_unscoped_app_network_error_keeps_retry_behaviour(monkeypatch):
    monkeypatch.delenv("DAILY_DELIVERY_FILE", raising=False)
    sender = app_sender()
    client, create = app_client(requests.Timeout("timeout"), sdk_response())
    with mock.patch.object(sender, "_ensure_app_client", return_value=client), \
         mock.patch(f"{SENDER_MODULE}.time.sleep") as sleep:
        assert sender.send_to_feishu("ad-hoc report")
    assert create.call_count == 2
    sleep.assert_called_once()


def test_acknowledged_phase_survives_new_sender_and_receipt_reload(prepare, monkeypatch):
    receipt = prepare()
    with mock.patch(f"{SENDER_MODULE}.requests.post", return_value=response(body={"StatusCode": 0})) as post:
        assert send_daily(FeishuSender(config()), timeout_seconds=4)
        daily_delivery.finish_daily_components("stocks")
        assert state(receipt)["posts"] == {"ack": 1, "rejected": 0, "unknown": 0}
        assert state(receipt)["finalized"]
        # Recreate the process-local cache so the next sender must read disk.
        monkeypatch.setattr(daily_delivery, "_GUARDS", {})
        assert send_daily(FeishuSender(config()), "regenerated AI wording")
    post.assert_called_once()
    assert post.call_args.kwargs["timeout"] == 4
    assert state(receipt)["calls"] == 1


def test_two_chunks_with_one_ack_block_automatic_whole_report_resend(prepare, monkeypatch):
    cfg = config(feishu_max_bytes=1000)
    receipt = prepare(cfg)
    with mock.patch(f"{SENDER_MODULE}.requests.post", side_effect=[
        response(), response(400), response(400),
    ]) as post, mock.patch(f"{SENDER_MODULE}.time.sleep"):
        assert not send_daily(FeishuSender(cfg), "A" * 1500)
    assert post.call_count == 3  # First chunk ACK, second card and text rejected.
    assert [call.kwargs["json"]["msg_type"] for call in post.call_args_list] == [
        "interactive", "interactive", "text",
    ]
    assert state(receipt)["status"] == "partial"
    assert state(receipt)["posts"] == {"ack": 1, "rejected": 2, "unknown": 0}
    monkeypatch.setattr(daily_delivery, "_GUARDS", {})
    with mock.patch(f"{SENDER_MODULE}.requests.post") as retry:
        assert not send_daily(FeishuSender(cfg), "different regenerated report")
    retry.assert_not_called()


def non_json_response():
    result = response()
    result.json.side_effect = ValueError("invalid JSON")
    return result


@pytest.mark.parametrize("outcome", [
    pytest.param(requests.Timeout("timeout"), id="timeout"),
    pytest.param(requests.ConnectionError("disconnected"), id="disconnect"),
    pytest.param(response(503), id="server-error"),
    pytest.param(non_json_response(), id="non-json"),
    pytest.param(response(body=[]), id="wrong-json-type"),
    pytest.param(response(body={"msg": "missing code"}), id="missing-code"),
    pytest.param(response(body={"code": None}), id="null-code"),
    pytest.param(response(body={"code": False}), id="boolean-code-is-not-ack"),
])
def test_uncertain_webhook_post_has_no_text_fallback_or_automatic_resend(prepare, outcome):
    receipt = prepare()
    with mock.patch(f"{SENDER_MODULE}.requests.post", side_effect=[outcome, response()]) as post:
        assert not send_daily(FeishuSender(config()))
    post.assert_called_once()
    assert state(receipt)["posts"] == {"ack": 0, "rejected": 0, "unknown": 1}
    assert state(receipt)["status"] == "unknown"
    with mock.patch(f"{SENDER_MODULE}.requests.post") as retry:
        assert not send_daily(FeishuSender(config()))
    retry.assert_not_called()


def test_unknown_second_chunk_stops_text_fallback_and_remaining_chunks(prepare):
    cfg = config(feishu_max_bytes=1000)
    receipt = prepare(cfg)
    with mock.patch(f"{SENDER_MODULE}.requests.post", side_effect=[response(), response(502)]) as post, \
         mock.patch(f"{SENDER_MODULE}.time.sleep") as sleep:
        assert not send_daily(FeishuSender(cfg), "A" * 2500)
    assert post.call_count == 2
    sleep.assert_called_once()
    assert state(receipt)["posts"] == {"ack": 1, "rejected": 0, "unknown": 1}


@pytest.mark.parametrize("rejection", [
    pytest.param(response(body={"code": 19024, "msg": "card rejected"}), id="business-rejection"),
    pytest.param(response(400), id="http-client-error"),
])
def test_explicit_rejection_can_fall_back_to_acknowledged_text(prepare, rejection):
    receipt = prepare()
    with mock.patch(f"{SENDER_MODULE}.requests.post", side_effect=[rejection, response()]) as post:
        assert send_daily(FeishuSender(config()))
    assert post.call_count == 2
    assert post.call_args.kwargs["json"]["msg_type"] == "text"
    assert state(receipt)["posts"] == {"ack": 1, "rejected": 1, "unknown": 0}
    assert state(receipt)["status"] == "acknowledged"


def test_prepared_target_mismatch_refuses_post_and_does_not_create_ack(prepare):
    receipt = prepare()
    with mock.patch(f"{SENDER_MODULE}.requests.post") as post:
        assert not send_daily(FeishuSender(config(feishu_webhook_url="https://feishu.example/other")))
    post.assert_not_called()
    assert state(receipt)["calls"] == 0
    assert state(receipt)["posts"]["ack"] == 0


def test_separate_market_dates_and_destinations_have_separate_receipts(prepare):
    first = prepare(as_of="2026-10-20")
    with mock.patch(f"{SENDER_MODULE}.requests.post", return_value=response()) as post:
        assert send_daily(FeishuSender(config()))
        daily_delivery.finish_daily_components("stocks")
        second = prepare(as_of="2026-10-21")
        assert send_daily(FeishuSender(config()))
        daily_delivery.finish_daily_components("stocks")
        other_cfg = config(feishu_webhook_url="https://feishu.example/other")
        third = prepare(other_cfg, as_of="2026-10-21")
        assert send_daily(FeishuSender(other_cfg))
    assert post.call_count == 3
    assert all(state(path)["posts"]["ack"] == 1 for path in (first, second, third))


@pytest.mark.parametrize("outcome", [
    pytest.param(requests.Timeout("timeout"), id="sdk-network-error"),
    pytest.param(sdk_response(503), id="sdk-server-error-even-with-zero-code"),
    pytest.param(sdk_response(200, None), id="sdk-missing-code"),
    pytest.param(sdk_response(None), id="sdk-no-http-evidence"),
])
def test_scoped_app_unknown_stops_sdk_retries_text_fallback_and_chunks(prepare, outcome):
    sender = app_sender(feishu_max_bytes=1000)
    receipt = prepare(config(
        feishu_webhook_url=None, feishu_app_id="app", feishu_receive_id_type="chat_id",
        feishu_chat_id="chat",
    ))
    client, create = app_client(outcome, sdk_response())
    with mock.patch.object(sender, "_ensure_app_client", return_value=client), \
         mock.patch(f"{SENDER_MODULE}.time.sleep") as sleep:
        assert not send_daily(sender, "A" * 1500)
    create.assert_called_once()
    sleep.assert_not_called()
    assert state(receipt)["posts"] == {"ack": 0, "rejected": 0, "unknown": 1}


def test_scoped_app_ack_and_normalized_domain_survive_new_sender(prepare, monkeypatch):
    cfg = config(
        feishu_webhook_url=None, feishu_app_id="app", feishu_app_secret="secret",
        feishu_receive_id_type="open_id", feishu_chat_id="user", feishu_domain=" LARK ",
    )
    receipt = prepare(cfg)
    client, create = app_client(sdk_response())
    with mock.patch.object(FeishuSender, "_ensure_app_client", return_value=client):
        assert send_daily(FeishuSender(cfg))
        daily_delivery.finish_daily_components("stocks")
        monkeypatch.setattr(daily_delivery, "_GUARDS", {})
        assert send_daily(FeishuSender(cfg))
    create.assert_called_once()
    assert create.call_args.args[0].receive_id_type == "open_id"
    assert state(receipt)["posts"] == {"ack": 1, "rejected": 0, "unknown": 0}


def test_scoped_app_explicit_rejections_keep_retry_and_text_fallback(prepare):
    sender = app_sender()
    receipt = prepare(config(feishu_webhook_url=None, feishu_app_id="app", feishu_chat_id="chat"))
    client, create = app_client(
        sdk_response(code=19024), sdk_response(code=19024), sdk_response(code=19024), sdk_response(),
    )
    with mock.patch.object(sender, "_ensure_app_client", return_value=client), \
         mock.patch(f"{SENDER_MODULE}.time.sleep"):
        assert send_daily(sender)
    assert create.call_count == 4
    assert create.call_args.args[0].request_body.msg_type == "text"
    assert state(receipt)["posts"] == {"ack": 1, "rejected": 3, "unknown": 0}


def test_missing_app_client_does_not_claim_delivery(prepare):
    sender = app_sender()
    receipt = prepare(config(feishu_webhook_url=None, feishu_app_id="app", feishu_chat_id="chat"))
    with mock.patch.object(sender, "_ensure_app_client", return_value=None):
        assert not send_daily(sender)
    assert state(receipt)["posts"] == {"ack": 0, "rejected": 0, "unknown": 0}
    assert state(receipt)["status"] == "not_attempted"


@pytest.mark.parametrize("contents", [(None, "valid report"), ("valid report", None)], ids=[
    "none-before-ack", "none-after-ack",
])
def test_none_callback_and_ack_cannot_finalize_a_partially_failed_phase(prepare, contents):
    receipt = prepare()
    sender = FeishuSender(config())
    with mock.patch(f"{SENDER_MODULE}.requests.post", return_value=response()) as post:
        results = [send_daily(sender, content) for content in contents]
    post.assert_called_once()
    assert results[contents.index(None)] is False
    assert state(receipt)["calls"] == 2
    assert state(receipt)["posts"] == {"ack": 1, "rejected": 0, "unknown": 0}
    assert state(receipt)["status"] == "partial"
    daily_delivery.finish_daily_components("stocks")
    assert not daily_delivery.finalize_daily_delivery()
    assert state(receipt)["status"] != "acknowledged"
    with mock.patch(f"{SENDER_MODULE}.requests.post") as retry:
        assert not send_daily(FeishuSender(config()), "regenerated report")
    retry.assert_not_called()


def test_unscoped_none_retains_false_without_opening_receipt(tmp_path, monkeypatch):
    receipt = tmp_path / "must-not-open.json"
    monkeypatch.setenv("DAILY_DELIVERY_FILE", str(receipt))
    with mock.patch(f"{SENDER_MODULE}.requests.post") as post:
        assert not FeishuSender(config()).send_to_feishu(None)
    post.assert_not_called()
    assert not receipt.exists()
