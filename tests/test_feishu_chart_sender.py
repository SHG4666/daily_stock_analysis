"""Structured Feishu cards keep exact byte limits and truthful delivery status."""
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

from src.notification_sender.feishu_sender import FeishuSender


def config(**overrides):
    values = {"feishu_webhook_url": "https://feishu.example/hook", "feishu_max_bytes": 20000}
    values.update(overrides)
    return SimpleNamespace(**values)


def card():
    return {
        "schema": "2.0",
        "header": {"title": {"tag": "plain_text", "content": "国货航 001391 · 走势图"}},
        "body": {"elements": [
            {"tag": "chart", "preview": True, "chart_spec": {
                "type": "line", "xField": "date", "yField": "price",
                "data": {"values": [{"date": "2026-10-12", "price": 7.98}]},
            }},
        ]},
    }


def response(body=None, status=200):
    result = mock.Mock(status_code=status)
    result.json.return_value = {"code": 0} if body is None else body
    return result


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def test_webhook_sends_exact_compact_utf8_with_timeout_and_tls_settings():
    payload = card()
    sender = FeishuSender(config(webhook_verify_ssl=False))
    with mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response()) as post:
        assert sender.send_card_to_feishu(payload, timeout_seconds=12)
    post.assert_called_once()
    assert post.call_args.args == ("https://feishu.example/hook",)
    kwargs = post.call_args.kwargs
    assert kwargs["data"] == compact({"msg_type": "interactive", "card": payload})
    assert "json" not in kwargs
    assert "国货航".encode("utf-8") in kwargs["data"]
    assert kwargs["headers"] == {"Content-Type": "application/json; charset=utf-8"}
    assert kwargs["timeout"] == 12
    assert kwargs["verify"] is False


def test_keyword_and_signature_do_not_mutate_input_and_count_toward_limit():
    original = card()
    snapshot = deepcopy(original)
    cfg = config(feishu_webhook_keyword="股票日报", feishu_webhook_secret="test-secret")
    sender = FeishuSender(cfg)
    with mock.patch("src.notification_sender.feishu_sender.time.time", return_value=1700000000), \
         mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response()) as post:
        assert sender.send_card_to_feishu(original)
        sent_bytes = post.call_args.kwargs["data"]
        sent = json.loads(sent_bytes)
        assert sent["timestamp"] == "1700000000"
        assert sent["sign"] == sender._build_security_fields()["sign"]
    assert original == snapshot
    assert sent["card"]["body"]["elements"][0] == {"tag": "markdown", "content": "股票日报"}
    assert sent["card"]["body"]["elements"][1] == original["body"]["elements"][0]
    assert len(sent_bytes) > len(compact({"msg_type": "interactive", "card": original}))

    # A body that fits before safety additions still must not be sent over budget.
    cfg.feishu_max_bytes = len(sent_bytes) - 1
    with mock.patch("src.notification_sender.feishu_sender.time.time", return_value=1700000000), \
         mock.patch("src.notification_sender.feishu_sender.requests.post") as post:
        assert not FeishuSender(cfg).send_card_to_feishu(original)
    post.assert_not_called()

    cfg.feishu_max_bytes = len(sent_bytes)
    with mock.patch("src.notification_sender.feishu_sender.time.time", return_value=1700000000), \
         mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response()) as post:
        assert FeishuSender(cfg).send_card_to_feishu(original)
    assert len(post.call_args.kwargs["data"]) == cfg.feishu_max_bytes


def test_webhook_never_exceeds_platform_limit_even_if_config_is_larger():
    payload = card()
    payload["body"]["elements"].append({"tag": "markdown", "content": "x" * 20000})
    with mock.patch("src.notification_sender.feishu_sender.requests.post") as post:
        assert not FeishuSender(config(feishu_max_bytes=50000)).send_card_to_feishu(payload)
    post.assert_not_called()


def test_legacy_card_keyword_uses_legacy_element_without_changing_original():
    payload = {"elements": [{"tag": "div", "text": {"tag": "lark_md", "content": "原报告"}}]}
    snapshot = deepcopy(payload)
    with mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response()) as post:
        assert FeishuSender(config(feishu_webhook_keyword="日报")).send_card_to_feishu(payload)
    sent = json.loads(post.call_args.kwargs["data"])
    assert sent["card"]["elements"][0]["text"]["content"] == "日报"
    assert payload == snapshot


@pytest.mark.parametrize("invalid", [None, [], {}, {"bad": float("nan")}, {"bad": object()}])
def test_invalid_or_non_json_card_is_rejected_before_send(invalid):
    with mock.patch("src.notification_sender.feishu_sender.requests.post") as post:
        assert not FeishuSender(config()).send_card_to_feishu(invalid)
    post.assert_not_called()


def test_keyword_needs_valid_card_elements():
    with mock.patch("src.notification_sender.feishu_sender.requests.post") as post:
        assert not FeishuSender(config(feishu_webhook_keyword="日报")).send_card_to_feishu({"schema": "2.0"})
    post.assert_not_called()


@pytest.mark.parametrize("body", [{"code": 19024}, {"StatusCode": 9499}, {}, [], "unexpected"])
def test_business_error_or_bad_response_has_no_text_fallback(body):
    with mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response(body)) as post:
        assert not FeishuSender(config()).send_card_to_feishu(card())
    post.assert_called_once()
    assert json.loads(post.call_args.kwargs["data"])["msg_type"] == "interactive"


def test_legacy_success_code_is_accepted():
    with mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response({"StatusCode": 0})):
        assert FeishuSender(config()).send_card_to_feishu(card())


def test_http_error_is_reported_without_fallback():
    with mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response(status=429)) as post:
        assert not FeishuSender(config()).send_card_to_feishu(card())
    post.assert_called_once()


def test_non_json_response_is_reported_without_logging_body(caplog):
    result = response()
    result.json.side_effect = ValueError("private-response-body")
    with mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=result) as post:
        assert not FeishuSender(config()).send_card_to_feishu(card())
    post.assert_called_once()
    assert "private-response-body" not in caplog.text


def test_network_exception_does_not_reveal_webhook_url(caplog):
    url = "https://feishu.example/hook/private-token"
    error = requests.exceptions.ConnectionError("Connection to " + url + " failed")
    with mock.patch("src.notification_sender.feishu_sender.requests.post", side_effect=error) as post:
        assert not FeishuSender(config(feishu_webhook_url=url)).send_card_to_feishu(card())
    post.assert_called_once()
    assert "ConnectionError" in caplog.text
    assert url not in caplog.text
    assert "private-token" not in caplog.text


def test_webhook_has_priority_over_app_bot_and_does_not_require_sdk():
    sender = FeishuSender(config(feishu_app_id="app", feishu_app_secret="secret", feishu_chat_id="chat"))
    with mock.patch("src.notification_sender.feishu_sender.FEISHU_SDK_AVAILABLE", False), \
         mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response()) as post, \
         mock.patch.object(sender, "_ensure_app_client", side_effect=AssertionError("must not use app")):
        assert sender.send_card_to_feishu(card())
    post.assert_called_once()


def test_app_bot_sends_compact_card_through_existing_client():
    sender = FeishuSender(config(feishu_webhook_url=None, feishu_chat_id="chat", feishu_webhook_keyword="webhook-only"))
    client = object()
    payload = card()
    with mock.patch.object(sender, "_ensure_app_client", return_value=client), \
         mock.patch.object(sender, "_app_send_raw", return_value=True) as send, \
         mock.patch("src.notification_sender.feishu_sender.requests.post") as post:
        assert sender.send_card_to_feishu(payload)
    send.assert_called_once_with(client, "interactive", compact(payload).decode("utf-8"))
    post.assert_not_called()


def test_app_bot_failure_has_no_text_fallback():
    sender = FeishuSender(config(feishu_webhook_url=None, feishu_chat_id="chat"))
    with mock.patch.object(sender, "_ensure_app_client", return_value=object()), \
         mock.patch.object(sender, "_app_send_raw", return_value=False) as send:
        assert not sender.send_card_to_feishu(card())
    send.assert_called_once()
    assert send.call_args.args[1] == "interactive"


def test_app_bot_missing_chat_does_not_initialize_client():
    sender = FeishuSender(config(feishu_webhook_url=None))
    with mock.patch.object(sender, "_ensure_app_client") as ensure:
        assert not sender.send_card_to_feishu(card())
    ensure.assert_not_called()


def test_app_bot_missing_client_cannot_deliver_card():
    sender = FeishuSender(config(feishu_webhook_url=None, feishu_chat_id="chat"))
    with mock.patch.object(sender, "_ensure_app_client", return_value=None), \
         mock.patch.object(sender, "_app_send_raw") as send:
        assert not sender.send_card_to_feishu(card())
    send.assert_not_called()


def test_app_bot_oversized_card_is_rejected_before_client_init():
    sender = FeishuSender(config(feishu_webhook_url=None, feishu_chat_id="chat", feishu_max_bytes=50))
    with mock.patch.object(sender, "_ensure_app_client") as ensure:
        assert not sender.send_card_to_feishu(card())
    ensure.assert_not_called()


def test_legacy_text_send_still_uses_old_formatter_and_json_path():
    sender = FeishuSender(config())
    with mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response()) as post:
        assert sender.send_to_feishu("# 原有报告")
    assert "data" not in post.call_args.kwargs
    assert post.call_args.kwargs["json"]["card"]["elements"][0]["text"]["content"] == "**原有报告**"
