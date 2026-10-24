"""Daily workflow wiring through the real notification dispatch and receipt gate."""
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import requests
import pytest
import yaml

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.config import Config
from src.core.daily_delivery import (
    create_delivery_files,
    feishu_target_fingerprint,
    finalize_daily_delivery,
    finish_daily_components,
)
from src.core.pipeline import StockAnalysisPipeline
from src.enums import ReportType
from src.notification import NotificationChannel, NotificationService
from tests.test_pipeline_market_phase_context import _analysis_result


URL = "https://open.feishu.cn/open-apis/bot/v2/hook/test-daily-destination"
ROOT = Path(__file__).resolve().parents[1]


def _service(tmp_path, monkeypatch, prior=None):
    identity = dict(repository="owner/repo", workflow_id=123,
                    workflow_path=".github/workflows/00-daily-analysis.yml", branch="main",
                    run_id="1", run_attempt=1, event_name="workflow_dispatch", as_of="2026-10-21",
                    target_fingerprint=feishu_target_fingerprint(URL))
    _, receipt = create_delivery_files(tmp_path, identity, ("stocks", "market", "document"), prior=prior)
    monkeypatch.setenv("DAILY_DELIVERY_FILE", str(receipt))
    for key, value in (("GITHUB_REPOSITORY", "owner/repo"), ("GITHUB_RUN_ID", "1"),
                       ("GITHUB_RUN_ATTEMPT", "1")):
        monkeypatch.setenv(key, value)
    config = Config(stock_list=[], feishu_webhook_url=URL)
    monkeypatch.setattr("src.notification.get_config", lambda: config)
    service = NotificationService()
    service._available_channels = [NotificationChannel.FEISHU, NotificationChannel.EMAIL]
    service.send_to_email = MagicMock(return_value=True)
    service.save_report_to_file = MagicMock(return_value="test-report.md")
    service.send = MagicMock(wraps=service.send)
    response = SimpleNamespace(status_code=200, json=lambda: {"code": 0})
    post = MagicMock(return_value=response)
    monkeypatch.setattr("src.notification_sender.feishu_sender.requests.post", post)
    return service, receipt, post


def test_daily_workflow_requires_durable_intent_before_analysis_and_always_saves_receipt():
    workflow = yaml.safe_load((ROOT / ".github/workflows/00-daily-analysis.yml").read_text())
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["concurrency"] == {"group": "stock-analysis", "cancel-in-progress": False}
    steps = workflow["jobs"]["analyze"]["steps"]
    indexes = {step["name"]: i for i, step in enumerate(steps)}
    prepare = next(step for step in steps if step.get("id") == "delivery")
    intent = next(step for step in steps if step.get("id") == "delivery_intent")
    assert indexes[prepare["name"]] < indexes[intent["name"]] < indexes["执行股票分析"]
    assert intent["if"] == "steps.delivery.outputs.enabled == 'true'"
    assert intent["with"]["if-no-files-found"] == "error"
    assert prepare["if"] == "steps.patterns.outputs.status != 'market_closed' && github.event.inputs.mode != 'learning-only'"
    assert "force_run" not in prepare["env"]["DELIVERY_RESEND"]
    assert "resend_daily" in prepare["env"]["DELIVERY_RESEND"]
    assert "steps.patterns.outputs.as_of" in prepare["env"]["DELIVERY_AS_OF"]
    for name in ("核对每日飞书发送回执", "保存每日飞书发送回执"):
        step = steps[indexes[name]]
        assert indexes[name] > indexes["执行股票分析"]
        assert "always()" in step["if"] and "delivery_intent.outcome == 'success'" in step["if"]


def test_workflow_closed_market_skips_analysis_before_any_send(tmp_path):
    workflow = yaml.safe_load((ROOT / ".github/workflows/00-daily-analysis.yml").read_text())
    step = next(step for step in workflow["jobs"]["analyze"]["steps"] if step["name"] == "执行股票分析")
    script = re.sub(r"\$\{\{.*?\}\}", "VALUE", step["run"])
    result = subprocess.run(["/bin/bash"], input=script, text=True, capture_output=True, cwd=tmp_path,
                            env={"PATTERN_SCREEN_ACTIVE": "true", "PATTERN_SCREEN_STATUS": "market_closed"})
    assert result.returncode == 0 and "A股休市" in result.stdout
    assert "🚀" not in result.stdout


def test_delivered_daily_report_skips_only_feishu_not_other_channels(tmp_path, monkeypatch):
    service, receipt, post = _service(tmp_path, monkeypatch, {"stocks": "delivered"})
    assert service.send("new model wording", route_type="report", daily_components=("stocks",))
    post.assert_not_called()
    service.send_to_email.assert_called_once()
    assert json.loads(receipt.read_text())["components"]["stocks"]["posts"]["ack"] == 0


def test_unscoped_alert_remains_deliverable_with_daily_gate_enabled(tmp_path, monkeypatch):
    service, _, post = _service(tmp_path, monkeypatch, {"stocks": "delivered"})
    assert service.send("unrelated alert", route_type="alert")
    post.assert_called_once()


def test_other_channel_success_cannot_acknowledge_a_failed_feishu_send(tmp_path, monkeypatch):
    service, receipt, post = _service(tmp_path, monkeypatch)
    post.side_effect = requests.Timeout("response lost")
    assert service.send("report", route_type="report", daily_components=("stocks",))
    assert post.call_count == 1  # No text fallback after an unknown result.
    finish_daily_components("stocks")
    state = json.loads(receipt.read_text())["components"]["stocks"]
    assert state["posts"]["ack"] == 0 and state["posts"]["unknown"] == 1
    assert not finalize_daily_delivery()


def test_stocks_receipt_does_not_suppress_market_report(tmp_path, monkeypatch):
    service, receipt, post = _service(tmp_path, monkeypatch, {"stocks": "delivered"})
    from src.core import market_review
    review = SimpleNamespace(report="market report", market_light_snapshot={}, market_review_payload={})
    monkeypatch.setattr(market_review, "MarketAnalyzer", MagicMock(**{
        "return_value.run_daily_review_with_snapshot.return_value": review,
    }))
    monkeypatch.setattr(market_review, "_persist_market_review_history", lambda **kw: None)
    assert market_review.run_market_review(service, config=SimpleNamespace(market_review_region="cn"))
    assert service.send.call_args.kwargs["daily_components"] == ("market",)
    post.assert_called_once()
    assert json.loads(receipt.read_text())["components"]["market"]["finalized"]


def _pipeline(service):
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.notifier = service
    pipeline.config = SimpleNamespace(stock_email_groups=[])
    pipeline.save_context_snapshot = False
    pipeline._generate_aggregate_report = MagicMock(return_value="stocks report")
    return pipeline


def test_aggregate_direct_feishu_path_uses_daily_receipt(tmp_path, monkeypatch):
    service, _, post = _service(tmp_path, monkeypatch, {"stocks": "delivered"})
    _pipeline(service)._send_notifications([_analysis_result()], ReportType.SIMPLE)
    post.assert_not_called()
    service.send_to_email.assert_called_once()


def test_single_stock_dispatch_uses_same_daily_component(tmp_path, monkeypatch):
    service, _, post = _service(tmp_path, monkeypatch, {"stocks": "delivered"})
    service.generate_single_stock_report = MagicMock(return_value="one stock report")
    _pipeline(service)._send_single_stock_notification(_analysis_result(), ReportType.SIMPLE)
    assert service.send.call_count == 1
    assert service.send.call_args.kwargs["daily_components"] == ("stocks",)
    post.assert_not_called()


@pytest.mark.parametrize("failure_first", [True, False])
def test_single_stock_render_failure_cannot_be_hidden_by_another_ack(tmp_path, monkeypatch, failure_first):
    service, receipt, post = _service(tmp_path, monkeypatch)
    outcomes = [ValueError("cannot render this stock"), "one stock report"]
    service.generate_single_stock_report = MagicMock(side_effect=outcomes if failure_first else outcomes[::-1])
    pipeline = _pipeline(service)
    for code in ("002727", "600519"):
        result = _analysis_result()
        result.code = code
        pipeline._send_single_stock_notification(result, ReportType.SIMPLE)

    finish_daily_components("stocks")
    assert post.call_count == 1
    assert service.generate_single_stock_report.call_count == 2
    assert not finalize_daily_delivery()
    state = json.loads(receipt.read_text())["components"]["stocks"]
    assert state["status"] == "partial" and state["failed_calls"] == 1


@pytest.mark.parametrize("prior,expected_posts,complete", [
    (None, 1, True),
    ({"stocks": "delivered"}, 0, False),
])
def test_main_combined_report_preserves_component_receipts(tmp_path, monkeypatch, prior, expected_posts, complete):
    import main

    service, receipt, post = _service(tmp_path, monkeypatch, prior)
    service.generate_aggregate_report = MagicMock(return_value="stocks report")
    pipeline = MagicMock(notifier=service)
    pipeline.run.return_value = [_analysis_result()]
    config = SimpleNamespace(stock_list=["002727"], market_review_enabled=True, single_stock_notify=False,
                             merge_email_notification=True, analysis_delay=0, backtest_enabled=False)
    args = SimpleNamespace(completed_session=None, no_market_review=False, no_context_snapshot=False,
                           workers=1, dry_run=False, no_notify=False)
    monkeypatch.setattr(main, "_refresh_stock_index_cache_for_analysis", lambda *a: None)
    monkeypatch.setattr(main, "_compute_trading_day_filter", lambda *a: (["002727"], "cn", False))
    monkeypatch.setattr("src.core.pipeline.StockAnalysisPipeline", lambda **kw: pipeline)
    monkeypatch.setattr(main, "_run_market_review_with_shared_lock", lambda *a, **kw: "market report")
    documents = MagicMock()
    documents.is_configured.return_value = False
    monkeypatch.setattr("src.feishu_doc.FeishuDocManager", lambda: documents)

    main.run_full_analysis(config, args, ["002727"])

    assert service.send.call_count == 1
    assert service.send.call_args.kwargs["daily_components"] == ("stocks", "market")
    assert post.call_count == expected_posts
    assert finalize_daily_delivery() is complete
    states = json.loads(receipt.read_text())["components"]
    assert all(states[key]["posts"]["ack"] == expected_posts for key in ("stocks", "market"))
