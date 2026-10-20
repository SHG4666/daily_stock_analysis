"""Regression coverage for the actual #55 overnight failure and its downstream date boundary."""
import json
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

import main
from scripts import screen_eight_patterns as entry
from src.core import screening_session as sessions
from src.core.pipeline import StockAnalysisPipeline
from src.core.trading_calendar import MarketPhase
from src.enums import ReportType
from src.services import history_loader
from tests.test_pipeline_market_phase_context import _analysis_result, _make_pipeline


SESSION = date(2026, 10, 19)
NOW = datetime(2026, 10, 20, 0, 57, 49, tzinfo=ZoneInfo("Asia/Shanghai"))


@pytest.mark.parametrize("now,target,event,expected", [
    (NOW, SESSION, "schedule", True),
    (NOW.astimezone(ZoneInfo("UTC")), SESSION, "schedule", True),
    (NOW.replace(hour=8, minute=59, second=59), SESSION, "schedule", True),
    (NOW.replace(hour=9, minute=0, second=0), SESSION, "schedule", False),
    (NOW.replace(hour=14), SESSION, "schedule", False),
    (NOW, SESSION, "workflow_dispatch", False),
    (NOW, SESSION, "", False),
    (NOW, date(2026, 10, 16), "schedule", False),
    (NOW, NOW.date(), "schedule", False),
    (datetime(2026, 10, 17, 1, tzinfo=NOW.tzinfo), date(2026, 10, 16), "schedule", True),
    (datetime(2026, 10, 18, 1, tzinfo=NOW.tzinfo), date(2026, 10, 17), "schedule", False),
])
def test_real_calendar_overnight_window(now, target, event, expected):
    assert sessions.is_delayed_schedule(now, target, event) is expected


def test_calendar_unknown_fails_closed(monkeypatch):
    monkeypatch.setattr(sessions, "infer_market_phase", lambda *a, **kw: MarketPhase.UNKNOWN)
    assert not sessions.is_delayed_schedule(NOW, SESSION, "schedule")
    with pytest.raises(ValueError):
        sessions.validate_completed_session(SESSION.isoformat(), NOW, "schedule")


@pytest.mark.parametrize("now,target", [
    (NOW, SESSION),
    (datetime(2026, 10, 17, 1, tzinfo=NOW.tzinfo), date(2026, 10, 16)),
])
def test_entrypoint_reproduces_actual_failure_time_and_hands_off_session(tmp_path, monkeypatch, now, target):
    monkeypatch.setattr("src.config.setup_env", lambda: None)
    monkeypatch.setattr(entry, "get_market_now", lambda *a: now)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(entry, "PatternMarketData", MagicMock())
    captured = []

    def screen(provider, as_of, **kwargs):
        captured.append(as_of)
        return {"status": "complete", "as_of": as_of.isoformat(), "selected": [], "candidates": [],
                "excluded": {}, "data_issues": {}, "coverage": 1, "universe_count": 1, "eligible_count": 1}

    monkeypatch.setattr(entry, "screen_market", screen)
    assert entry.main(["--output-dir", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "screen.json").read_text())
    assert captured == [target]
    assert result["delayed_schedule"] is True
    assert result["generated_at"] == now.isoformat()
    assert "as_of=" + target.isoformat() in output.read_text()
    assert "delayed_schedule=true" in output.read_text()
    rendered = (tmp_path / "report.html").read_text()
    assert target.isoformat() in rendered and "定时延迟运行" in rendered


def test_pipeline_fetch_clips_even_a_provider_ignoring_end_date():
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.completed_session = SESSION
    pipeline.db = MagicMock()
    pipeline.db.has_today_data.return_value = False
    pipeline.fetcher_manager = MagicMock()
    pipeline.fetcher_manager.get_daily_data.return_value = (
        pd.DataFrame({"date": ["2026-10-16", "2026-10-19", "2026-10-20"], "close": [10, 11, 999]}), "test",
    )
    assert pipeline.fetch_and_save_stock_data("002727", current_time=NOW) == (True, None)
    pipeline.fetcher_manager.get_daily_data.assert_called_once_with("002727", days=30, end_date="2026-10-19")
    saved = pipeline.db.save_daily_data.call_args.args[0]
    assert saved["close"].tolist() == [10, 11]
    pipeline.db.has_today_data.assert_called_once_with("002727", SESSION)


def test_missing_completed_bar_does_not_continue_with_news_only():
    pipeline = _make_pipeline()
    pipeline.completed_session = SESSION
    pipeline.fetch_and_save_stock_data = MagicMock(return_value=(False, "missing session"))
    pipeline.analyze_stock = MagicMock()
    assert pipeline.process_single_stock("002727", current_time=NOW) is None
    pipeline.analyze_stock.assert_not_called()
    assert history_loader.get_completed_session() is None


def test_real_analysis_path_disables_quotes_and_bounds_context():
    pipeline = _make_pipeline()
    pipeline.completed_session = SESSION
    pipeline.config.enable_realtime_quote = True
    pipeline.db.get_analysis_context.return_value["date"] = SESSION.isoformat()
    result = pipeline.analyze_stock("600519", report_type=ReportType.SIMPLE,
                                    query_id="delayed-test", current_time=NOW)
    assert result is not None
    pipeline.fetcher_manager.get_realtime_quote.assert_not_called()
    pipeline.fetcher_manager.get_chip_distribution.assert_not_called()
    pipeline.db.get_analysis_context.assert_called_with("600519", target_date=SESSION)
    pipeline.analyzer.analyze.assert_called_once()


def test_delayed_analysis_rejects_stale_database_context():
    pipeline = _make_pipeline()
    pipeline.completed_session = SESSION
    assert pipeline.analyze_stock("600519", ReportType.SIMPLE, "stale", current_time=NOW) is None
    pipeline.analyzer.analyze.assert_not_called()


@pytest.mark.parametrize("value,event", [
    ("2026-10-19", "workflow_dispatch"), ("2026-10-16", "schedule"),
    ("20261019", "schedule"), ("not-a-date", "schedule"),
])
def test_main_revalidates_handoff_even_with_force_run(monkeypatch, value, event):
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setattr("src.core.trading_calendar.get_market_now", lambda *a, **kw: kw.get("current_time") or NOW)
    with pytest.raises(ValueError):
        main._resolve_completed_session(SimpleNamespace(completed_session=value, force_run=True))


def test_history_network_fallback_and_agent_quote_obey_context(monkeypatch):
    from src.agent.tools import data_tools
    manager = MagicMock()
    manager.get_daily_data.return_value = (
        pd.DataFrame({"date": ["2026-10-19", "2026-10-20"], "close": [11, 999]}), "test",
    )
    db = MagicMock()
    db.get_data_range.return_value = []
    monkeypatch.setattr("src.storage.get_db", lambda: db)
    monkeypatch.setattr(history_loader, "_get_fetcher_manager", lambda: manager)
    monkeypatch.setattr(data_tools, "_get_fetcher_manager", lambda: pytest.fail("must not request live quote"))
    token = history_loader.set_completed_session(SESSION)
    frozen = history_loader.set_frozen_target_date(SESSION)
    try:
        df, source = history_loader.load_history_df("002727")
        assert df["close"].tolist() == [11]
        assert source == "test"
        assert manager.get_daily_data.call_args.kwargs["end_date"] == "2026-10-19"
        assert data_tools._handle_get_realtime_quote("002727")["as_of"] == "2026-10-19"
        assert data_tools._handle_get_chip_distribution("002727")["as_of"] == "2026-10-19"
    finally:
        history_loader.reset_frozen_target_date(frozen)
        history_loader.reset_completed_session(token)


def test_friday_delayed_to_saturday_runs_full_flow_without_force_run(monkeypatch):
    saturday = datetime(2026, 10, 17, 1, tzinfo=NOW.tzinfo)
    friday = date(2026, 10, 16)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setattr("src.core.trading_calendar.get_market_now", lambda *a, **kw: kw.get("current_time") or saturday)
    monkeypatch.setattr(main, "_refresh_stock_index_cache_for_analysis", lambda *a: None)
    config = SimpleNamespace(stock_list=["002727"], market_review_enabled=True, single_stock_notify=False,
                             merge_email_notification=False, trading_day_check_enabled=True, analysis_delay=0,
                             backtest_enabled=False)
    args = SimpleNamespace(completed_session=friday.isoformat(), force_run=False, no_market_review=False,
                           no_context_snapshot=False, workers=1, dry_run=False, no_notify=True)
    events = []
    pipeline = MagicMock()
    result = _analysis_result()
    result.code = "002727"
    pipeline.run.side_effect = lambda **kw: events.append("stocks") or [result]
    review = MagicMock(side_effect=lambda **kw: events.append("market") or "market report")
    with patch("src.core.pipeline.StockAnalysisPipeline", return_value=pipeline) as factory, \
            patch("src.core.market_review.run_market_review", review), \
            patch.object(main, "_run_market_review_with_shared_lock", side_effect=lambda config, fn, **kw: fn(**kw)), \
            patch("src.feishu_doc.FeishuDocManager") as docs:
        docs.return_value.is_configured.return_value = False
        main.run_full_analysis(config, args, ["002727"])
    assert events == ["market", "stocks"]
    assert factory.call_args.kwargs["completed_session"] == friday
    assert pipeline.run.call_args.kwargs["stock_codes"] == ["002727"]
    assert review.call_args.kwargs["completed_session"] == friday
    assert review.call_args.kwargs["send_notification"] is False


def test_market_overview_uses_completed_date_and_rejects_slow_crossing(monkeypatch):
    from src.market_analyzer import MarketAnalyzer
    analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
    analyzer.region = "cn"
    analyzer.completed_session = SESSION
    analyzer.profile = SimpleNamespace(has_market_stats=False, has_sector_rankings=False)
    analyzer._get_main_indices = MagicMock(return_value=[])
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    with patch("src.core.trading_calendar.get_market_now", side_effect=lambda *a, **kw: kw.get("current_time") or NOW):
        assert analyzer.get_market_overview().date == "2026-10-19"
    clocks = iter([NOW, NOW.replace(hour=9)])
    with patch("src.core.trading_calendar.get_market_now", side_effect=lambda *a, **kw: kw.get("current_time") or next(clocks)):
        with pytest.raises(ValueError):
            analyzer.get_market_overview()


def test_aggregate_report_distinguishes_market_date_and_generation():
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.completed_session = SESSION
    pipeline.notifier = MagicMock()
    pipeline.notifier.generate_aggregate_report.return_value = "# daily report"
    with patch("src.core.pipeline.get_market_now", return_value=NOW):
        report = pipeline._generate_aggregate_report([], "brief")
    pipeline.notifier.generate_aggregate_report.assert_called_once_with([], "brief", report_date="2026-10-19")
    assert "行情截至 2026-10-19" in report and "生成 2026-10-20" in report
    assert "非历史回测" in report


def test_market_review_save_and_notification_share_completed_date():
    from src.core import market_review
    notifier = MagicMock()
    review = SimpleNamespace(report="## 2026-10-19 A股复盘\n\n收盘行情", market_light_snapshot={},
                             structured_payload={})
    analyzer = MagicMock()
    analyzer.run_daily_review_with_snapshot.return_value = review
    with patch.object(market_review, "MarketAnalyzer", return_value=analyzer) as factory, \
            patch.object(market_review, "_persist_market_review_history"), \
            patch("src.core.trading_calendar.get_market_now", return_value=NOW):
        result = market_review.run_market_review(
            notifier, config=SimpleNamespace(report_language="zh", market_review_region="cn"),
            completed_session=SESSION,
        )
    assert result == review.report
    assert factory.call_args.kwargs["completed_session"] == SESSION
    saved, filename = notifier.save_report_to_file.call_args.args
    assert filename == "market_review_20261019.md"
    assert "行情截至 2026-10-19" in saved and "生成 2026-10-20" in saved
    assert "行情截至 2026-10-19" in notifier.send.call_args.args[0]
