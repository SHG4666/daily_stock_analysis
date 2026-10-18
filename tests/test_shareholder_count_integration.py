"""披露信息独立于筛选与基本面失败路径的集成检查。"""
from copy import deepcopy
from datetime import date, datetime, timezone
from threading import Event, Timer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.pipeline import StockAnalysisPipeline
from src.enums import ReportType
from src.services.pattern_screening_service import enrich_shareholder_counts, render_html, screen_market
from tests.test_pattern_screening_service import AS_OF, FakeProvider
from tests.test_pipeline_market_phase_context import _make_pipeline, _phase_payload


HOLDERS = {
    "code": "000001", "count": 39518, "end_date": "2026-06-30",
    "announcement_date": "2026-08-27", "source": "东方财富",
    "source_url": "https://data.eastmoney.com/gdhs/detail/000001.html",
    "fetched_at": "2026-10-12T10:10:00+00:00",
}


@pytest.mark.parametrize("response", [HOLDERS, None, TimeoutError("unavailable")])
def test_optional_disclosure_does_not_change_screening_decision_or_order(response):
    result = screen_market(FakeProvider(), AS_OF)
    original = deepcopy(result)
    provider = MagicMock()
    if isinstance(response, Exception):
        provider.get_latest.side_effect = response
    else:
        provider.get_latest.return_value = response
    enrich_shareholder_counts(result, provider)
    provider.get_latest.assert_called_once_with("000001", AS_OF)
    if response is HOLDERS:
        assert result["selected"][0]["shareholder_count"] == HOLDERS
        assert "39,518" in render_html(result)
        assert "截至 2026-06-30" in render_html(result)
        for candidate in result["candidates"]:
            candidate.pop("shareholder_count", None)
    assert result == original


def test_incomplete_screen_does_not_spend_time_on_optional_disclosures():
    provider = MagicMock()
    result = screen_market(FakeProvider("stale"), AS_OF)
    enrich_shareholder_counts(result, provider)
    provider.get_latest.assert_not_called()


def test_optional_budget_cancels_queued_lookups_without_losing_candidates():
    released = Event()
    calls = []

    def lookup(code, as_of):
        calls.append(code)
        assert released.wait(1)
        return None

    result = {"status": "complete", "as_of": AS_OF.isoformat(),
              "candidates": [{"code": "000001"}, {"code": "600519"}], "selected": []}
    original = deepcopy(result)
    timer = Timer(.08, released.set)
    timer.start()
    try:
        enrich_shareholder_counts(result, SimpleNamespace(get_latest=lookup), workers=1, deadline_seconds=.01)
    finally:
        released.set()
        timer.join()
    assert calls == ["000001"]
    assert result == original


def test_html_ignores_disclosure_for_a_different_stock():
    result = screen_market(FakeProvider(), AS_OF)
    result["candidates"][0]["shareholder_count"] = {**HOLDERS, "code": "002727"}
    assert "39,518" not in render_html(result)
    assert "股东户数" not in render_html(result)


@pytest.mark.parametrize("failed", [False, True])
def test_pipeline_disclosure_is_independent_of_disabled_or_failed_fundamental_bundle(failed):
    pipeline = _make_pipeline()
    context = {"status": "failed" if failed else "not_supported", "coverage": {}}
    payload = {**HOLDERS, "code": "600519"}
    pipeline.shareholder_count_provider = MagicMock()
    pipeline.shareholder_count_provider.get_latest.return_value = payload
    enriched = pipeline._attach_shareholder_count_to_fundamental_context("600519", context, AS_OF)
    assert enriched["shareholder_count"] == payload
    assert enriched["status"] == context["status"]
    assert "shareholder_count" not in context  # Cached fundamental contexts are not mutated.


def test_real_legacy_analysis_retains_disclosure_for_notification_and_snapshot():
    pipeline = _make_pipeline()
    payload = {**HOLDERS, "code": "600519"}
    pipeline.shareholder_count_provider = MagicMock()
    pipeline.shareholder_count_provider.get_latest.return_value = payload
    phase = SimpleNamespace(to_dict=lambda: _phase_payload())
    # UTC reference must be converted to the China calendar date before lookup.
    frozen = datetime(2026, 3, 26, 18, 0, tzinfo=timezone.utc)
    with patch("src.core.pipeline.build_market_phase_context", return_value=phase):
        result = pipeline.analyze_stock("600519", ReportType.SIMPLE, "holders-integration", current_time=frozen)
    assert result is not None
    pipeline.shareholder_count_provider.get_latest.assert_called_once_with("600519", date(2026, 3, 27))
    assert result.fundamental_context["shareholder_count"] == payload
    snapshot = pipeline.db.save_fundamental_snapshot.call_args.kwargs["payload"]
    assert snapshot["shareholder_count"] == payload


def test_holder_failure_does_not_turn_successful_analysis_into_failure():
    pipeline = _make_pipeline()
    pipeline.shareholder_count_provider = MagicMock()
    pipeline.shareholder_count_provider.get_latest.side_effect = RuntimeError("optional data unavailable")
    phase = SimpleNamespace(to_dict=lambda: _phase_payload())
    with patch("src.core.pipeline.build_market_phase_context", return_value=phase):
        result = pipeline.analyze_stock(
            "600519", ReportType.SIMPLE, "holders-fail-open", current_time=datetime(2026, 3, 27, 10),
        )
    assert result is not None and result.success
    assert "shareholder_count" not in result.fundamental_context


def test_real_agent_analysis_retains_provider_disclosure_instead_of_model_numbers():
    from src.agent.executor import AgentResult

    pipeline = _make_pipeline(agent_mode=True)
    pipeline._ensure_agent_history = MagicMock()
    payload = {**HOLDERS, "code": "600519"}
    pipeline.shareholder_count_provider = MagicMock()
    pipeline.shareholder_count_provider.get_latest.return_value = payload
    executor = MagicMock()
    executor.run.return_value = AgentResult(
        success=True, content="{}", provider="test",
        dashboard={"stock_name": "贵州茅台", "sentiment_score": 66,
                   "trend_prediction": "震荡", "operation_advice": "持有", "decision_type": "hold"},
    )
    phase = SimpleNamespace(to_dict=lambda: _phase_payload())
    with (patch("src.core.pipeline.build_market_phase_context", return_value=phase),
          patch("src.agent.factory.build_agent_executor", return_value=executor)):
        result = pipeline.analyze_stock(
            "600519", ReportType.SIMPLE, "holders-agent", current_time=datetime(2026, 3, 27, 10),
        )
    assert result is not None and result.success
    assert result.fundamental_context["shareholder_count"] == payload
