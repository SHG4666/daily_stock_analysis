"""模型看空的一票否决必须先于单股、聚合和合并报告。"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.recommendation_filter import rejection_reason
from tests.test_pipeline_single_stock_notify import TestPipelineSingleStockNotify, _make_result


@pytest.mark.parametrize("trend,advice,decision", [
    ("看空", "买入", "buy"), ("Strong Bearish", "Hold", "hold"),
    ("空头排列，短期反弹", "观望", "hold"), ("Downtrend", "Buy", "buy"),
    ("看多", "减仓", "hold"), ("Bullish", "Reduce", "hold"),
    ("震荡", "持有", "sell"),
])
def test_negative_signal_vetoes_conflicting_positive_fields(monkeypatch, trend, advice, decision):
    monkeypatch.setenv("PATTERN_SCREEN_ACTIVE", "true")
    result = SimpleNamespace(trend_prediction=trend, operation_advice=advice, decision_type=decision)
    assert rejection_reason(result)


@pytest.mark.parametrize("trend,advice", [("看多", "买入"), ("震荡", "观望"),
                                        ("不看空", "不建议卖出"), ("not bearish", "Hold")])
def test_non_bearish_or_negated_conclusions_are_not_misclassified(monkeypatch, trend, advice):
    monkeypatch.setenv("PATTERN_SCREEN_ACTIVE", "true")
    assert rejection_reason(SimpleNamespace(trend_prediction=trend, operation_advice=advice,
                                           decision_type="hold")) is None


def test_manual_analysis_keeps_negative_conclusions(monkeypatch):
    monkeypatch.delenv("PATTERN_SCREEN_ACTIVE", raising=False)
    assert rejection_reason(SimpleNamespace(trend_prediction="看空", operation_advice="卖出",
                                           decision_type="sell")) is None


@pytest.mark.parametrize("single,merge", [(True, False), (False, False), (False, True)])
def test_batch_filters_before_any_notification_local_report_and_return(monkeypatch, single, merge):
    monkeypatch.setenv("PATTERN_SCREEN_ACTIVE", "true")
    pipeline = TestPipelineSingleStockNotify._build_batch_pipeline()
    pipeline.config.single_stock_notify = single
    positive, negative = _make_result("000001"), _make_result("600519")
    negative.trend_prediction = "看空"
    pipeline.process_single_stock = MagicMock(side_effect=lambda code, **kw: {
        "000001": positive, "600519": negative}[code])
    pipeline._send_single_stock_notification = MagicMock()
    results = pipeline.run(["000001", "600519"], merge_notification=merge)
    assert results == [positive]  # main's combined report and cloud doc consume this list.
    assert pipeline.analysis_success_count == 2
    assert pipeline.recommendation_exclusions == [{"code": "600519", "reason": "模型趋势看空"}]
    assert pipeline._save_local_report.call_args.args[0] == [positive]
    assert pipeline._send_notifications.call_args.args[0] == [positive]
    if single:
        assert pipeline._send_single_stock_notification.call_args.args[0] is positive
        assert pipeline._send_single_stock_notification.call_count == 1
    else:
        pipeline._send_single_stock_notification.assert_not_called()


def test_all_bearish_is_successful_analysis_without_recommendations_or_notifications(monkeypatch):
    monkeypatch.setenv("PATTERN_SCREEN_ACTIVE", "true")
    pipeline = TestPipelineSingleStockNotify._build_batch_pipeline()
    negative = _make_result("000001")
    negative.trend_prediction = "强烈看空"
    pipeline.process_single_stock = MagicMock(return_value=negative)
    pipeline._send_single_stock_notification = MagicMock()
    assert pipeline.run(["000001"]) == []
    assert pipeline.analysis_success_count == 1 and len(pipeline.recommendation_exclusions) == 1
    pipeline._save_local_report.assert_not_called()
    pipeline._send_notifications.assert_not_called()
    pipeline._send_single_stock_notification.assert_not_called()


@pytest.mark.parametrize("success_count,raises", [(1, False), (0, True)])
def test_delayed_main_distinguishes_model_veto_from_analysis_failure(monkeypatch, success_count, raises):
    from datetime import date
    from unittest.mock import patch
    import main

    pipeline = MagicMock()
    pipeline.analysis_success_count = success_count
    pipeline.run.return_value = []
    config = SimpleNamespace(stock_list=["000001"], market_review_enabled=False,
                             single_stock_notify=False, analysis_delay=0, backtest_enabled=False)
    args = SimpleNamespace(workers=1, no_market_review=True, dry_run=False, no_notify=True)
    monkeypatch.setattr(main, "_resolve_completed_session", lambda args: date(2026, 10, 22))
    monkeypatch.setattr(main, "_refresh_stock_index_cache_for_analysis", lambda *args: None)
    monkeypatch.setattr(main, "_compute_trading_day_filter", lambda *args: (["000001"], "cn", False))
    with patch("src.core.pipeline.StockAnalysisPipeline", return_value=pipeline), \
            patch("src.feishu_doc.FeishuDocManager") as document:
        document.return_value.is_configured.return_value = False
        if raises:
            with pytest.raises(RuntimeError, match="分析均失败"):
                main.run_full_analysis(config, args, ["000001"])
        else:
            main.run_full_analysis(config, args, ["000001"])
            document.return_value.create_daily_doc.assert_not_called()
