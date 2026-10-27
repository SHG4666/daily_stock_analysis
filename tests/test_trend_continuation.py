"""上涨—缩量回踩—再突破的时序、反例、共同质量门槛与每日入口。"""
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src import trend_continuation as trend
from src.bottom_reversal import PROFILE, RULESET_VERSION, inspect_bottom_reversal
from src.pattern_screener import prepare_bars
from src.services.pattern_screening_service import screen_market, write_result
from tests.test_bottom_reversal import AS_OF, Provider


def continuation_bars():
    closes = np.r_[np.linspace(7, 9, 120), np.linspace(9, 10, 20), np.linspace(10, 12, 12),
                   [11.85, 11.6, 11.45, 11.55, 11.65, 11.7, 11.8], 12.6]
    frame = pd.DataFrame({"date": pd.bdate_range(end=AS_OF, periods=160), "close": closes,
                          "open": closes * .995, "high": closes * 1.01,
                          "low": closes * .99, "volume": 1000.})
    frame.loc[140:151, "volume"] = 2000.
    frame.loc[152:158, "volume"] = 900.
    frame.loc[159, ["open", "low", "high", "volume"]] = [11.85, 11.8, 12.7, 3500.]
    return frame


class ContinuationProvider(Provider):
    def __init__(self, turnover_problem=None):
        super().__init__(turnover_problem)
        self.frame = continuation_bars()

    def get_universe(self):
        return [{"code": "000001", "name": "合成中继样本", "open": 11.85, "price": 12.6,
                 "amount": 30_000_000, "volume": 3500., "change_percent": 6.78}]


def test_uptrend_continuation_enters_daily_pool_without_requiring_a_current_downtrend(tmp_path):
    provider = ContinuationProvider()
    metrics, rejection = inspect_bottom_reversal(prepare_bars(provider.frame, AS_OF))
    assert metrics["prior_decline"] > 0 and rejection is None
    assert metrics["continuation"]["matched"] and metrics["continuation"]["base_sessions"] == 7
    assert all(metrics["continuation"]["checks"].values())
    result = screen_market(provider, AS_OF, profile=PROFILE)
    assert result["status"] == "complete" and len(result["selected"]) == 1
    candidate = result["selected"][0]
    assert [s["name"] for s in candidate["signals"]] == [trend.SIGNAL_NAME]
    assert candidate["bottom_reversal"]["turnover"] == 2.2
    assert result["parameters"]["bottom_reversal"]["trend_continuation"] == trend.LIMITS
    write_result(result, tmp_path)
    rendered = (tmp_path / "report.html").read_text()
    assert trend.SIGNAL_NAME in rendered and "整理交易日：7" in rendered
    assert "https://xueqiu.com/S/SZ000001" in rendered and '<svg' not in rendered


@pytest.mark.parametrize("case,check", [
    ("no_base", "整理4至12日"), ("late_peak", "整理4至12日"),
    ("weak_advance", "上涨先于整理"), ("falling_trend", "中期趋势向上"),
    ("deep_pullback", "回调幅度受控"), ("no_shrink", "整理缩量"),
    ("lost_support", "守住中期支撑"), ("no_breakout", "放量突破前高"),
    ("too_far_above_peak", "放量突破前高"), ("large_gap", "开盘跳空受控"),
])
def test_continuation_rejects_incomplete_or_deteriorating_setups(case, check):
    frame = continuation_bars()
    if case == "no_base":
        frame.loc[158, "high"] = 13.
    elif case == "late_peak":
        frame.loc[156, "high"] = 12.3
    elif case == "weak_advance":
        frame.loc[131:151, ["low", "open", "close", "high"]] = [11.2, 11.25, 11.3, 11.35]
        frame.loc[151, ["low", "open", "close", "high"]] = [11.8, 11.9, 12., 12.12]
    elif case == "falling_trend":
        frame.loc[:130, ["low", "open", "close", "high"]] = [15., 15.1, 15.2, 15.3]
    elif case == "deep_pullback":
        frame.loc[154, "low"] = 10.5
    elif case == "no_shrink":
        frame.loc[152:158, "volume"] = 2100.
    elif case == "lost_support":
        frame.loc[158, ["low", "open", "close", "high"]] = [10.68, 10.73, 10.75, 11.5]
    elif case == "no_breakout":
        frame.loc[159, ["open", "low", "close", "high"]] = [11.85, 11.8, 12.1, 12.2]
    elif case == "too_far_above_peak":
        frame.loc[159, ["close", "high"]] = [13.2, 13.3]
    elif case == "large_gap":
        frame.loc[159, "open"] = 12.5
    frame = prepare_bars(frame, AS_OF)
    inspected = trend.inspect_continuation(frame)
    assert not inspected["matched"] and not inspected["checks"][check]
    metrics, rejection = inspect_bottom_reversal(frame)
    if case == "weak_advance":
        # 中继所需的前段涨幅不足，但收敛后的首段突破可由新分支独立确认。
        from src.pattern_screener import VOLUME_BREAKOUT_NAME
        assert rejection is None and metrics["setup"] == VOLUME_BREAKOUT_NAME
        assert metrics["volume_breakout"]["matched"]
    else:
        assert rejection is not None


@pytest.mark.parametrize("case,reason", [
    ("weak_volume", "当日放量不足1.8倍"), ("weak_close", "收盘回落过多"),
    ("negative_body", "当日上涨不足2%或非阳线"),
])
def test_continuation_cannot_bypass_common_daily_quality(case, reason):
    frame = continuation_bars()
    if case == "weak_volume": frame.loc[159, "volume"] = 1600.
    elif case == "weak_close": frame.loc[159, "high"] = 14.
    else: frame.loc[159, ["open", "close", "high"]] = [12.3, 12.28, 12.4]
    metrics, rejection = inspect_bottom_reversal(prepare_bars(frame, AS_OF))
    assert metrics["continuation"]["matched"]
    assert rejection == reason


@pytest.mark.parametrize("problem,status", [("flat", "complete"), ("error", "incomplete")])
def test_continuation_requires_real_turnover_and_propagates_missing_data(problem, status):
    result = screen_market(ContinuationProvider(problem), AS_OF, profile=PROFILE)
    assert result["status"] == status and result["selected"] == []
    if problem == "error": assert result["turnover_coverage"] == 0


def test_future_bars_cannot_create_a_historical_continuation():
    frame = continuation_bars()
    future = frame.iloc[-1:].assign(date=pd.Timestamp("2026-10-26"), close=100, high=101, volume=1e9)
    assert inspect_bottom_reversal(prepare_bars(pd.concat([frame, future]), AS_OF)) == inspect_bottom_reversal(frame)


def test_continuation_reaches_real_daily_cli_and_github_stock_handoff(tmp_path, monkeypatch):
    from scripts import screen_eight_patterns as entry
    from tests.test_pattern_entrypoint import set_clock
    set_clock(monkeypatch, datetime(2026, 10, 23, 18, tzinfo=ZoneInfo("Asia/Shanghai")), AS_OF, True)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "handoff"))
    monkeypatch.setattr(entry, "PatternMarketData", lambda *args: ContinuationProvider())
    monkeypatch.setattr(entry, "enrich_shareholder_counts", lambda *args, **kwargs: None)
    assert entry.main(["--profile", PROFILE, "--output-dir", str(tmp_path)]) == 0
    result = json.loads((tmp_path / "screen.json").read_text())
    assert result["ruleset_version"] == RULESET_VERSION
    assert result["selected"][0]["signals"][0]["name"] == trend.SIGNAL_NAME
    assert "stock_codes=000001\n" in (tmp_path / "handoff").read_text()
