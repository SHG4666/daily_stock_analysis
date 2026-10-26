"""每日偏好的正反例、真实换手率契约与覆盖门槛。"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.bottom_reversal import PROFILE, RULESET_VERSION, confirm_turnover, inspect_bottom_reversal
from src.pattern_screener import ScreeningDataError, prepare_bars
from src.services.pattern_screening_service import screen_market, write_result

AS_OF = date(2026, 10, 23)


def reversal_bars():
    close = np.r_[np.linspace(18, 15, 100), np.linspace(15, 10, 59), 10.5]
    frame = pd.DataFrame({"date": pd.bdate_range(end=AS_OF, periods=160), "close": close,
                          "open": close * .995, "high": close * 1.01,
                          "low": close * .99, "volume": 1000.})
    frame.loc[159, ["open", "high", "low", "volume"]] = [10.05, 10.6, 9.95, 2200]
    return frame


def turnover_bars(frame):
    return pd.DataFrame({"date": frame.date.tail(6), "turnover": [1., 1.2, .9, 1.1, 1., 2.2]})


class Provider:
    source_name = "测试日K"
    universe_source = "测试市场"
    universe_url = history_url = turnover_url = "https://example.invalid"
    turnover_source = "测试实际换手率"

    def __init__(self, turnover_problem=None):
        self.frame = reversal_bars()
        self.turnover_problem = turnover_problem

    def get_universe(self):
        return [{"code": "002727", "name": "测试股票", "open": 10.05, "price": 10.5,
                 "amount": 30_000_000, "volume": 2200}]

    def get_daily_data(self, *args):
        return self.frame

    def get_turnover_data(self, *args):
        if self.turnover_problem == "error":
            raise TimeoutError()
        result = turnover_bars(self.frame)
        if self.turnover_problem == "flat":
            result["turnover"] = 1.
        return result


def test_full_path_accepts_today_bottom_rally_with_actual_turnover(tmp_path):
    result = screen_market(Provider(), AS_OF, profile=PROFILE)
    assert result["status"] == "complete" and result["turnover_coverage"] == 1
    assert result["selected"][0]["bottom_reversal"]["turnover"] == 2.2
    assert result["ruleset_version"] == RULESET_VERSION
    write_result(result, tmp_path)
    rendered = (tmp_path / "report.html").read_text()
    assert "长期下跌后底部放量上涨" in rendered and "2.20%" in rendered
    assert "https://xueqiu.com/S/SZ002727" in rendered
    assert "换手率未明显提高" not in result["excluded"]


@pytest.mark.parametrize("problem,status", [("flat", "complete"), ("error", "incomplete")])
def test_volume_expansion_cannot_replace_turnover_and_failure_is_not_no_match(problem, status):
    result = screen_market(Provider(problem), AS_OF, profile=PROFILE)
    assert result["status"] == status and result["selected"] == []
    if problem == "error":
        assert result["turnover_coverage"] == 0 and result["data_issues"]["turnover_data"] == 1
    else:
        assert result["excluded"]["换手率未明显提高"] == 1


@pytest.mark.parametrize("case,reason", [
    ("uptrend", "此前60日跌幅不足15%"), ("high_bounce", "离60日低点过远"),
    ("weak_volume", "当日放量不足1.8倍"), ("red_but_down", "当日上涨不足2%或非阳线"),
    ("upper_wick", "收盘回落过多"), ("one_day_crash", "缺少持续下跌背景"),
])
def test_counterexamples_are_excluded(case, reason):
    frame = reversal_bars()
    if case == "uptrend":
        frame.loc[:158, "close"] = np.linspace(5, 10, 159)
    elif case == "high_bounce":
        frame.loc[120, "low"] = 5
    elif case == "weak_volume":
        frame.loc[159, "volume"] = 1500
    elif case == "red_but_down":
        frame.loc[159, ["open", "close"]] = [9.7, 9.9]
    elif case == "upper_wick":
        frame.loc[159, "high"] = 11.5
    elif case == "one_day_crash":
        frame.loc[:157, "close"] = 15
    _, rejection = inspect_bottom_reversal(frame)
    assert rejection == reason


@pytest.mark.parametrize("problem", ["missing", "stale", "duplicate", "nan", "zero", "text"])
def test_turnover_requires_six_exact_sessions_without_filling(problem):
    frame = reversal_bars()
    data = turnover_bars(frame).reset_index(drop=True)
    if problem == "missing":
        data = data.drop(2)
    elif problem == "stale":
        data = data.iloc[:-1]
    elif problem == "duplicate":
        data = pd.concat([data, data.iloc[-1:]])
    else:
        data["turnover"] = data["turnover"].astype(object)
        data.loc[5, "turnover"] = {"nan": float("nan"), "zero": 0, "text": "--"}[problem]
    with pytest.raises(ScreeningDataError, match="turnover_data"):
        confirm_turnover(frame, data)


def test_future_rows_do_not_change_historical_decision_or_turnover():
    frame = reversal_bars()
    future = frame.iloc[-1:].assign(date=pd.Timestamp("2026-10-26"), close=500, volume=1e10)
    prepared = prepare_bars(pd.concat([frame, future]), AS_OF)
    assert inspect_bottom_reversal(prepared) == inspect_bottom_reversal(frame)
    data = turnover_bars(frame)
    assert confirm_turnover(frame, data) == confirm_turnover(
        frame, pd.concat([data, data.iloc[-1:].assign(date=pd.Timestamp("2026-10-26"), turnover=99)]))


def test_unqualified_technicals_do_not_request_turnover_or_force_a_pick():
    provider = Provider()
    provider.frame["volume"] = 1000.
    provider.get_turnover_data = lambda *args: pytest.fail("should prefilter before requesting turnover")
    result = screen_market(provider, AS_OF, profile=PROFILE)
    assert result["status"] == "complete" and result["selected"] == []
    assert result["turnover_required"] == 0


def test_small_turnover_outage_cannot_hide_behind_high_whole_market_coverage():
    provider = Provider("error")
    original = provider.get_universe()[0]
    provider.get_universe = lambda: [original] + [dict(original, code=f"{i:06}") for i in range(100)]
    def bars(code, as_of):
        frame = reversal_bars()
        if code != "002727":
            frame["volume"] = 1000.
        return frame
    provider.get_daily_data = bars
    result = screen_market(provider, AS_OF, profile=PROFILE)
    assert result["coverage"] > .98 and result["turnover_coverage"] == 0
    assert result["status"] == "incomplete" and result["selected"] == []


def test_reversal_ranking_uses_turnover_instead_of_pattern_count():
    provider = Provider()
    original = provider.get_universe()[0]
    provider.get_universe = lambda: [original, dict(original, code="000001")]
    def turnover(code, as_of):
        data = turnover_bars(provider.frame)
        if code == "002727":
            data.iloc[-1, 1] = 3.3
        return data
    provider.get_turnover_data = turnover
    result = screen_market(provider, AS_OF, profile=PROFILE)
    assert [item["code"] for item in result["selected"]] == ["002727", "000001"]


def test_quote_return_prefilter_reduces_downloads_but_does_not_replace_candle_confirmation():
    provider = Provider()
    stock = provider.get_universe()[0]
    provider.get_universe = lambda: [dict(stock, change_percent=1.9)]
    provider.get_daily_data = lambda *args: pytest.fail("known insufficient return should not download")
    result = screen_market(provider, AS_OF, profile=PROFILE)
    assert result["excluded"]["报价涨幅不足2%"] == 1 and result["selected"] == []
    provider.get_universe = lambda: [dict(stock, change_percent=5.)]
    frame = reversal_bars()
    frame.loc[159, ["close", "open"]] = [10.1, 10.05]
    provider.get_daily_data = lambda *args: frame
    provider.get_universe = lambda: [dict(stock, price=10.1, change_percent=5.)]
    result = screen_market(provider, AS_OF, profile=PROFILE)
    assert result["excluded"]["当日上涨不足2%或非阳线"] == 1 and result["selected"] == []


def rebound_bars():
    """合成结构样本：长期下跌、近期探底、回升后放量突破；不是图片原股。"""
    frame = reversal_bars()
    closes = [10.3, 10.1, 10., 10.2, 10.55, 10.7, 10.6, 10.8, 11.1, 12.15]
    frame.loc[150:, "close"] = closes
    frame.loc[150:, "open"] = np.array(closes) * .995
    frame.loc[150:, "high"] = np.array(closes) * 1.01
    frame.loc[150:, "low"] = np.array(closes) * .99
    frame.loc[159, ["open", "low", "high", "volume"]] = [11.15, 11.05, 12.25, 2400]
    return frame


def test_recent_bottom_rebound_extends_pool_and_explains_the_actual_branch(tmp_path):
    from src.bottom_reversal import REBOUND_SIGNAL_NAME, RULESET_VERSION
    provider = Provider()
    provider.frame = prepare_bars(rebound_bars(), AS_OF)
    provider.get_universe = lambda: [{"code": "002727", "name": "合成走势样本", "open": 11.15,
                                    "price": 12.15, "amount": 30_000_000, "volume": 2400}]
    metrics, rejection = inspect_bottom_reversal(provider.frame)
    assert metrics["bottom_distance"] > .20  # Old rule excluded this stock.
    assert rejection is None and metrics["rebound"]["matched"]
    assert all(metrics["rebound"]["checks"].values())
    result = screen_market(provider, AS_OF, profile=PROFILE)
    assert result["status"] == "complete" and len(result["selected"]) == 1
    assert result["ruleset_version"] == RULESET_VERSION
    candidate = result["selected"][0]
    assert [item["name"] for item in candidate["signals"]] == [REBOUND_SIGNAL_NAME]
    assert candidate["bottom_reversal"]["turnover"] == 2.2
    write_result(result, tmp_path)
    rendered = (tmp_path / "report.html").read_text()
    assert "探底日期" in rendered and "MA5/10/20" in rendered
    assert "https://xueqiu.com/S/SZ002727" in rendered and "<svg" not in rendered


@pytest.mark.parametrize("problem,failed_check", [
    ("old_low", "近期探底"), ("yesterday_low", "近期探底"), ("fresh_low", "未再破底"),
    ("no_breakout", "突破前五日高点"), ("too_far", "反弹位置受控"),
    ("overheated", "前五日涨幅受控"), ("no_cross", "短均线新近金叉"), ("ma10_down", "短均线向上"),
])
def test_rebound_extension_rejects_missing_structure_and_chasing(problem, failed_check):
    frame = rebound_bars()
    if problem == "old_low":
        frame.loc[130, "low"] = 9.8
    elif problem == "yesterday_low":
        frame.loc[158, "low"] = 9.8
    elif problem == "fresh_low":
        frame.loc[159, "low"] = 9.8
    elif problem == "no_breakout":
        frame.loc[157, "high"] = 12.5
    elif problem == "too_far":
        frame.loc[159, ["close", "high"]] = [13., 13.1]
    elif problem == "overheated":
        frame.loc[153, ["open", "close", "low", "high"]] = [9.5, 9.55, 9.4, 9.6]
    elif problem == "no_cross":
        prices = np.array([10, 9.9, 10, 10.1, 10.2])
        for column, factor in (("close", 1), ("open", .995), ("low", .99), ("high", 1.01)):
            frame.loc[145:149, column] = prices * factor
    elif problem == "ma10_down":
        frame.loc[149, ["open", "close", "low", "high"]] = [13.9, 14., 13.8, 14.1]
    metrics, rejection = inspect_bottom_reversal(prepare_bars(frame, AS_OF))
    assert not metrics["rebound"]["matched"], metrics["rebound"]
    assert not metrics["rebound"]["checks"][failed_check]
    assert rejection is not None  # Must not bypass the baseline position limit.


@pytest.mark.parametrize("problem,reason", [
    ("uptrend", "此前60日跌幅不足15%"), ("weak_volume", "当日放量不足1.8倍"),
    ("weak_close", "收盘回落过多"), ("down_day", "当日上涨不足2%或非阳线"),
])
def test_rebound_never_bypasses_shared_quality_gates(problem, reason):
    frame = rebound_bars()
    if problem == "uptrend":
        frame.loc[99, "close"] = 11.5
    elif problem == "weak_volume":
        frame.loc[159, "volume"] = 1000
    elif problem == "weak_close":
        frame.loc[159, "high"] = 15
    elif problem == "down_day":
        frame.loc[159, "open"] = 12.2
    assert inspect_bottom_reversal(frame)[1] == reason


@pytest.mark.parametrize("turnover_problem,expected_status", [("flat", "complete"), ("error", "incomplete")])
def test_rebound_still_requires_real_turnover(turnover_problem, expected_status):
    provider = Provider(turnover_problem)
    provider.frame = rebound_bars()
    stock = provider.get_universe()[0]
    provider.get_universe = lambda: [dict(stock, open=11.15, price=12.15)]
    result = screen_market(provider, AS_OF, profile=PROFILE)
    assert result["status"] == expected_status and result["selected"] == []


def test_rebound_historical_decision_does_not_use_future_recovery():
    frame = rebound_bars()
    future = frame.iloc[-1:].assign(date=pd.Timestamp("2026-10-26"), close=40, high=41, volume=1e9)
    assert inspect_bottom_reversal(prepare_bars(pd.concat([frame, future]), AS_OF)) == inspect_bottom_reversal(frame)


def test_recent_rebound_can_have_more_than_ten_percent_prior_rise_without_losing_the_cap():
    frame = rebound_bars()
    frame.loc[153, ["open", "close", "low", "high"]] = [9.85, 9.9, 9.8, 10.]
    metrics, rejection = inspect_bottom_reversal(prepare_bars(frame, AS_OF))
    assert .10 < metrics["prior_five_session_return"] < .15
    assert rejection is None and metrics["rebound"]["matched"]


def test_new_shape_is_not_mandatory_for_the_original_bottom_reversal():
    from src.bottom_reversal import SIGNAL_NAME
    metrics, rejection = inspect_bottom_reversal(prepare_bars(reversal_bars(), AS_OF))
    assert rejection is None and not metrics["rebound"]["matched"]
    assert metrics["setup"] == SIGNAL_NAME
