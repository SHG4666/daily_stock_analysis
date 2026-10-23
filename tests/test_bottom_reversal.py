"""每日偏好的正反例、真实换手率契约与覆盖门槛。"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.bottom_reversal import PROFILE, confirm_turnover, inspect_bottom_reversal
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
    assert result["ruleset_version"] == "3.0-bottom-reversal"
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
