"""筛选结果到股票池的端到端离线检查。"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.services.pattern_screening_service import screen_market, write_result


AS_OF = date(2026, 10, 12)


class FakeProvider:
    source_name = "测试前复权"
    history_url = "https://example.invalid/daily"
    universe_source = "测试范围"
    universe_url = "https://example.invalid/list"

    def __init__(self, problem=None):
        self.problem = problem

    def get_universe(self):
        return [{"code": "000001", "name": "测试银行", "open": 9.99, "price": 10.8,
                 "amount": 30_000_000, "volume": 250000}]

    def get_daily_data(self, code, as_of):
        if self.problem == "error":
            raise TimeoutError("test")
        count = 80 if self.problem == "short" else 160
        close = np.full(count, 10.0)
        close[-1] = 10.8
        frame = pd.DataFrame({
            "date": pd.bdate_range(end=AS_OF, periods=count),
            "open": np.full(count, 9.99), "close": close,
            "high": close + .02, "low": np.full(count, 9.98),
            "volume": np.full(count, 1000),
        })
        frame.loc[count-1, "volume"] = 2500
        return frame.iloc[:-1] if self.problem == "stale" else frame


def test_verified_matches_feed_current_stock_pool_without_overwriting_config(tmp_path):
    result = screen_market(FakeProvider(), AS_OF)
    assert result["status"] == "complete"
    assert result["selected"][0]["code"] == "000001"
    assert "一阳穿多线" in [signal["name"] for signal in result["selected"][0]["signals"]]
    write_result(result, tmp_path)
    assert (tmp_path / "stocks.txt").read_text() == "000001"
    assert "测试银行" in (tmp_path / "report.html").read_text()
    assert "2026-10-12" in (tmp_path / "report.html").read_text()


@pytest.mark.parametrize("problem", ["stale", "error"])
def test_missing_latest_data_never_falls_back_to_old_stock_list(problem):
    result = screen_market(FakeProvider(problem), AS_OF)
    assert result["status"] == "incomplete"
    assert result["selected"] == []
    assert result["coverage"] == 0


def test_short_history_is_explicit_exclusion_not_network_failure():
    result = screen_market(FakeProvider("short"), AS_OF)
    assert result["status"] == "complete"
    assert result["selected"] == []
    assert result["data_issues"] == {"short_history": 1}


def test_invalid_quote_is_excluded_before_history_lookup():
    provider = FakeProvider()
    provider.get_universe = lambda: [{"code": "000001", "name": "样本", "open": 1,
                                    "price": float("nan"), "amount": 100, "volume": 10}]
    result = screen_market(provider, AS_OF)
    assert result["eligible_count"] == 0
    assert result["excluded"] == {"无效报价": 1}
