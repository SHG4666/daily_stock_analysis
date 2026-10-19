"""筛选结果到股票池的端到端离线检查。"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pattern_screener import RULESET_VERSION, VOLUME_BREAKOUT_LIMITS, VOLUME_BREAKOUT_NAME, detect_patterns
from src.services.pattern_screening_service import rank_candidates, screen_market, write_result


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
    bars = result["selected"][0]["chart_bars"]
    assert len(bars) == 120
    assert bars[0]["ma30"] == 10  # Includes earlier data outside the displayed window.
    assert bars[-1]["date"] == "2026-10-12"
    assert bars[-1]["volume"] == 2500  # Provider already returns shares; no second conversion.
    assert bars[-1]["ma20"] == 10.04
    rendered = (tmp_path / "report.html").read_text()
    assert 'href="https://xueqiu.com/S/SZ000001">测试银行 <small>000001</small>' in rendered
    assert '<svg' not in rendered
    assert 'stock-chart' not in rendered


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


def test_real_volume_breakout_enters_daily_stock_pool_with_versioned_evidence(tmp_path):
    class HistoricalProvider(FakeProvider):
        def get_universe(self):
            return [{"code": "002727", "name": "一心堂", "open": 11.67, "price": 12.46,
                     "amount": 405798470.16, "volume": 33122800}]

        def get_daily_data(self, code, as_of):
            return pd.read_csv(Path(__file__).parent / "fixtures/patterns/yxt_002727_20261014.csv")

    provider = HistoricalProvider()
    as_of = date(2026, 10, 14)
    assert detect_patterns(provider.get_daily_data("002727", as_of), as_of, include_extensions=False) == []
    result = screen_market(provider, as_of)
    assert result["status"] == "complete" and result["coverage"] == 1
    assert result["ruleset_version"] == RULESET_VERSION
    assert result["parameters"]["volume_breakout"] == VOLUME_BREAKOUT_LIMITS
    candidate = result["selected"][0]
    assert candidate["code"] == "002727" and candidate["priority_count"] == 1
    assert [signal["name"] for signal in candidate["signals"]] == [VOLUME_BREAKOUT_NAME]
    assert candidate["signals"][0]["evidence"]["突破前20日高点"] == 11.98
    write_result(result, tmp_path)
    assert (tmp_path / "stocks.txt").read_text() == "002727"
    rendered = (tmp_path / "report.html").read_text()
    assert RULESET_VERSION in rendered and VOLUME_BREAKOUT_NAME in rendered
    assert 'href="https://xueqiu.com/S/SZ002727">一心堂' in rendered
    assert '<svg' not in rendered and 'stock-chart' not in rendered


def test_extension_confers_eligibility_without_duplicating_original_pattern_priority():
    def candidate(code, names, volume):
        return {"code": code, "signals": [{"name": name} for name in names], "volume_ratio": volume}

    candidates = [
        candidate("000001", ["老鸭头", "黄金坑"], 1),
        candidate("000002", ["黄金坑", VOLUME_BREAKOUT_NAME], 1.5),
        candidate("000003", ["黄金坑"], 2),
        candidate("000004", [VOLUME_BREAKOUT_NAME], 3),
    ]
    ranked = rank_candidates(candidates)
    assert [item["code"] for item in ranked] == ["000001", "000004", "000003", "000002"]
    assert [item["priority_count"] for item in ranked] == [2, 1, 1, 1]


def test_original_only_ranking_remains_count_then_volume_then_code():
    candidates = [
        {"code": "600001", "signals": [{"name": "黄金坑"}], "volume_ratio": 3},
        {"code": "000001", "signals": [{"name": "黄金坑"}], "volume_ratio": 3},
        {"code": "600002", "signals": [{"name": "黄金坑"}, {"name": "多方炮"}], "volume_ratio": 2},
    ]
    assert [item["code"] for item in rank_candidates(candidates)] == ["600002", "000001", "600001"]
