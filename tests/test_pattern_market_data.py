"""公开行情解析、缓存日期与分页完整性的确定性检查。"""
import json
from datetime import date, datetime
from unittest.mock import Mock, patch

import pytest
import requests
from zoneinfo import ZoneInfo

from data_provider import pattern_market_data as market_data
from data_provider.pattern_market_data import PatternMarketData
from src.core.trading_calendar import MarketPhase


def response(data):
    result = Mock()
    result.json.return_value = data
    result.text = json.dumps(data, ensure_ascii=False)
    return result


def test_actual_turnover_is_eighth_field_and_reuses_same_day_cache(tmp_path):
    provider = PatternMarketData(tmp_path)
    rows = [["2026-10-22", "13.25", "12.97", "13.26", "12.77", "155431", {}, "3.91", "20150.41", ""]]
    with patch("requests.get", return_value=history(rows)) as get:
        frame = provider.get_daily_data("600519", date(2026, 10, 22))
        rates = provider.get_turnover_data("600519", date(2026, 10, 22))
    assert get.call_count == 1
    assert frame.volume.iloc[0] == 15543100
    assert rates.to_dict("records") == [{"date": "2026-10-22", "turnover": 3.91}]
    assert "amount" not in frame


@pytest.mark.parametrize("extra", [[], [{}, "--"], [{}, ""], [{}, None]])
def test_legacy_or_missing_turnover_never_uses_volume_or_amount(tmp_path, extra):
    import math
    row = ["2026-10-22", "10", "11", "11", "10", "123"] + extra
    with patch("requests.get", return_value=history([row])):
        rates = PatternMarketData(tmp_path).get_turnover_data("600519", date(2026, 10, 22))
    assert math.isnan(rates.turnover.iloc[0])


def history(rows, key="qfqday"):
    return response({"code": 0, "data": {"sh600519": {key: rows}}})


def test_tencent_sixth_field_is_volume_and_cache_is_same_day_only(tmp_path):
    provider = PatternMarketData(tmp_path)
    rows = [["2026-10-09", "10", "11", "11.1", "9.9", "123"]]
    with patch("requests.get", return_value=history(rows)) as get:
        first = provider.get_daily_data("600519", date(2026, 10, 9))
        provider.get_daily_data("600519", date(2026, 10, 9))
    assert first.volume.iloc[0] == 12300
    assert "amount" not in first
    assert get.call_count == 1
    assert get.call_args.kwargs["timeout"] == (3, 8)
    with patch("requests.get", return_value=history(rows)) as get:
        provider.get_daily_data("600519", date(2026, 10, 12))
    assert get.call_count == 1


def test_never_silently_uses_unadjusted_day_array(tmp_path):
    provider = PatternMarketData(tmp_path)
    with patch("requests.get", return_value=history([["2026-10-09", 10, 10, 10, 10, 1]], "day")):
        with pytest.raises(ValueError, match="前复权"):
            provider.get_daily_data("600519", date(2026, 10, 9))


def test_corrupt_cache_is_refetched(tmp_path):
    (tmp_path / "600519.json").write_text("[]")
    provider = PatternMarketData(tmp_path)
    with patch("requests.get", return_value=history([["2026-10-09", 10, 10, 10, 10, 1]])) as get:
        provider.get_daily_data("600519", date(2026, 10, 9))
    assert get.call_count == 1


@pytest.mark.parametrize("incomplete", [False, True])
def test_universe_pages_must_be_complete_and_unique(tmp_path, incomplete):
    def get(url, **kwargs):
        if url.endswith("StockCount"):
            return response("1000")
        page = kwargs["params"]["page"]
        rows = [{"code": f"{index:06d}", "name": "样本", "open": 1, "trade": 2,
                 "volume": 100, "amount": 1000} for index in range((page-1)*100, page*100)]
        return response(rows[:-1] if incomplete and page == 10 else rows)
    provider = PatternMarketData(tmp_path)
    with patch("requests.get", side_effect=get):
        if incomplete:
            with pytest.raises(ValueError, match="不完整"):
                provider.get_universe()
        else:
            assert len(provider.get_universe()) == 1000


def fixed_clock(monkeypatch, now, phase):
    monkeypatch.setattr(market_data, "get_market_now", lambda market: now)
    monkeypatch.setattr(market_data, "infer_market_phase", lambda *a, **kwargs: phase)


def candle(day):
    return [day, "10", "11", "11.1", "9.9", "123"]


@pytest.mark.parametrize("now,target,phase,expected_end", [
    (datetime(2026, 10, 22, 15, 47, tzinfo=ZoneInfo("Asia/Shanghai")), date(2026, 10, 22), MarketPhase.POSTMARKET, ""),
    (datetime(2026, 10, 22, 15, 47, tzinfo=ZoneInfo("Asia/Shanghai")), date(2026, 10, 22), MarketPhase.UNKNOWN, "2026-10-22"),
    (datetime(2026, 10, 22, 14, tzinfo=ZoneInfo("Asia/Shanghai")), date(2026, 10, 22), MarketPhase.INTRADAY, "2026-10-22"),
    (datetime(2026, 10, 22, 15, 47, tzinfo=ZoneInfo("Asia/Shanghai")), date(2026, 10, 21), MarketPhase.POSTMARKET, "2026-10-21"),
    (datetime(2026, 10, 23, 0, 20, tzinfo=ZoneInfo("Asia/Shanghai")), date(2026, 10, 22), MarketPhase.PREMARKET, "2026-10-22"),
])
def test_latest_endpoint_only_for_same_day_confirmed_postmarket(tmp_path, monkeypatch, now, target, phase, expected_end):
    fixed_clock(monkeypatch, now, phase)
    provider = PatternMarketData(tmp_path)
    with patch("requests.get", return_value=history([candle(target.isoformat())])) as get:
        frame = provider.get_daily_data("600519", target)
    assert get.call_args.kwargs["params"]["param"] == f"sh600519,day,,{expected_end},240,qfq"
    assert frame.date.iloc[-1] == target.isoformat()


@pytest.mark.parametrize("target", [date(2026, 10, 21), date(2026, 10, 22)])
def test_future_rows_are_clipped_before_validation_and_caching(tmp_path, monkeypatch, target):
    fixed_clock(monkeypatch, datetime(2026, 10, 22, 16, tzinfo=ZoneInfo("Asia/Shanghai")), MarketPhase.POSTMARKET)
    provider = PatternMarketData(tmp_path)
    # A future row can be incomplete and must not invalidate the frozen day.
    rows = [candle("2026-10-20"), candle(target.isoformat()), ["2026-10-23"]]
    with patch("requests.get", return_value=history(rows)):
        frame = provider.get_daily_data("600519", target)
    assert frame.date.tolist() == ["2026-10-20", target.isoformat()]
    cached = json.loads((tmp_path / "600519.json").read_text())
    assert cached["rows"] == rows[:2]


def test_poison_cache_label_cannot_hide_stale_actual_last_bar(tmp_path, monkeypatch):
    fixed_clock(monkeypatch, datetime(2026, 10, 22, 16, tzinfo=ZoneInfo("Asia/Shanghai")), MarketPhase.POSTMARKET)
    path = tmp_path / "600519.json"
    path.write_text(json.dumps({"as_of": "2026-10-22", "code": "600519", "source": "tencent_qfq_v1",
                                "rows": [candle("2026-10-21")]}))
    with patch("requests.get", return_value=history([candle("2026-10-22")])) as get:
        provider = PatternMarketData(tmp_path)
        assert provider.get_daily_data("600519", date(2026, 10, 22)).date.iloc[-1] == "2026-10-22"
        provider.get_daily_data("600519", date(2026, 10, 22))
    assert get.call_count == 1
    assert json.loads(path.read_text())["rows"][-1][0] == "2026-10-22"


def test_stale_response_is_not_cached_and_still_reaches_existing_freshness_guard(tmp_path, monkeypatch):
    import pandas as pd
    from src.pattern_screener import ScreeningDataError, prepare_bars

    fixed_clock(monkeypatch, datetime(2026, 10, 22, 16, tzinfo=ZoneInfo("Asia/Shanghai")), MarketPhase.POSTMARKET)
    provider = PatternMarketData(tmp_path)
    rows = [candle(day.strftime("%Y-%m-%d")) for day in pd.bdate_range(end="2026-10-21", periods=100)]
    path = tmp_path / "600519.json"
    # Also reproduce an already-written poisoned cache from the old adapter.
    path.write_text(json.dumps({"as_of": "2026-10-22", "code": "600519", "source": "tencent_qfq_v1", "rows": rows}))
    with patch("requests.get", return_value=history(rows)) as get:
        first = provider.get_daily_data("600519", date(2026, 10, 22))
        provider.get_daily_data("600519", date(2026, 10, 22))
    assert get.call_count == 2 and not path.exists()
    with pytest.raises(ScreeningDataError, match="stale"):
        prepare_bars(first, date(2026, 10, 22))


def test_existing_future_cache_is_bounded_before_use(tmp_path, monkeypatch):
    path = tmp_path / "600519.json"
    path.write_text(json.dumps({"as_of": "2026-10-22", "code": "600519", "source": "tencent_qfq_v1",
                                "rows": [candle("2026-10-22"), ["2026-10-23"]]}))
    with patch("requests.get", side_effect=AssertionError("bounded valid cache should not fetch")):
        frame = PatternMarketData(tmp_path).get_daily_data("600519", date(2026, 10, 22))
    assert frame.date.tolist() == ["2026-10-22"]


def test_only_transient_list_gets_retry_once_with_safe_stage_logs(caplog):
    with patch("requests.get", side_effect=[requests.ReadTimeout("private-url-should-not-be-logged"), response("1000")]) as get, \
            patch.object(market_data.time, "sleep") as sleep:
        assert PatternMarketData._get("https://example.test/list", stage="sina_count", attempts=2).text == '"1000"'
    assert get.call_count == 2
    sleep.assert_called_once_with(1)
    assert "stage=sina_count" in caplog.text and "error=ReadTimeout" in caplog.text
    assert "private-url" not in caplog.text and "https://" not in caplog.text


def test_list_retry_budget_is_bounded_and_tencent_does_not_expand_it():
    with patch("requests.get", side_effect=requests.ConnectionError("offline")) as get, \
            patch.object(market_data.time, "sleep") as sleep:
        with pytest.raises(requests.ConnectionError):
            PatternMarketData._get("https://example.test/list", stage="sina_page_3", attempts=2)
    assert get.call_count == 2 and sleep.call_count == 1
    with patch("requests.get", side_effect=requests.ReadTimeout("offline")) as get, \
            patch.object(market_data.time, "sleep") as sleep:
        with pytest.raises(requests.ReadTimeout):
            PatternMarketData._get("https://example.test/history")
    assert get.call_count == 1 and not sleep.called


def test_http_failure_and_invalid_universe_count_do_not_retry(tmp_path):
    failed = response({})
    failed.raise_for_status.side_effect = requests.HTTPError("bad status")
    with patch("requests.get", return_value=failed) as get, patch.object(market_data.time, "sleep") as sleep:
        with pytest.raises(requests.HTTPError):
            PatternMarketData._get("https://example.test/list", stage="sina_count", attempts=2)
    assert get.call_count == 1 and not sleep.called
    with patch("requests.get", return_value=response("10")) as get:
        with pytest.raises(ValueError, match="数量异常"):
            PatternMarketData(tmp_path).get_universe()
    assert get.call_count == 1
