"""公开行情解析、缓存日期与分页完整性的确定性检查。"""
import json
from datetime import date
from unittest.mock import Mock, patch

import pytest

from data_provider.pattern_market_data import PatternMarketData


def response(data):
    result = Mock()
    result.json.return_value = data
    result.text = json.dumps(data, ensure_ascii=False)
    return result


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
