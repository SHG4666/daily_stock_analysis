"""全市场主备源：实际HTTP入口解析、完整性及失败切换。"""
import json
from datetime import date, datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
import requests

from data_provider import pattern_market_data as module
from data_provider.pattern_market_data import PatternMarketData


def response(payload=None, text=None):
    result = Mock()
    result.json.return_value = payload
    result.text = text if text is not None else json.dumps(payload)
    return result


def quote(symbol, *, stamp="20261028160000"):
    fields = [""] * 50
    for index, value in {1: "测试股票", 2: symbol[2:], 3: "10", 4: "9.7", 5: "9.8",
                         6: "1000", 30: stamp, 32: "3.09", 35: "10/1000/123456",
                         37: "12", 38: "2"}.items():
        fields[index] = value
    return f'v_{symbol}="' + "~".join(fields) + '";'


@pytest.fixture
def tencent_http(monkeypatch):
    symbols = [f"sh60{i:04d}" for i in range(1000)]
    symbols[-2:] = ["sz002727", "bj920220"]
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if "getBoardRankList" in url:
            offset = int(kwargs["params"]["offset"])
            return response({"code": 0, "data": {"total": len(symbols), "rank_list": [
                {"code": symbol, "name": "测试股票"} for symbol in symbols[offset:offset+200]
            ]}})
        if "qt.gtimg.cn/q=" in url:
            return response(text="\n".join(quote(symbol) for symbol in url.split("=", 1)[1].split(",")))
        raise AssertionError("主源正常时不得请求新浪")

    now = datetime(2026, 10, 28, 19, tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(module, "get_market_now", lambda market: now)
    monkeypatch.setattr(module, "get_effective_trading_date", lambda *a, **kw: date(2026, 10, 28))
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return get, calls


def test_primary_full_market_quotes_keep_identity_and_units(tmp_path, tencent_http):
    _, calls = tencent_http
    provider = PatternMarketData(tmp_path)
    rows = provider.get_universe()
    assert len(rows) == len({item["code"] for item in rows}) == 1000
    assert rows[-2]["code"] == "002727" and rows[-1]["code"] == "920220"
    assert rows[0] == {"code": "600000", "name": "测试股票", "open": 9.8, "price": 10,
                       "amount": 123456, "volume": 100000, "change_percent": 3.09}
    assert provider.universe_source.startswith("腾讯")
    assert len(calls) == 10 and all(kwargs["timeout"] == (3, 8) for _, kwargs in calls)


@pytest.mark.parametrize("fault", ["timeout", "missing_page", "duplicate_page", "changed_total",
                                   "business_error", "wrong_market", "missing_quote", "duplicate_quote",
                                   "wrong_quote_code", "old_quote", "intraday_quote", "bad_number"])
def test_primary_fault_falls_back_as_a_whole(tmp_path, monkeypatch, tencent_http, fault):
    original_get, calls = tencent_http

    def faulty_get(url, **kwargs):
        if "getBoardRankList" in url:
            if fault == "timeout":
                raise requests.ReadTimeout("private request details")
            result = original_get(url, **kwargs)
            payload = result.json()
            offset = int(kwargs["params"]["offset"])
            if fault == "missing_page" and offset == 200:
                payload["data"]["rank_list"].pop()
            elif fault == "duplicate_page" and offset == 200:
                payload["data"]["rank_list"][0]["code"] = "sh600000"
            elif fault == "changed_total" and offset == 200:
                payload["data"]["total"] += 1
            elif fault == "business_error":
                payload["code"] = 1
            elif fault == "wrong_market" and offset == 0:
                payload["data"]["rank_list"][0]["code"] = "sz600000"
            return result
        result = original_get(url, **kwargs)
        if "sh600000" in url:
            first = quote("sh600000")
            if fault == "missing_quote":
                result.text = result.text.replace(first, "")
            elif fault == "duplicate_quote":
                result.text += first
            elif fault == "wrong_quote_code":
                result.text = result.text.replace("~600000~", "~600001~", 1)
            elif fault == "old_quote":
                result.text = result.text.replace("20261028160000", "20261027160000", 1)
            elif fault == "intraday_quote":
                result.text = result.text.replace("20261028160000", "20261028145959", 1)
            elif fault == "bad_number":
                result.text = result.text.replace("~9.8~", "~nan~", 1)
        return result

    monkeypatch.setattr(module.requests, "get", faulty_get)
    backup = Mock(return_value=[{"code": "000001", "name": "备用完整列表"}])
    monkeypatch.setattr(PatternMarketData, "_get_sina_universe", backup)
    provider = PatternMarketData(tmp_path)
    assert provider.get_universe() == backup.return_value
    backup.assert_called_once_with()
    assert provider.universe_source.startswith("新浪") and "sina" in provider.universe_url
    if fault == "timeout":
        assert not calls


def test_both_sources_fail_with_observable_safe_errors(tmp_path, monkeypatch, caplog):
    provider = PatternMarketData(tmp_path)
    monkeypatch.setattr(provider, "_get_tencent_universe", Mock(side_effect=requests.ReadTimeout("private-url")))
    monkeypatch.setattr(provider, "_get_sina_universe", Mock(side_effect=ValueError("private-payload")))
    with pytest.raises(ValueError, match="tencent=ReadTimeout, sina=ValueError"):
        provider.get_universe()
    assert "所有数据源失败" in caplog.text
    assert "private" not in caplog.text


def test_primary_network_failure_reaches_real_sina_parser(tmp_path, monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    def get(url, **kwargs):
        if "finance.qq.com" in url:
            raise requests.ConnectionError("offline")
        if url.endswith("StockCount"):
            return response("1000")
        page = int(kwargs["params"]["page"])
        return response([{"code": f"60{i:04d}", "name": "备用股票", "open": "9.8", "trade": "10",
                          "amount": "123456", "volume": "100000", "changepercent": "3.09"}
                         for i in range((page-1)*100, page*100)])

    monkeypatch.setattr(module.requests, "get", get)
    provider = PatternMarketData(tmp_path)
    rows = provider.get_universe()
    assert len(rows) == 1000 and rows[0]["price"] == 10 and rows[0]["volume"] == 100000
    assert provider.universe_source.startswith("新浪")
