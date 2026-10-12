"""日程入口的休市、盘中与上游失败行为，保证不回退旧股票池。"""
import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
import requests

from scripts import screen_eight_patterns as entry


def set_clock(monkeypatch, now, effective, market_open):
    monkeypatch.setattr("src.config.setup_env", lambda: None)
    monkeypatch.setattr(entry, "get_market_now", lambda market: now)
    monkeypatch.setattr(entry, "get_effective_trading_date", lambda *a, **k: effective)
    monkeypatch.setattr(entry, "is_market_open", lambda *a: market_open)


def test_holiday_is_normal_skip_without_fetching_or_old_stock_pool(tmp_path, monkeypatch):
    set_clock(monkeypatch, datetime(2026, 10, 11, 18, tzinfo=ZoneInfo("Asia/Shanghai")),
              date(2026, 10, 9), False)
    monkeypatch.setattr(entry.PatternMarketData, "__init__", lambda *a: pytest.fail("must not fetch"))
    output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert entry.main(["--output-dir", str(tmp_path)]) == 0
    assert "status=market_closed" in output.read_text()
    assert "has_candidates=false" in output.read_text()
    assert (tmp_path / "stocks.txt").read_text() == ""


def test_intraday_does_not_use_yesterdays_signals(tmp_path, monkeypatch):
    set_clock(monkeypatch, datetime(2026, 10, 12, 14, tzinfo=ZoneInfo("Asia/Shanghai")),
              date(2026, 10, 9), True)
    monkeypatch.setattr(entry.PatternMarketData, "__init__", lambda *a: pytest.fail("must not fetch"))
    assert entry.main(["--output-dir", str(tmp_path)]) == 2
    assert not (tmp_path / "stocks.txt").exists()


def test_universe_error_writes_failure_report_and_no_stock_pool(tmp_path, monkeypatch):
    set_clock(monkeypatch, datetime(2026, 10, 12, 18, tzinfo=ZoneInfo("Asia/Shanghai")),
              date(2026, 10, 12), True)
    def fail(*args):
        raise requests.ConnectionError("upstream unavailable")
    monkeypatch.setattr(entry.PatternMarketData, "get_universe", fail)
    assert entry.main(["--output-dir", str(tmp_path), "--cache-dir", str(tmp_path / "cache")]) == 2
    result = json.loads((tmp_path / "screen.json").read_text())
    assert result["status"] == "incomplete"
    assert result["selected"] == []
    assert "ConnectionError" in result["data_issues"]


def test_nonfinite_liquidity_configuration_is_rejected():
    with pytest.raises(SystemExit):
        entry.parse_args(["--min-amount", "nan"])
