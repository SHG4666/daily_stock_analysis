"""Safety and data-contract checks for standalone report charts."""

import json
import re
from datetime import date, timedelta

import pytest

from src.services.pattern_chart_renderer import render_stock_chart


def candidate(count=120):
    bars = []
    for index in range(count):
        bars.append({
            "date": (date(2026, 1, 1) + timedelta(days=index)).isoformat(),
            "open": 10.0, "high": 12.0, "low": 9.0,
            "close": 11.0 if index % 2 else 9.5, "volume": 100_000.0,
            "ma5": 10.0, "ma10": 10.1, "ma20": 10.2, "ma30": None,
        })
    return {"name": "示例股票", "code": "600001", "data_date": bars[-1]["date"], "chart_bars": bars}


def embedded_data(fragment):
    return json.loads(re.search(r'class="stock-chart-data">(.*?)</script>', fragment, re.S)[1])


def test_chart_embeds_exact_share_volume_and_precomputed_moving_averages():
    stock = candidate(150)
    fragment = render_stock_chart(stock)
    data = embedded_data(fragment)
    assert len(data) == 120
    assert data[0]["date"] == stock["chart_bars"][-120]["date"]
    assert data[-1]["volume"] == 100_000
    assert data[-1]["ma30"] is None
    assert "量 10.00 万股" in fragment
    assert "MA30 —" in fragment
    assert "前复权 · 截至" in fragment
    assert 'data-chart-range="30"' in fragment
    assert 'data-chart-range="60" aria-pressed="true"' in fragment
    assert 'data-chart-range="120"' in fragment


def test_stock_labels_cannot_inject_html_or_terminate_the_data_script():
    stock = candidate()
    stock["name"] = '</script><img src=x onerror="bad()">'
    stock["code"] = '"><script>bad()</script>'
    fragment = render_stock_chart(stock)
    assert "<img src=" not in fragment
    assert '<script>bad()' not in fragment
    assert "&lt;/script&gt;" in fragment
    assert len(re.findall(r"</script>", fragment)) == 1
    assert embedded_data(fragment)[-1]["date"] == stock["data_date"]


@pytest.mark.parametrize("mutate", [
    lambda stock: stock["chart_bars"][-1].update(close=float("nan")),
    lambda stock: stock["chart_bars"][-1].update(high=8),
    lambda stock: stock["chart_bars"][-1].update(volume=-1),
    lambda stock: stock["chart_bars"][-1].update(ma5=float("inf")),
    lambda stock: stock["chart_bars"][-1].update(date=stock["chart_bars"][-2]["date"]),
    lambda stock: stock.update(data_date="2026-10-12"),
    lambda stock: stock["chart_bars"][-1].update(date="2026-01-01<script>"),
])
def test_invalid_or_stale_bars_show_clear_unavailable_message(mutate):
    stock = candidate()
    mutate(stock)
    fragment = render_stock_chart(stock)
    assert "未包含可用的日 K 历史" in fragment
    assert "<svg" not in fragment


def test_legacy_candidate_has_no_fake_candles():
    fragment = render_stock_chart({"name": "旧报告", "closes": [10, 11, 12]})
    assert "暂无法在此展示走势图" in fragment
    assert "<svg" not in fragment


def test_initial_svg_includes_mainland_red_up_green_down_price_and_volume():
    fragment = render_stock_chart(candidate())
    assert fragment.count('<rect ') == 120  # 60 price candles and their real volumes.
    assert 'fill="#dc4848"' in fragment
    assert 'fill="#218869"' in fragment
    assert '成交量（万股）' in fragment
    assert 'stroke="#b98a19"' in fragment
    assert 'stroke="#437abd"' in fragment
    assert 'stroke="#9968b1"' in fragment
    assert 'stroke="#4f8263"' not in fragment  # Missing MA30 values are not invented.
