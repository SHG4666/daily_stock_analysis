"""复用腾讯解析器，按逐股行情时间戳采集全市场收盘快照。"""
import math
import re
from concurrent.futures import ThreadPoolExecutor

from data_provider.akshare_fetcher import _normalize_tencent_volume, _parse_tencent_amount, _to_sina_tx_symbol
from data_provider.pattern_market_data import PatternMarketData

COLUMNS = ("close", "previous_close", "open", "high", "low", "volume", "amount", "turnover")


def parse_batch(text, symbols, as_of):
    expected = {symbol: code for code, symbol in symbols.items()}
    rows = {}
    for symbol, body in re.findall(r'v_([a-z]+\d+)="([^"]*)"', text):
        if symbol not in expected or expected[symbol] in rows:
            continue
        fields = body.split("~")
        try:
            if (len(fields) < 45 or fields[2] != expected[symbol]
                    or not re.fullmatch(r"\d{14}", fields[30])
                    or fields[30][:8] != as_of.strftime("%Y%m%d")
                    or fields[30][8:] < "150000"):
                continue
            values = [float(fields[i]) for i in (3, 4, 5, 33, 34)]
            values += [_normalize_tencent_volume(fields), _parse_tencent_amount(fields), float(fields[38])]
            if not all(value is not None and math.isfinite(value) and value > 0 for value in values):
                continue  # 停牌/旧时间戳不当成零涨跌，也不填充。
            close, previous, opening, high, low = values[:5]
            if low > min(close, opening) or high < max(close, opening) or high < low:
                continue
            rows[expected[symbol]] = [round(value, 6) for value in values]
        except (ValueError, TypeError, IndexError):
            continue
    return rows


def collect_close_snapshot(provider, as_of):
    universe = provider.get_universe()
    symbols = {item["code"]: _to_sina_tx_symbol(item["code"]) for item in universe}
    chunks = [dict(list(symbols.items())[i:i+80]) for i in range(0, len(symbols), 80)]

    def fetch(chunk):
        try:
            response = PatternMarketData._get("https://qt.gtimg.cn/q=" + ",".join(chunk.values()),
                                             stage="learning_close")
            response.encoding = "gbk"
            return parse_batch(response.text, chunk, as_of)
        except Exception:
            return {}  # 缺口逐股计入覆盖；学习失败不能中断正常选股。

    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in pool.map(fetch, chunks):
            rows.update(batch)
    from data_provider.base import is_st_stock
    eligible = [item["code"] for item in universe if not is_st_stock(item["name"])
                and item["amount"] >= 20_000_000 and item["volume"] > 0]
    return {"as_of": as_of.isoformat(), "rows": rows, "eligible": eligible,
            "universe_count": len(universe), "coverage": len(rows) / len(universe) if universe else 0,
            "source": "腾讯逐股收盘快照（含行情日期），量股/额元/换手率百分数",
            "columns": list(COLUMNS)}
