"""全市场形态筛选的批量行情入口：有超时的列表与一致前复权日 K。"""
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from data_provider.akshare_fetcher import _to_sina_tx_symbol
from src.core.trading_calendar import MarketPhase, get_market_now, infer_market_phase

logger = logging.getLogger(__name__)


class PatternMarketData:
    """复用既有代码市场映射；不把腾讯第六列成交量误作成交额。"""
    source_name = "腾讯前复权日K"
    universe_source = "新浪沪深京A股实时列表"
    universe_url = "https://vip.stock.finance.sina.com.cn/mkt/#hs_a"
    history_url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    turnover_source = "腾讯日K实际换手率"
    turnover_url = history_url

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get(url, *, stage="tencent_history", attempts=1, **kwargs):
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(url, timeout=(3, 8), **kwargs)
                response.raise_for_status()
                return response
            except (requests.Timeout, requests.ConnectionError) as exc:
                logger.warning("行情GET失败 stage=%s attempt=%s/%s error=%s", stage, attempt, attempts,
                               type(exc).__name__)
                if attempt == attempts:
                    raise
                time.sleep(1)

    def get_universe(self) -> list[dict]:
        from akshare.utils import demjson
        base = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center."
        response = self._get(base + "getHQNodeStockCount", stage="sina_count", attempts=2, params={"node": "hs_a"})
        count = int(str(demjson.decode(response.text)))
        if not 1000 <= count <= 15000:
            raise ValueError("A股列表数量异常，停止筛选")

        def page(number):
            response = self._get(base + "getHQNodeData", stage=f"sina_page_{number}", attempts=2, params={
                "page": number, "num": 100, "sort": "symbol", "asc": 1,
                "node": "hs_a", "symbol": "",
            })
            rows = demjson.decode(response.text)
            if not isinstance(rows, list):
                raise ValueError("股票列表返回格式异常")
            return rows

        with ThreadPoolExecutor(max_workers=4) as pool:
            pages = list(pool.map(page, range(1, math.ceil(count / 100) + 1)))
        raw = [item for part in pages for item in part]
        if len(raw) != count or len({item.get("code") for item in raw}) != count:
            raise ValueError("股票列表分页不完整或重复，停止筛选")
        result = []
        for item in raw:
            code = str(item.get("code", ""))
            if not re.fullmatch(r"(?:00|30|60|68|43|83|87|88|92)\d{4}", code):
                continue
            try:
                change_percent = float(item["changepercent"])
            except (KeyError, TypeError, ValueError):
                # Optional shortcut only: absent/placeholder quotes still need
                # the full candle check, and must not break the legacy profile.
                change_percent = None
            result.append({
                "code": code, "name": str(item.get("name", "")).strip(),
                "open": float(item.get("open") or 0),
                "price": float(item.get("trade") or 0),
                "amount": float(item.get("amount") or 0),
                "volume": float(item.get("volume") or 0),
                "change_percent": change_percent,
            })
        return result

    def get_turnover_data(self, code, as_of):
        """复用已核验的同日缓存，第8项为百分数换手率，不从成交量估算。"""
        return self.get_daily_data(code, as_of)[["date", "turnover"]]

    @staticmethod
    def _bounded_rows(rows, as_of):
        if not isinstance(rows, list) or not rows:
            raise ValueError("缺少一致前复权日K")
        bounded = []
        for row in rows:
            if not isinstance(row, list) or not row or not isinstance(row[0], str):
                raise ValueError("腾讯日K字段不完整")
            session = date.fromisoformat(row[0])
            if session.isoformat() != row[0]:
                raise ValueError("腾讯日K日期无效")
            # Clip before checking OHLCV so a future candle cannot affect the
            # frozen target date, even if that future row is incomplete.
            if session > as_of:
                continue
            if len(row) < 6:
                raise ValueError("腾讯日K字段不完整")
            bounded.append(row)
        return bounded

    def get_daily_data(self, code, as_of):
        cache_path = self.cache_dir / f"{code}.json"
        payload = None
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (isinstance(cached, dict) and cached.get("as_of") == as_of.isoformat()
                        and cached.get("source") == "tencent_qfq_v1" and cached.get("code") == code):
                    rows = self._bounded_rows(cached.get("rows"), as_of)
                    if rows and max(row[0] for row in rows) == as_of.isoformat():
                        payload = {**cached, "rows": rows}
            except (OSError, ValueError, TypeError):
                pass
            if payload is None:
                # A matching as_of label does not prove the bars are fresh.
                # In particular, old explicit-end responses can stop at D-1.
                cache_path.unlink(missing_ok=True)
        if payload is None:
            symbol = _to_sina_tx_symbol(code)
            now = get_market_now("cn")
            same_day_postmarket = as_of == now.date() and infer_market_phase(
                "cn", current_time=now) == MarketPhase.POSTMARKET
            # Tencent's explicit-end endpoint can exclude the current daily
            # bar even when end is in the future. Only the normal same-day
            # post-close path uses its latest endpoint; frozen days stay dated.
            end = "" if same_day_postmarket else as_of.isoformat()
            response = self._get(self.history_url, params={
                "param": f"{symbol},day,,{end},240,qfq",
            })
            result = response.json()
            if result.get("code") != 0:
                raise ValueError("腾讯日K业务返回失败")
            stock = result.get("data", {}).get(symbol, {})
            # 不混入未复权 day 数组；缺前复权数组时明确判为数据缺失。
            rows = self._bounded_rows(stock.get("qfqday"), as_of)
            payload = {"as_of": as_of.isoformat(), "code": code,
                       "source": "tencent_qfq_v1", "rows": rows}
            # Leave stale data visible to the existing freshness/coverage
            # checks, but never persist it as a fresh same-day cache hit.
            if rows and max(row[0] for row in rows) == as_of.isoformat():
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                temporary.replace(cache_path)
        rows = payload.get("rows", [])
        if any(not isinstance(row, list) or len(row) < 6 for row in rows):
            raise ValueError("腾讯日K字段不完整")
        # 第6项是手数，形态使用相对量；转为股，不能用来冒充成交额。
        frame = pd.DataFrame([row[:6] for row in rows], columns=[
            "date", "open", "close", "high", "low", "volume",
        ])
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * 100
        # 第8项是百分数换手率（如3.74表示3.74%），不是成交额。
        # 老响应/缓存可以只有6列，仍供传统形态使用，但不能通过换手率核验。
        frame["turnover"] = pd.to_numeric(pd.Series(
            [row[7] if len(row) > 7 else None for row in rows]), errors="coerce")
        return frame
