"""全市场形态筛选的批量行情入口：有超时的列表与一致前复权日 K。"""
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from data_provider.akshare_fetcher import _to_sina_tx_symbol


class PatternMarketData:
    """复用既有代码市场映射；不把腾讯第六列成交量误作成交额。"""
    source_name = "腾讯前复权日K"
    universe_source = "新浪沪深京A股实时列表"
    universe_url = "https://vip.stock.finance.sina.com.cn/mkt/#hs_a"
    history_url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get(url, **kwargs):
        response = requests.get(url, timeout=(3, 8), **kwargs)
        response.raise_for_status()
        return response

    def get_universe(self) -> list[dict]:
        from akshare.utils import demjson
        base = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center."
        response = self._get(base + "getHQNodeStockCount", params={"node": "hs_a"})
        count = int(str(demjson.decode(response.text)))
        if not 1000 <= count <= 15000:
            raise ValueError("A股列表数量异常，停止筛选")

        def page(number):
            response = self._get(base + "getHQNodeData", params={
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
            result.append({
                "code": code, "name": str(item.get("name", "")).strip(),
                "open": float(item.get("open") or 0),
                "price": float(item.get("trade") or 0),
                "amount": float(item.get("amount") or 0),
                "volume": float(item.get("volume") or 0),
            })
        return result

    def get_daily_data(self, code, as_of):
        cache_path = self.cache_dir / f"{code}.json"
        payload = None
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (isinstance(cached, dict) and cached.get("as_of") == as_of.isoformat()
                        and cached.get("source") == "tencent_qfq_v1" and cached.get("code") == code):
                    payload = cached
            except (OSError, ValueError):
                pass
        if payload is None:
            symbol = _to_sina_tx_symbol(code)
            response = self._get(self.history_url, params={
                "param": f"{symbol},day,,{as_of.isoformat()},240,qfq",
            })
            result = response.json()
            if result.get("code") != 0:
                raise ValueError("腾讯日K业务返回失败")
            stock = result.get("data", {}).get(symbol, {})
            # 不混入未复权 day 数组；缺前复权数组时明确判为数据缺失。
            rows = stock.get("qfqday")
            if not isinstance(rows, list) or not rows:
                raise ValueError("缺少一致前复权日K")
            payload = {"as_of": as_of.isoformat(), "code": code,
                       "source": "tencent_qfq_v1", "rows": rows}
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
        return frame
