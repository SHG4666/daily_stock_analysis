"""Optional, dated A-share shareholder counts from East Money disclosures."""
from __future__ import annotations

import json
import logging
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from data_provider.base import normalize_stock_code
from src.stock_links import get_a_share_quote_url

logger = logging.getLogger(__name__)
_API_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_CACHE_TTL = timedelta(hours=1)
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _disclosure_date(value) -> Optional[date]:
    if not isinstance(value, str):
        return None
    try:
        # The API returns YYYY-MM-DD 00:00:00; cached payloads use ISO dates.
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _validated_payload(payload, code: str, as_of: date, now: datetime) -> Optional[dict]:
    """Validate fetched and cached values with the same identity/date contract."""
    if not isinstance(payload, dict) or payload.get("code") != code:
        return None
    count = payload.get("count")
    if type(count) is not int or count <= 0:
        return None
    end = _disclosure_date(payload.get("end_date"))
    announcement = _disclosure_date(payload.get("announcement_date"))
    if end is None or announcement is None or not end <= announcement <= as_of:
        return None
    source_url = f"https://data.eastmoney.com/gdhs/detail/{code}.html"
    if payload.get("source") != "东方财富" or payload.get("source_url") != source_url:
        return None
    fetched_at = payload.get("fetched_at")
    if not isinstance(fetched_at, str):
        return None
    try:
        fetched = datetime.fromisoformat(fetched_at)
        if fetched.tzinfo is None or not timedelta(0) <= now - fetched < _CACHE_TTL:
            return None
        if fetched.astimezone(_SHANGHAI).date() < announcement:
            return None
    except (ValueError, TypeError, OverflowError):
        return None
    return {
        "code": code, "count": count,
        "end_date": end.isoformat(), "announcement_date": announcement.isoformat(),
        "source": "东方财富", "source_url": source_url, "fetched_at": fetched_at,
    }


class ShareholderCountProvider:
    """Fetch one latest available disclosure; failures never become cached data."""

    def __init__(self, cache_dir: Path = Path("data/shareholder_count_cache")):
        self.cache_dir = Path(cache_dir)

    def _read_cache(self, path: Path, code: str, as_of: date, now: datetime) -> Optional[dict]:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.warning("股东户数 %s 缓存读取失败（%s）", code, type(exc).__name__)
            return None
        if isinstance(cached, dict) and cached.get("as_of") == as_of.isoformat():
            payload = _validated_payload(cached.get("data"), code, as_of, now)
            if payload is not None:
                return payload
        logger.info("股东户数 %s 缓存过期或字段无效，重新查询", code)
        return None

    def _write_cache(self, path: Path, payload: dict, as_of: date) -> None:
        temporary = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Each thread/process owns its temporary file; readers see a whole JSON.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.cache_dir,
                prefix=path.stem + ".", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump({"as_of": as_of.isoformat(), "data": payload}, handle,
                          ensure_ascii=False, allow_nan=False)
            temporary.replace(path)
        except (OSError, ValueError) as exc:
            logger.warning("股东户数 %s 缓存保存失败（%s）", payload["code"], type(exc).__name__)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    def get_latest(self, code: str, as_of: date) -> Optional[dict]:
        """Return latest disclosed count in 户, never an estimate of live holders.

        ``as_of`` is the disclosure availability date, which can differ from the
        daily candle's date. A historical replay must pass its own cutoff date.
        """
        now = _utc_now()
        if (not isinstance(code, str) or not get_a_share_quote_url(code)
                or type(as_of) is not date or as_of > now.astimezone(_SHANGHAI).date()):
            logger.warning("股东户数请求的A股代码或截止日期无效")
            return None
        code = normalize_stock_code(code.strip())
        path = self.cache_dir / f"{code}_{as_of.isoformat()}.json"
        cached = self._read_cache(path, code, as_of, now)
        if cached is not None:
            return cached
        cutoff = as_of.isoformat()
        params = {
            "reportName": "RPT_HOLDERNUM_DET",
            "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,HOLDER_NUM,END_DATE,HOLD_NOTICE_DATE",
            "filter": f'(SECURITY_CODE="{code}")(END_DATE<=\'{cutoff}\')(HOLD_NOTICE_DATE<=\'{cutoff}\')',
            "sortColumns": "END_DATE,HOLD_NOTICE_DATE", "sortTypes": "-1,-1",
            "pageSize": "1", "pageNumber": "1", "source": "WEB", "client": "WEB",
        }
        try:
            response = requests.get(_API_URL, params=params, timeout=(3, 5))
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("股东户数 %s 获取失败（%s）", code, type(exc).__name__)
            return None
        if (not isinstance(result, dict) or result.get("success") is not True
                or type(result.get("code")) is not int or result["code"] != 0):
            logger.warning("股东户数 %s 接口业务返回无效", code)
            return None
        data = result.get("result")
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows:
            logger.info("股东户数 %s 截至%s无可用披露", code, cutoff)
            return None
        row = rows[0]
        if not isinstance(row, dict):
            logger.warning("股东户数 %s 披露记录结构无效", code)
            return None
        payload = _validated_payload({
            "code": row.get("SECURITY_CODE"), "count": row.get("HOLDER_NUM"),
            "end_date": row.get("END_DATE"), "announcement_date": row.get("HOLD_NOTICE_DATE"),
            "source": "东方财富", "source_url": f"https://data.eastmoney.com/gdhs/detail/{code}.html",
            "fetched_at": now.isoformat(),
        }, code, as_of, now)
        if payload is None:
            logger.warning("股东户数 %s 披露的代码、户数或日期无效", code)
            return None
        self._write_cache(path, payload, as_of)
        return payload
