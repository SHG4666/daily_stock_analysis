"""Pure formatting of validated disclosed shareholder-account counts."""
from __future__ import annotations

import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.report_language import normalize_report_language
from src.stock_links import get_a_share_quote_url


def _today():
    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def _iso_date(value):
    if not isinstance(value, str):
        raise ValueError("date must be an ISO date string")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("date must use YYYY-MM-DD")
    return parsed


def format_shareholder_count(payload, language="zh", *, expected_code=None) -> str:
    """Format a disclosed count, hiding missing, future or mismatched data.

    This function performs no fetching. The optional expected code lets report
    renderers reject a valid record belonging to a different security.
    """
    if not isinstance(payload, dict) or payload.get("source") != "东方财富":
        return ""
    quote_url = get_a_share_quote_url(payload.get("code"))
    if not quote_url or (expected_code is not None and quote_url != get_a_share_quote_url(expected_code)):
        return ""
    count = payload.get("count")
    try:
        if (isinstance(count, bool) or not isinstance(count, (int, float))
                or not math.isfinite(count) or count <= 0 or int(count) != count):
            return ""
        end_date = _iso_date(payload.get("end_date"))
        announcement_date = _iso_date(payload.get("announcement_date"))
        today = _today()
        if not end_date <= announcement_date <= today:
            return ""
        fetched_at = payload.get("fetched_at")
        if fetched_at is not None:
            if not isinstance(fetched_at, str) or "T" not in fetched_at:
                return ""
            fetched_time = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            fetched_date = (
                fetched_time.astimezone(ZoneInfo("Asia/Shanghai")).date()
                if fetched_time.tzinfo else fetched_time.date()
            )
            if not announcement_date <= fetched_date <= today:
                return ""
        formatted_count = f"{int(count):,}"
    except (TypeError, ValueError, OverflowError):
        return ""
    if normalize_report_language(language) == "en":
        return (
            f"Latest disclosed shareholder accounts: {formatted_count} "
            f"(as of {end_date.isoformat()}, announced {announcement_date.isoformat()}; source: Eastmoney)"
        )
    return (
        f"最新披露股东户数：{formatted_count} 户"
        f"（截至 {end_date.isoformat()}，公告 {announcement_date.isoformat()}；来源：东方财富）"
    )
