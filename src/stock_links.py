"""Fixed A-share quote links for report stock names."""
from __future__ import annotations

import re
from typing import Optional

from data_provider.base import normalize_stock_code


def get_a_share_quote_url(code: str) -> Optional[str]:
    """Return a known A-share quote URL without discarding exchange identity."""
    raw_code = str(code or "").strip()
    explicit_market = re.fullmatch(r"(SH|SZ|SS|BJ)\.?[0-9]{6}", raw_code, re.IGNORECASE)
    if explicit_market is None:
        explicit_market = re.fullmatch(r"[0-9]{6}\.(SH|SZ|SS|BJ)", raw_code, re.IGNORECASE)
    declared_exchange = explicit_market.group(1).lower() if explicit_market else None
    if declared_exchange == "ss":
        declared_exchange = "sh"
    normalized = normalize_stock_code(raw_code)
    if not re.fullmatch(r"(?:00|30|60|68|43|83|87|88|92)[0-9]{4}", normalized):
        return None
    exchange = "sh" if normalized.startswith(("60", "68")) else (
        "sz" if normalized.startswith(("00", "30")) else "bj"
    )
    # SH000001 is the Shanghai index, not the SZ000001 A-share with the same digits.
    if declared_exchange is not None and declared_exchange != exchange:
        return None
    return f"https://gu.qq.com/{exchange}{normalized}"


def format_stock_name_markdown(name: str, code: str, *, escape_underscores: bool = False) -> str:
    """Link only a report label; raw names remain suitable for subjects and data.

    Escaping backslashes and brackets prevents a supplied stock name from
    closing the label or injecting a different Markdown link. Legacy report
    renderers differ on underscores, so callers retain that existing choice.
    """
    label = str(name or "").replace("\r", " ").replace("\n", " ")
    label = label.replace("\\", "\\\\").replace("*", "\\*").replace("[", "\\[").replace("]", "\\]")
    label = label.replace("<", "\\<").replace(">", "\\>")
    if escape_underscores:
        label = label.replace("_", "\\_")
    url = get_a_share_quote_url(code)
    return f"[{label}]({url})" if url and label else label
