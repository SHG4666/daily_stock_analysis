#!/usr/bin/env python3
"""从筛选报告生成飞书原生走势图；仅显式--send时推送。"""
import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.trading_calendar import get_effective_trading_date
from src.services.pattern_chart_cards import deliver_chart_cards


def main(argv=None):
    from src.config import setup_env
    from src.notification import NotificationService
    setup_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="reports/pattern_screen/screen.json")
    parser.add_argument("--output-dir", default="reports/pattern_screen/feishu_charts")
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        result = json.loads(Path(args.report).read_text(encoding="utf-8"))
        if args.send and date.fromisoformat(result["as_of"]) != get_effective_trading_date("cn"):
            raise ValueError("图表已过期，不推送旧报告")
        attempts = deliver_chart_cards(result, NotificationService(), args.output_dir, send=args.send)
        return 1 if any(item["status"] == "failed" for item in attempts) else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logging.error("图表任务无法完成：%s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
