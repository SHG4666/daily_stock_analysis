#!/usr/bin/env python3
"""收盘后按底部反弹或原图形态模式筛选；独立运行不调用AI或发送通知。"""
import argparse
import logging
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.pattern_market_data import PatternMarketData
from data_provider.shareholder_count import ShareholderCountProvider
from src.core.trading_calendar import get_effective_trading_date, get_market_now, is_market_open
from src.core.screening_session import is_delayed_schedule
from src.pattern_screener import RULESET_VERSION
from src.services.pattern_screening_service import enrich_shareholder_counts, screen_market, write_result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/pattern_screen")
    parser.add_argument("--cache-dir", default="data/pattern_screen_cache")
    parser.add_argument("--top", type=int, default=int(os.getenv("PATTERN_SCREEN_TOP_N", "10")))
    parser.add_argument("--workers", type=int, default=int(os.getenv("PATTERN_SCREEN_WORKERS", "4")))
    parser.add_argument("--min-amount", type=float, default=float(os.getenv("PATTERN_SCREEN_MIN_AMOUNT", "20000000")))
    parser.add_argument("--deadline", type=int, default=int(os.getenv("PATTERN_SCREEN_DEADLINE_SECONDS", "1200")))
    parser.add_argument("--min-coverage", type=float, default=float(os.getenv("PATTERN_SCREEN_MIN_COVERAGE", "0.9")))
    parser.add_argument("--profile", choices=("all_patterns", "bottom_reversal"),
                        default=os.getenv("PATTERN_SCREEN_PROFILE", "all_patterns"))
    args = parser.parse_args(argv)
    if args.profile not in ("all_patterns", "bottom_reversal"):
        parser.error("未知筛选模式")
    if (not 1 <= args.top <= 50 or not 1 <= args.workers <= 8
            or not math.isfinite(args.min_amount) or args.min_amount < 0
            or args.deadline < 1 or not 0 < args.min_coverage <= 1):
        parser.error("top需1-50，workers需1-8，成交额非负，时限为正，覆盖阈值在(0,1]")
    return args


def write_github_outputs(result):
    if os.getenv("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write("stock_codes=" + ",".join(stock["code"] for stock in result["selected"]) + "\n")
            stream.write("has_candidates=" + str(bool(result["selected"])).lower() + "\n")
            stream.write("status=" + result["status"] + "\n")
            stream.write("as_of=" + result["as_of"] + "\n")
            stream.write("delayed_schedule=" + str(bool(result.get("delayed_schedule"))).lower() + "\n")


def empty_result(status, as_of, issue=None, profile="all_patterns"):
    from src import bottom_reversal
    return {
        "status": status, "as_of": as_of.isoformat(), "selected": [], "candidates": [],
        "profile": profile,
        "ruleset_version": bottom_reversal.RULESET_VERSION if profile == bottom_reversal.PROFILE else RULESET_VERSION,
        "data_issues": {issue: 1} if issue else {}, "coverage": 0,
        "universe_source": PatternMarketData.universe_source,
        "universe_url": PatternMarketData.universe_url,
        "history_source": PatternMarketData.source_name, "history_url": PatternMarketData.history_url,
    }


def main(argv=None):
    from src.config import setup_env
    setup_env()
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    now = get_market_now("cn")
    as_of = get_effective_trading_date("cn", current_time=now)
    event_name = os.getenv("GITHUB_EVENT_NAME", "")
    delayed = is_delayed_schedule(now, as_of, event_name)
    if not delayed and not is_market_open("cn", now.date()):
        result = empty_result("market_closed", as_of, profile=args.profile)
        write_result(result, args.output_dir)
        write_github_outputs(result)
        logging.info("A股休市，正常跳过，不生成今天的新候选")
        return 0
    if not delayed and (as_of != now.date() or (event_name == "schedule" and now.hour < 9)):
        logging.error("仅在交易日收盘后筛选；最近完整行情日为 %s，本次不生成当日候选", as_of)
        return 2
    try:
        provider = PatternMarketData(Path(args.cache_dir))
        result = screen_market(provider, as_of, top_n=args.top, workers=args.workers,
                               min_amount=args.min_amount, deadline_seconds=args.deadline,
                               min_coverage=args.min_coverage, profile=args.profile)
        result["delayed_schedule"] = delayed
        result["generated_at"] = now.isoformat()
        enrich_shareholder_counts(result, ShareholderCountProvider(), workers=args.workers)
        write_result(result, args.output_dir)
        write_github_outputs(result)
        logging.info("筛选结束：状态=%s 全市场=%s 基础通过=%s 候选=%s 选入分析=%s",
                     result["status"], result["universe_count"], result["eligible_count"],
                     len(result["candidates"]), len(result["selected"]))
        return 0 if result["status"] == "complete" else 2
    except Exception as exc:
        result = empty_result("incomplete", as_of, type(exc).__name__, profile=args.profile)
        write_result(result, args.output_dir)
        write_github_outputs(result)
        logging.error("筛选无法完成：%s", type(exc).__name__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
