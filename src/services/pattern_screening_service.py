"""八形态筛选编排，保留完整覆盖统计并生成独立可审阅报告。"""
import html
import json
import logging
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path

from data_provider.base import is_st_stock
from src.pattern_screener import (
    PATTERN_NAMES, ScreeningDataError, detect_patterns, prepare_bars, volume_ratio,
)

logger = logging.getLogger(__name__)


def screen_market(provider, as_of, top_n=10, workers=4, min_amount=20_000_000,
                  deadline_seconds=1200, min_coverage=.9):
    universe = provider.get_universe()
    excluded = Counter()
    eligible = []
    for stock in universe:
        if not all(math.isfinite(stock[key]) for key in ("open", "price", "amount", "volume")):
            excluded["无效报价"] += 1
        elif not stock["name"] or is_st_stock(stock["name"]):
            excluded["ST或名称缺失"] += 1
        elif stock["volume"] <= 0 or stock["price"] <= 0 or stock["open"] <= 0:
            excluded["停牌或无有效报价"] += 1
        elif stock["amount"] < min_amount:
            excluded["成交额不足"] += 1
        elif stock["price"] <= stock["open"]:
            excluded["当天不是阳线"] += 1
        else:
            eligible.append(stock)

    candidates = []
    failures = Counter()
    failure_examples = []
    completed = 0
    verified = 0
    started = time.monotonic()

    def inspect(stock):
        frame = prepare_bars(provider.get_daily_data(stock["code"], as_of), as_of)
        # 最新前复权价格应与当天未复权收盘一致，防止响应错码或复权基准异常。
        if abs(float(frame.close.iloc[-1]) / stock["price"] - 1) > .01:
            raise ScreeningDataError("quote_mismatch")
        signals = detect_patterns(frame, as_of)
        if not signals:
            return None
        return {
            "code": stock["code"], "name": stock["name"],
            "data_date": as_of.isoformat(), "close": round(float(frame.close.iloc[-1]), 3),
            "amount": stock["amount"], "volume_ratio": round(volume_ratio(frame), 2),
            "signals": [signal.to_dict() for signal in signals],
            "source": provider.source_name,
            "source_url": "https://gu.qq.com/" + (
                "bj" if stock["code"].startswith(("43", "83", "87", "88", "92"))
                else "sh" if stock["code"].startswith("6") else "sz"
            ) + stock["code"],
            "closes": [round(float(value), 3) for value in frame.close.tail(60)],
        }

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(inspect, stock): stock for stock in eligible}
    try:
        for future in as_completed(futures, timeout=deadline_seconds):
            stock = futures[future]
            completed += 1
            try:
                candidate = future.result()
                verified += 1
                if candidate:
                    candidates.append(candidate)
            except ScreeningDataError as exc:
                failures[exc.reason] += 1
                if exc.reason == "short_history":
                    # 最新日期已检查通过：不足120根是明确排除项，而不是网络覆盖缺口。
                    verified += 1
            except Exception as exc:
                failures["fetch_error"] += 1
                if len(failure_examples) < 10:
                    failure_examples.append({"code": stock["code"], "error": type(exc).__name__})
            if completed % 100 == 0:
                logger.info("筛选进度 %s/%s，命中 %s，数据异常 %s",
                            completed, len(eligible), len(candidates), sum(failures.values()))
    except TimeoutError:
        failures["budget_unprocessed"] = len(eligible) - completed
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    # 仅是研究优先顺序，不是上涨概率：多形态交集→相对放量→代码稳定排序。
    candidates.sort(key=lambda item: (-len(item["signals"]), -item["volume_ratio"], item["code"]))
    coverage = verified / len(eligible) if eligible else 1.0
    complete = completed == len(eligible) and coverage >= min_coverage
    return {
        "status": "complete" if complete else "incomplete", "as_of": as_of.isoformat(),
        "universe_count": len(universe), "eligible_count": len(eligible),
        "completed_count": completed, "verified_count": verified, "coverage": round(coverage, 4),
        "excluded": dict(excluded), "data_issues": dict(failures), "failure_examples": failure_examples,
        "candidates": candidates, "selected": candidates[:top_n] if complete else [],
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "universe_source": provider.universe_source, "universe_url": provider.universe_url,
        "history_source": provider.source_name, "history_url": provider.history_url,
        "parameters": {"top_n": top_n, "workers": workers, "min_amount": min_amount,
                       "deadline_seconds": deadline_seconds, "min_coverage": min_coverage},
    }


def _sparkline(values):
    bottom, top = min(values), max(values)
    points = " ".join(f"{5 + index * 290 / max(len(values)-1, 1):.1f},"
                      f"{65 - (value-bottom)*55/max(top-bottom, .001):.1f}"
                      for index, value in enumerate(values))
    return f'<svg viewBox="0 0 300 75" role="img" aria-label="最近60日收盘价"><polyline points="{points}" fill="none" stroke="#318777" stroke-width="2"/></svg>'


def render_html(result):
    escape = html.escape
    cards = []
    selected_codes = {item["code"] for item in result.get("selected", [])}
    for candidate in result.get("candidates", []):
        names = [signal["name"] for signal in candidate["signals"]]
        evidence = "".join(
            f'<li><strong>{escape(signal["name"])}</strong>：' +
            escape("；".join(f"{key}：{value}" for key, value in signal["evidence"].items())) + "</li>"
            for signal in candidate["signals"]
        )
        badge = '<span class="selected">本次分析</span>' if candidate["code"] in selected_codes else ''
        cards.append(f'''<article data-patterns="{escape('|'.join(names), quote=True)}">
<div class="card-top"><h3>{escape(candidate['name'])} <small>{escape(candidate['code'])}</small></h3>
<a target="_blank" rel="noopener" href="{escape(candidate['source_url'], quote=True)}">查看行情 ↗</a></div>
<p class="tags">{escape(' · '.join(names))} {badge}</p>{_sparkline(candidate['closes'])}
<p>行情 {escape(candidate['data_date'])} · 收盘 {candidate['close']:.3f} 元 · 量比 {candidate['volume_ratio']:.2f} · 成交额 {candidate['amount']/1e8:.2f} 亿元</p>
<ul>{evidence}</ul><p class="muted">待人工查看K线确认；随后核查公告、估值与交易可行性。</p></article>''')
    counts = Counter(signal["name"] for candidate in result.get("candidates", []) for signal in candidate["signals"])
    buttons = '<button class="active" data-filter="">全部</button>' + "".join(
        f'<button data-filter="{name}">{name} <b>{counts[name]}</b></button>' for name in PATTERN_NAMES
    )
    issue_labels = {
        "short_history": "历史不足120日", "fetch_error": "抓取失败或缺少前复权数据",
        "stale": "行情日期滞后", "quote_mismatch": "报价与历史末价不一致",
        "invalid_ohlcv": "量价数据无效", "missing_columns": "行情字段缺失",
        "duplicate_dates": "同日数据重复", "invalid_dates": "行情日期无效",
        "budget_unprocessed": "筛选预算内未完成", "ConnectionError": "市场列表连接失败",
    }
    issues = escape(" · ".join(
        f"{issue_labels.get(key, '其他数据异常')} {count}只"
        for key, count in result.get("data_issues", {}).items()
    ) or "无")
    exclusions = escape(" · ".join(f"{key} {value}" for key, value in result.get("excluded", {}).items()))
    complete = result.get("status") == "complete"
    status = "筛选已完成" if complete else "覆盖不完整 · 不进入自动分析"
    if result.get("status") == "market_closed":
        status = "A股休市 · 本次正常跳过"
    minimum_amount = result.get("parameters", {}).get("min_amount", 20_000_000) / 10_000
    no_candidates = '<p class="empty">本次没有符合条件的形态候选。</p>' if complete else '<p class="empty">本次数据尚不足以完成筛选，请查看数据异常统计。</p>'
    if result.get("status") == "market_closed":
        no_candidates = '<p class="empty">当前A股休市，未执行市场筛选，不生成今天的新候选。</p>'
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>八种技术形态候选 · {escape(result['as_of'])}</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f5f3;color:#20322f;font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1200px;margin:auto;padding:44px 28px}}h1{{font-size:34px;line-height:1.3;margin:10px 0}}h2{{font-size:22px}}.eyebrow{{color:#56746c;font-size:13px;letter-spacing:2px}}.muted,small{{color:#63756e}}.status{{background:{'#deeee8' if complete else '#f4e6cc'};padding:14px 18px;border-radius:12px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.stat{{background:#fff;padding:18px;border-radius:12px}}.stat b{{display:block;font-size:30px}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}}button{{border:1px solid #ccd8d1;background:#fff;padding:9px 14px;border-radius:20px;color:#24483e;cursor:pointer}}button.active{{background:#244f43;color:#fff}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}}article{{background:#fff;border:1px solid #dde4df;border-radius:14px;padding:22px}}h3{{margin:0;font-size:21px}}.card-top{{display:flex;justify-content:space-between;gap:12px}}a{{color:#217261}}.tags{{color:#277d68;font-weight:600}}.selected{{display:inline-block;background:#dceee7;font-size:12px;padding:2px 8px;border-radius:8px;margin-left:8px}}svg{{width:100%;height:110px}}article p,li{{font-size:14px}}ul{{padding-left:20px}}.note{{background:#e8ece8;padding:22px;border-radius:12px;margin-top:28px}}.empty{{padding:30px;background:white;border-radius:12px}}[hidden]{{display:none}}@media(max-width:700px){{main{{padding:24px 16px}}h1{{font-size:27px}}.grid{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,1fr)}}.card-top{{display:block}}}}</style>
<main><div class="eyebrow">DAILY STOCK ANALYSIS / 技术形态筛选</div><h1>八种技术形态候选</h1>
<p class="muted">行情截止 {escape(result['as_of'])} · 沪深京 A 股 · 原图形态的量化近似，筛选命中不代表后续必然上涨。</p>
<div class="status">{status} · 当日行情日期核验覆盖 {result.get('coverage',0):.1%}</div>
<div class="stats"><div class="stat">市场股票<b>{result.get('universe_count',0):,}</b></div><div class="stat">基础条件通过<b>{result.get('eligible_count',0):,}</b></div><div class="stat">形态候选<b>{len(result.get('candidates',[]))}</b></div><div class="stat">交给现有分析<b>{len(result.get('selected',[]))}</b></div></div>
<p class="muted">{exclusions}</p><h2>按形态查看</h2><nav>{buttons}</nav>
<div class="grid">{''.join(cards) or no_candidates}</div>
<section class="note"><h2>筛选依据与数据限制</h2><p>同一股票命中多形态时合并。研究顺序按命中数、量比排列，分数不是上涨概率。均量、阻力与形态拟合均不包含当日突破；只接受最新完整交易日的日K。前复权窗口每天重新获取，同日缓存仅供重跑；不混入未复权实时价格。</p>
<p>本次排除ST、停牌、非阳线、当日成交额不足{minimum_amount:g}万元及120根历史不足的股票。杯柄和圆弧底尤其需要人工看图复核。覆盖不足或查询未完成时不会自动回退到原来的自选股。</p>
<p>数据异常或排除：{issues}。本次日K核验耗时：{result.get('elapsed_seconds',0)}秒。</p>
<p>股票范围与成交额：<a href="{escape(result.get('universe_url',''),quote=True)}">{escape(result.get('universe_source',''))}</a>；日K：<a href="{escape(result.get('history_url',''),quote=True)}">{escape(result.get('history_source',''))}</a>。每个候选均附独立行情链接。</p></section></main>
<script>document.querySelectorAll('button[data-filter]').forEach(button=>button.onclick=()=>{{document.querySelectorAll('button[data-filter]').forEach(x=>x.classList.toggle('active',x===button));document.querySelectorAll('article').forEach(card=>card.hidden=!!button.dataset.filter&&!card.dataset.patterns.split('|').includes(button.dataset.filter));}});</script></html>'''


def write_result(result, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "screen.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "report.html").write_text(render_html(result), encoding="utf-8")
    (directory / "stocks.txt").write_text(",".join(stock["code"] for stock in result["selected"]), encoding="utf-8")
