"""原图形态与扩展规则筛选，保留覆盖统计并生成独立可审阅报告。"""
import html
import json
import logging
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import date
from pathlib import Path

from data_provider.base import is_st_stock
from src.pattern_screener import (
    BASE_PATTERN_NAMES, PATTERN_NAMES, RULESET_VERSION, VOLUME_BREAKOUT_LIMITS,
    ScreeningDataError, detect_patterns, prepare_bars, volume_ratio,
)
from src.stock_links import get_a_share_quote_url
from src.shareholder_count import format_shareholder_count
from src import bottom_reversal

logger = logging.getLogger(__name__)


def rank_candidates(candidates):
    """扩展规则提供候选资格，和原图形态重叠时不重复增加研究计数。"""
    for candidate in candidates:
        base_matches = {signal["name"] for signal in candidate["signals"]
                        if signal["name"] in BASE_PATTERN_NAMES}
        candidate["priority_count"] = max(1, len(base_matches))
    candidates.sort(key=lambda item: (-item["priority_count"], -item["volume_ratio"], item["code"]))
    return candidates


def build_chart_bars(frame):
    """保留120根真实日K；均线先在完整核验窗口计算，再截取展示区间。"""
    display = frame.copy()
    for window in (5, 10, 20, 30):
        display[f"ma{window}"] = display.close.rolling(window).mean()
    rows = []
    for _, row in display.tail(120).iterrows():
        item = {"date": row["date"].date().isoformat()}
        for key in ("open", "high", "low", "close", "volume", "ma5", "ma10", "ma20", "ma30"):
            value = float(row[key])
            item[key] = round(value, 4) if math.isfinite(value) else None
        rows.append(item)
    return rows


def screen_market(provider, as_of, top_n=10, workers=4, min_amount=20_000_000,
                  deadline_seconds=1200, min_coverage=.9, profile="all_patterns", learning_context=None):
    if profile not in ("all_patterns", bottom_reversal.PROFILE):
        raise ValueError("unknown screening profile")
    bottom_focus = profile == bottom_reversal.PROFILE
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
        elif (bottom_focus and stock.get("change_percent") is not None
              and math.isfinite(stock["change_percent"])
              and stock["change_percent"] < bottom_reversal.LIMITS["daily_return_min"] * 100):
            # 仅提前执行同一涨幅门槛；通过后仍以完整日K再次核实，减少无效下载。
            excluded["报价涨幅不足2%"] += 1
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
        metrics = {}
        if bottom_focus:
            metrics, rejection = bottom_reversal.inspect_bottom_reversal(frame)
            if rejection:
                return {"rejection": rejection}
            try:
                turnover, passed = bottom_reversal.confirm_turnover(
                    frame, provider.get_turnover_data(stock["code"], as_of))
            except Exception as exc:
                raise ScreeningDataError("turnover_data") from exc
            if not passed:
                return {"rejection": "换手率未明显提高"}
            metrics.update(turnover)
            signals = [bottom_reversal.reversal_signal(metrics)]
        else:
            signals = detect_patterns(frame, as_of)
        if not signals:
            return None
        return {
            "code": stock["code"], "name": stock["name"],
            "data_date": as_of.isoformat(), "close": round(float(frame.close.iloc[-1]), 3),
            "amount": stock["amount"], "volume_ratio": round(volume_ratio(frame), 2),
            "signals": [signal.to_dict() for signal in signals],
            "source": provider.source_name,
            "source_url": get_a_share_quote_url(stock["code"]),
            "closes": [round(float(value), 3) for value in frame.close.tail(60)],
            "chart_bars": build_chart_bars(frame),
            **({"bottom_reversal": metrics, "turnover_source": provider.turnover_source,
                "turnover_url": provider.turnover_url} if bottom_focus else {}),
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
                if candidate and "rejection" in candidate:
                    excluded[candidate["rejection"]] += 1
                elif candidate:
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

    if bottom_focus:
        # 当前规则要求当日触发；先比较真实换手率提升，再比较放量。
        candidates.sort(key=lambda item: (-item["bottom_reversal"]["turnover_five_mean_ratio"],
                                          -item["volume_ratio"], item["code"]))
    else:
        rank_candidates(candidates)
    learning = {"applied": False, "rule": "baseline", "baseline_pool": [item["code"] for item in candidates]}
    if bottom_focus:
        from src.services.screen_learning_service import apply_learned_ranking
        learning = apply_learned_ranking(candidates, learning_context, as_of)
    turnover_required = len(candidates) + excluded["换手率未明显提高"] + failures["turnover_data"]
    turnover_coverage = (1 - failures["turnover_data"] / turnover_required) if turnover_required else 1.0
    coverage = verified / len(eligible) if eligible else 1.0
    complete = (completed == len(eligible) and coverage >= min_coverage
                and (not bottom_focus or turnover_coverage >= min_coverage))
    return {
        "status": "complete" if complete else "incomplete", "as_of": as_of.isoformat(),
        "ruleset_version": bottom_reversal.RULESET_VERSION if bottom_focus else RULESET_VERSION,
        "profile": profile,
        "learning": learning,
        "turnover_required": turnover_required if bottom_focus else 0,
        "turnover_coverage": round(turnover_coverage, 4) if bottom_focus else None,
        "universe_count": len(universe), "eligible_count": len(eligible),
        "completed_count": completed, "verified_count": verified, "coverage": round(coverage, 4),
        "excluded": dict(excluded), "data_issues": dict(failures), "failure_examples": failure_examples,
        "candidates": candidates, "selected": candidates[:top_n] if complete else [],
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "universe_source": provider.universe_source, "universe_url": provider.universe_url,
        "history_source": provider.source_name, "history_url": provider.history_url,
        "parameters": {"top_n": top_n, "workers": workers, "min_amount": min_amount,
                       "deadline_seconds": deadline_seconds, "min_coverage": min_coverage,
                       "volume_breakout": dict(VOLUME_BREAKOUT_LIMITS),
                       "bottom_reversal": dict(bottom_reversal.LIMITS) if bottom_focus else None,
                       "ranking": ("turnover_five_mean_ratio desc, volume_ratio desc, code asc" if bottom_focus
                                   else "max(1, base_pattern_count), volume_ratio desc, code asc")},
    }


def enrich_shareholder_counts(result, provider, workers=4, deadline_seconds=30):
    """技术筛选完成后补充披露数据，失败或超时不改变候选、排序和覆盖结论。"""
    if result.get("status") != "complete" or not result.get("candidates"):
        return
    as_of = date.fromisoformat(result["as_of"])
    pool = ThreadPoolExecutor(max_workers=min(4, max(1, workers)))
    futures = {
        pool.submit(provider.get_latest, candidate["code"], as_of): candidate
        for candidate in result["candidates"]
    }
    try:
        for future in as_completed(futures, timeout=deadline_seconds):
            candidate = futures[future]
            try:
                payload = future.result()
                if payload:
                    candidate["shareholder_count"] = payload
            except Exception as exc:
                logger.warning("%s 股东户数补充失败：%s", candidate["code"], type(exc).__name__)
    except TimeoutError:
        logger.info("股东户数补充预算用尽，继续输出已完成的候选报告")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


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
        quote_url = get_a_share_quote_url(candidate['code'])
        stock_label = f"{escape(candidate['name'])} <small>{escape(candidate['code'])}</small>"
        if quote_url:
            stock_label = (f'<a target="_blank" rel="noopener noreferrer" '
                           f'href="{escape(quote_url, quote=True)}">{stock_label} ↗</a>')
        holder_text = format_shareholder_count(
            candidate.get("shareholder_count"), expected_code=candidate["code"],
        )
        holder_line = f'<p class="holders">{escape(holder_text)}</p>' if holder_text else ''
        cards.append(f'''<article data-patterns="{escape('|'.join(names), quote=True)}">
<div class="card-top"><h3>{stock_label}</h3></div>
<p class="tags">{escape(' · '.join(names))} {badge}</p>
<p>行情 {escape(candidate['data_date'])} · 收盘 {candidate['close']:.3f} 元 · 量比 {candidate['volume_ratio']:.2f} · 成交额 {candidate['amount']/1e8:.2f} 亿元</p>
{holder_line}
<ul>{evidence}</ul><p class="muted">待人工查看K线确认；随后核查公告、估值与交易可行性。</p></article>''')
    counts = Counter(signal["name"] for candidate in result.get("candidates", []) for signal in candidate["signals"])
    bottom_focus = result.get("profile") == bottom_reversal.PROFILE
    pattern_names = bottom_reversal.SIGNAL_NAMES if bottom_focus else PATTERN_NAMES
    buttons = '<button class="active" data-filter="">全部</button>' + "".join(
        f'<button data-filter="{name}">{name} <b>{counts[name]}</b></button>' for name in pattern_names
    )
    issue_labels = {
        "short_history": "历史不足120日", "fetch_error": "抓取失败或缺少前复权数据",
        "stale": "行情日期滞后", "quote_mismatch": "报价与历史末价不一致",
        "invalid_ohlcv": "量价数据无效", "missing_columns": "行情字段缺失",
        "duplicate_dates": "同日数据重复", "invalid_dates": "行情日期无效",
        "budget_unprocessed": "筛选预算内未完成", "ConnectionError": "市场列表连接失败",
        "turnover_data": "实际换手率缺失或日期不完整",
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
    ruleset_version = escape(str(result.get("ruleset_version") or "历史报告未标注"))
    universe_scope = escape(str(result.get("universe_source") or ""))
    scope_line = f'<p class="muted">范围：{universe_scope}</p>' if universe_scope else ''
    delay_line = ''
    if result.get("delayed_schedule"):
        generated_at = escape(str(result.get("generated_at") or ""))
        delay_line = (f'<p class="muted">定时延迟运行 · 使用上一完整交易日行情；生成时间 {generated_at}。'
                      '后续分析沿用同一行情日，资讯按实际生成时刻查询。</p>')
    no_candidates = '<p class="empty">本次没有符合条件的形态候选。</p>' if complete else '<p class="empty">本次数据尚不足以完成筛选，请查看数据异常统计。</p>'
    if result.get("status") == "market_closed":
        no_candidates = '<p class="empty">当前A股休市，未执行市场筛选，不生成今天的新候选。</p>'
    focus_note = ''
    if bottom_focus:
        focus_note = ('<p><strong>当前每日策略：长期下跌后底部放量上涨。</strong>'
                      '此前60日跌幅至少15%，20日均线较40个交易日前至少下降8%；'
                      '涨幅至少2%且收阳线，成交量至少为此前20日均量的1.8倍，收盘处于日内振幅上方35%。'
                      '原底部反弹要求距60日低点不超过20%、此前5日涨幅不超过10%。'
                      '新增探底回升分支：60日低点在此前2–10个交易日内，当日不再破底；'
                      '近5日MA5新金叉MA10、两线向上，收盘站上MA5/10/20并突破此前5日最高价；'
                      '仅此分支允许距低点不超过30%、此前5日涨幅不超过15%。'
                      '实际换手率至少为上一日的1.2倍、此前5日均值的1.5倍。'
                      '按换手率提升倍数、量比排序；原图形态不再作为每日入选的替代条件。'
                      f'换手率核验覆盖 {result.get("turnover_coverage", 0) or 0:.1%}。'
                      '换手率源：腾讯日K；缺失不以量比替代。后续分析看空或建议减仓/卖出者不进入推荐推送。</p>')
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>技术形态候选 · {escape(result['as_of'])}</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f5f3;color:#20322f;font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1200px;margin:auto;padding:44px 28px}}h1{{font-size:34px;line-height:1.3;margin:10px 0}}h2{{font-size:22px}}.eyebrow{{color:#56746c;font-size:13px;letter-spacing:2px}}.muted,small{{color:#63756e}}.status{{background:{'#deeee8' if complete else '#f4e6cc'};padding:14px 18px;border-radius:12px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:24px 0}}.stat{{background:#fff;padding:18px;border-radius:12px}}.stat b{{display:block;font-size:30px}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}}button{{border:1px solid #ccd8d1;background:#fff;padding:9px 14px;border-radius:20px;color:#24483e;cursor:pointer}}button.active{{background:#244f43;color:#fff}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}}article{{background:#fff;border:1px solid #dde4df;border-radius:14px;padding:22px}}h3{{margin:0;font-size:21px}}.card-top{{display:flex;justify-content:space-between;gap:12px}}a{{color:#217261}}.tags{{color:#277d68;font-weight:600}}.selected{{display:inline-block;background:#dceee7;font-size:12px;padding:2px 8px;border-radius:8px;margin-left:8px}}article p,li{{font-size:14px}}ul{{padding-left:20px}}.note{{background:#e8ece8;padding:22px;border-radius:12px;margin-top:28px}}.empty{{padding:30px;background:white;border-radius:12px}}[hidden]{{display:none}}@media(max-width:700px){{main{{padding:24px 16px}}h1{{font-size:27px}}.grid{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,1fr)}}.card-top{{display:block}}}}</style>
<main><div class="eyebrow">DAILY STOCK ANALYSIS / 技术形态筛选</div><h1>技术形态候选</h1>
<p class="muted">行情截止 {escape(result['as_of'])} · 沪深京 A 股 · 规则版本 {ruleset_version} · {'底部放量反弹筛选' if bottom_focus else '原图八形态及扩展突破规则'}。</p>
{scope_line}
{delay_line}
<div class="status">{status} · 当日行情日期核验覆盖 {result.get('coverage',0):.1%}</div>
<div class="stats"><div class="stat">市场股票<b>{result.get('universe_count',0):,}</b></div><div class="stat">基础条件通过<b>{result.get('eligible_count',0):,}</b></div><div class="stat">形态候选<b>{len(result.get('candidates',[]))}</b></div><div class="stat">交给现有分析<b>{len(result.get('selected',[]))}</b></div></div>
<p class="muted">{exclusions}</p><h2>按形态查看</h2><nav>{buttons}</nav>
<div class="grid">{''.join(cards) or no_candidates}</div>
<section class="note"><h2>筛选依据与数据限制</h2>{focus_note}<p>{'传统形态模式：' if bottom_focus else ''}同一股票命中多形态时合并。研究计数取原图八形态命中数，最低为1；扩展突破提供候选资格，与原图形态重叠不重复加分。再按量比、代码排序；计数不是上涨概率，命中不代表后续必然上涨。均量、阻力与形态拟合均不包含当日突破；只接受最新完整交易日的日K。前复权窗口每天重新获取，同日缓存仅供重跑；不混入未复权实时价格。</p>
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
