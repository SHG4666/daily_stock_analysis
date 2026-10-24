"""有时间边界的选股反馈与有限策略升级。只优化候选排序，不修改底部硬条件。"""
import hashlib
import json
import math
from datetime import date, timedelta
from types import SimpleNamespace
from functools import lru_cache

import numpy as np
from src.core.backtest_engine import BacktestEngine, EvaluationConfig
from src.bottom_reversal import LIMITS, RULESET_VERSION

VERSION = 1
HORIZONS = (1, 3, 5, 20)
RULES = {
    "close_strong": "收盘位于日内振幅上方20%",
    "moderate_rise": "当日上涨2%至6%且收盘靠近日高",
    "turnover_rise": "实际换手率较昨日提高至少50%",
    "volume_rise": "成交量达到此前20日均量1.8倍",
    "volume_and_turnover": "放量、换手率同步提高且收盘偏强",
    "base_then_volume": "此前5日横盘或回落后放量收高",
}
MIN_DAYS = 90
MIN_COVERAGE = .95
HARD_FILTER_DIGEST = hashlib.sha256(json.dumps(LIMITS, sort_keys=True).encode()).hexdigest()[:16]


@lru_cache(maxsize=1024)
def sessions(start, end):
    # 学习不使用日历fail-open：缺交易日历就不能声称“次日”结果。
    import exchange_calendars as xcals
    return [item.date().isoformat() for item in xcals.get_calendar("XSHG").sessions_in_range(start, end)]


@lru_cache(maxsize=1024)
def following_sessions(start, count):
    # 只查询已公布日历。未知的次年开市日保持待定，不让远端边界阻断已到期结果。
    import exchange_calendars as xcals
    end = min(date.fromisoformat(start) + timedelta(days=50),
              xcals.get_calendar("XSHG").last_session.date())
    return sessions(start, end)[1:count+1]


def features(days, index):
    day = days[index]
    expected = sessions(date.fromisoformat(day["as_of"]) - timedelta(days=45), day["as_of"])
    history = {item["as_of"]: item for item in days[:index]}
    previous = history.get(expected[-2], {}).get("rows", {}) if len(expected) > 1 else {}
    twenty = expected[-21:-1]
    result = {}
    for code, row in day["rows"].items():
        close, prev, opening, high, low, volume, amount, turnover = row
        prior = previous.get(code)
        volumes = [history.get(d, {}).get("rows", {}).get(code, [None] * 8)[5] for d in twenty]
        prior_five = [history.get(d, {}).get("rows", {}).get(code) for d in expected[-6:-1]]
        result[code] = {
            "return": close / prev - 1,
            "close_position": (close - low) / (high - low) if high > low else 0,
            "turnover_growth": turnover / prior[7] if prior and prior[7] > 0 else None,
            "volume_ratio": volume / np.mean(volumes) if len(volumes) == 20 and all(volumes) else None,
            "prior_five_return": float(np.prod([r[0]/r[1] for r in prior_five])-1)
            if len(prior_five) == 5 and all(prior_five) else None,
        }
    return result


def matches(rule, value):
    if not value:
        return False
    if rule == "close_strong":
        return value["close_position"] >= .8
    if rule == "moderate_rise":
        return .02 <= value["return"] <= .06 and value["close_position"] >= .7
    if rule == "turnover_rise":
        return value.get("turnover_growth") is not None and value["turnover_growth"] >= 1.5
    if rule == "volume_rise":
        return value.get("volume_ratio") is not None and value["volume_ratio"] >= 1.8
    if rule == "volume_and_turnover":
        return matches("volume_rise", value) and matches("turnover_rise", value) and value["close_position"] >= .65
    if rule == "base_then_volume":
        return (value.get("prior_five_return") is not None and -.15 <= value["prior_five_return"] <= .02
                and matches("volume_rise", value) and value["close_position"] >= .65)
    return False


def observed(rule, value):
    required = {"turnover_rise": ("turnover_growth",), "volume_rise": ("volume_ratio",),
                "volume_and_turnover": ("turnover_growth", "volume_ratio"),
                "base_then_volume": ("volume_ratio", "prior_five_return")}
    return bool(value) and all(value.get(key) is not None for key in required.get(rule, ()))


def order_codes(codes, feature_map, rule):
    return sorted(codes, key=lambda code: not matches(rule, feature_map.get(code))) if rule in RULES else list(codes)


def forward_result(days, index, code, horizon):
    start = days[index]["as_of"]
    expected = following_sessions(start, horizon)
    last = days[-1]["as_of"]
    if len(expected) != horizon or expected[-1] > last:
        return {"status": "pending", "expected_end": expected[-1] if len(expected) == horizon else None}
    by_date = {item["as_of"]: item for item in days[index+1:]}
    bars, price = [], 100.
    for d in expected:
        row = by_date.get(d, {}).get("rows", {}).get(code)
        if not row:
            return {"status": "unavailable", "expected_end": expected[-1]}
        # 使用同源前收基准构造涨跌幅指数。不是D收盘买入的可交易回测。
        bars.append(SimpleNamespace(date=date.fromisoformat(d), high=price*row[3]/row[1],
                                    low=price*row[4]/row[1], close=price*row[0]/row[1]))
        price = bars[-1].close
    raw = BacktestEngine.evaluate_single(operation_advice="买入", analysis_date=date.fromisoformat(start),
                                        start_price=100., forward_bars=bars, stop_loss=None, take_profit=None,
                                        config=EvaluationConfig(horizon, neutral_band_pct=0))
    return {"status": "complete", "return_pct": raw["stock_return_pct"],
            "up": raw["stock_return_pct"] > 0, "max_adverse_pct": min(b.low for b in bars)-100,
            "expected_end": expected[-1]}


def group_metrics(days, index, codes, horizon):
    outcomes = {code: forward_result(days, index, code, horizon) for code in codes}
    ready = [value for value in outcomes.values() if value["status"] == "complete"]
    return {"total": len(codes), "evaluated": len(ready),
            "pending": sum(value["status"] == "pending" for value in outcomes.values()),
            "unavailable": sum(value["status"] == "unavailable" for value in outcomes.values()),
            "hit_rate": sum(value["up"] for value in ready) / len(ready) if ready else None,
            "mean_return_pct": float(np.mean([v["return_pct"] for v in ready])) if ready else None,
            "outcomes": outcomes}


def winner_patterns(days, indices, maps, minimum=30):
    """先冻结上涨前特征，再用未来5日>=5%的结果标签，含未上涨对照。"""
    stats = {rule: {"matched": 0, "winners": 0, "others": 0, "other_winners": 0} for rule in RULES}
    for index in indices:
        for code in days[index]["eligible"]:
            outcome = forward_result(days, index, code, 5)
            if outcome["status"] != "complete":
                continue
            winner = outcome["return_pct"] >= 5
            for rule, row in stats.items():
                hit = matches(rule, maps[index].get(code))
                # 缺特征不归为反例；两个组都必须有真实观察值。
                if not observed(rule, maps[index].get(code)):
                    continue
                row["matched" if hit else "others"] += 1
                row["winners" if hit else "other_winners"] += int(winner)
    for row in stats.values():
        row["winner_rate"] = row["winners"] / row["matched"] if row["matched"] else None
        row["control_rate"] = row["other_winners"] / row["others"] if row["others"] else None
        row["lift"] = (row["winner_rate"] - row["control_rate"]
                       if min(row["matched"], row["others"]) >= minimum else None)
    return stats


def paired_scores(days, indices, maps, rule):
    result = []
    for index in indices:
        day = days[index]
        pool = day.get("baseline_pool", [])
        top = day.get("top_n", 10)
        if (day.get("profile") != "bottom_reversal" or day["coverage"] < MIN_COVERAGE
                or len(pool) <= top or any(code not in maps[index] for code in pool)):
            continue
        baseline = group_metrics(days, index, pool[:top], 1)
        trial = group_metrics(days, index, order_codes(pool, maps[index], rule)[:top], 1)
        if baseline["evaluated"] != top or trial["evaluated"] != top:
            continue
        result.append({"as_of": day["as_of"], "delta": trial["mean_return_pct"] - baseline["mean_return_pct"],
                       "hit_delta": trial["hit_rate"] - baseline["hit_rate"],
                       "trial": trial["mean_return_pct"], "baseline": baseline["mean_return_pct"]})
    return result


def block_improvement(values):
    """五个交易日一组，减少把同日股票数当成独立时间样本的误判。"""
    if len(values) < 20:
        return False
    blocks = [np.mean([v["delta"] for v in values[i:i+5]]) for i in range(0, len(values)-4, 5)]
    return (len(blocks) >= 4 and np.mean(blocks) >= .2
            and np.mean(blocks) > 2 * np.std(blocks, ddof=1) / math.sqrt(len(blocks))
            and np.mean([v["hit_delta"] for v in values]) >= .05
            and min(v["trial"] for v in values) >= min(v["baseline"] for v in values) - 2)


def score_summary(values):
    return {"days": len(values), "mean_excess_pp": float(np.mean([v["delta"] for v in values])) if values else None,
            "hit_rate_improvement": float(np.mean([v["hit_delta"] for v in values])) if values else None,
            "passed": bool(block_improvement(values)), "paired_daily_results": values}


def select_policy(days, active, maps):
    """训练选候选，隔离标签重叠后验证及最终留出；样本不足保持基线。"""
    recent_sessions = set(sessions(date.fromisoformat(days[-1]["as_of"]) - timedelta(days=250), days[-1]["as_of"])[-120:])
    eligible = [i for i, day in enumerate(days[:-5]) if day["coverage"] >= MIN_COVERAGE and day["as_of"] in recent_sessions
                and day.get("profile") == "bottom_reversal" and day.get("baseline_pool")
                and day.get("hard_filter_digest") == HARD_FILTER_DIGEST and day.get("ruleset_version") == RULESET_VERSION]
    decision = {"rule": active.get("rule", "baseline"), "status": "collecting",
                "mature_days": len(eligible), "required_days": MIN_DAYS,
                "hard_filter_digest": HARD_FILTER_DIGEST, "ruleset_version": RULESET_VERSION}
    if decision["rule"] in RULES and (active.get("hard_filter_digest") != HARD_FILTER_DIGEST
                                      or active.get("ruleset_version") != RULESET_VERSION):
        return {**decision, "rule": "baseline", "status": "ruleset_changed"}
    if decision["rule"] in RULES:
        expected_previous = sessions(date.fromisoformat(days[-1]["as_of"]) - timedelta(days=45), days[-1]["as_of"])[-2]
        if len(days) < 2 or days[-2]["as_of"] != expected_previous or days[-2]["coverage"] < MIN_COVERAGE:
            return {**decision, "rule": "baseline", "status": "history_gap", "reviewed_at": days[-1]["as_of"]}
    # 已升级策略在上线之后的20个可评估日期持续与原排序比较，明显退化回滚。
    if decision["rule"] in RULES:
        after = [i for i in eligible if days[i]["as_of"] > active.get("promoted_at", "9999")]
        scores = paired_scores(days, after, maps, decision["rule"])[-20:]
        if len(scores) == 20 and (np.mean([v["delta"] for v in scores]) < -.2
                                  or np.mean([v["hit_delta"] for v in scores]) < -.1):
            return {**decision, "rule": "baseline", "status": "rolled_back",
                    "promoted_at": days[-1]["as_of"], "reviewed_at": days[-1]["as_of"]}
        return {**active, **decision, "status": "monitoring"}
    if len(eligible) < MIN_DAYS:
        return decision
    reviewed = active.get("reviewed_at")
    if reviewed and len(sessions(reviewed, days[-1]["as_of"])) < 21:
        return {**decision, "reviewed_at": reviewed, "status": "cooldown"}
    if date.fromisoformat(days[-1]["as_of"]).weekday() != 4:
        return {**decision, "status": "waiting_weekly_review"}
    decision["reviewed_at"] = days[-1]["as_of"]
    # 固定25日验证+25日留出，边界空出5日，使5日赢家标签不跨分割边界。
    train, validation, holdout = eligible[:-60], eligible[-55:-30], eligible[-25:]
    patterns = winner_patterns(days, train, maps)
    candidates = [rule for rule, row in patterns.items() if row["lift"] is not None and row["lift"] > .05]
    if not candidates:
        return {**decision, "status": "no_improvement"}
    rule = max(candidates, key=lambda key: (patterns[key]["lift"], key))
    validation_scores = paired_scores(days, validation, maps, rule)
    holdout_scores = paired_scores(days, holdout, maps, rule)
    audit = {"train_end": days[train[-1]]["as_of"], "validation_start": days[validation[0]]["as_of"],
             "holdout_start": days[holdout[0]]["as_of"], "candidate_rule": rule,
             "validation_days": len(validation_scores), "holdout_days": len(holdout_scores),
             "training_pattern": patterns[rule], "validation": score_summary(validation_scores),
             "holdout": score_summary(holdout_scores)}
    if block_improvement(validation_scores) and block_improvement(holdout_scores):
        return {**decision, **audit, "rule": rule, "status": "promoted", "promoted_at": days[-1]["as_of"]}
    return {**decision, **audit, "status": "validation_failed"}


def build_review(days, active):
    maps = {i: features(days, i) for i in range(len(days))}
    latest = days[-1]
    history = []
    for i in range(max(0, len(days)-21), len(days)-1):
        if "selected" not in days[i]:
            continue
        groups = {name: {str(h): group_metrics(days, i, days[i].get(name, []), h) for h in HORIZONS}
                  for name in ("selected", "published", "rejected")}
        benchmark = group_metrics(days, i, days[i]["eligible"], 1)
        history.append({"as_of": days[i]["as_of"], "groups": groups,
                        "market_mean_return_pct": benchmark["mean_return_pct"],
                        "market_evaluated": benchmark["evaluated"], "market_total": benchmark["total"],
                        "delivery_status": days[i].get("delivery_status", "unverified")})
    examples = []
    if len(days) >= 6:
        start_index = len(days)-6
        for code in days[start_index]["eligible"]:
            outcome = forward_result(days, start_index, code, 5)
            if outcome["status"] == "complete" and outcome["expected_end"] == latest["as_of"] and outcome["return_pct"] >= 5:
                examples.append({"code": code, "return_pct": outcome["return_pct"],
                                 "before_date": days[start_index]["as_of"],
                                 "before_features": maps[start_index].get(code),
                                 "rules": [rule for rule in RULES if matches(rule, maps[start_index].get(code))]})
        examples.sort(key=lambda item: (-item["return_pct"], item["code"]))
    patterns = winner_patterns(days, range(max(0, len(days)-26), max(0, len(days)-5)), maps)
    policy = select_policy(days, active, maps)
    return {"version": VERSION, "as_of": latest["as_of"], "history": history, "patterns": patterns,
            "policy": policy, "coverage": latest["coverage"],
            "winner_examples": examples[:20],
            "current_selection": {"selected": latest.get("selected", []), "published": latest.get("published", []),
                                  "rejected": latest.get("rejected", []),
                                  "delivery_status": latest.get("delivery_status", "unverified")},
            "note": "收盘涨跌幅诊断，非可成交收益；不模拟D收盘买入、不含费用，不能当T+1交易回测。"}, maps[len(days)-1]


def summary_markdown(review):
    lines = ["## 选股自我复盘", f"行情截至 {review['as_of']} · 快照有效覆盖 {review['coverage']:.1%}"]
    if review["history"]:
        latest = review["history"][-1]
        for group, label in (("published", "实际推送"), ("selected", "技术选入"), ("rejected", "模型否决")):
            value = latest["groups"][group]["1"]
            if group == "published" and latest.get("delivery_status") != "acknowledged":
                lines.append("实际推送：没有确认发送的股票记录，不计算准确率。")
                continue
            if value["evaluated"]:
                lines.append(f"{latest['as_of']} {label}：次日已核验{value['evaluated']}/{value['total']}只，"
                             f"上涨比例{value['hit_rate']:.1%}，平均涨跌{value['mean_return_pct']:+.2f}%，"
                             f"未到期{value['pending']}、数据缺失{value['unavailable']}只。")
                if group == "published" and latest["market_mean_return_pct"] is not None:
                    lines.append(f"同期市场对照平均涨跌{latest['market_mean_return_pct']:+.2f}%，"
                                 f"相对对照{value['mean_return_pct']-latest['market_mean_return_pct']:+.2f}个百分点；"
                                 f"对照核验{latest['market_evaluated']}/{latest['market_total']}只。")
            else:
                lines.append(f"{latest['as_of']} {label}：次日结果尚未到期或数据不足，不能计算准确率。")
    else:
        lines.append("首日建档，尚无次日结果；周末/休市不当成预测失败。")
    current = review.get("current_selection", {})
    if current.get("selected"):
        lines.append(f"当前交易日已建档：技术入选{len(current['selected'])}只，"
                     f"确认推送{len(current.get('published', []))}只，模型否决{len(current.get('rejected', []))}只；"
                     "后续交易日收盘后再评价，尚未产生未来结果。")
    policy = review["policy"]
    status = {"collecting": "积累样本", "monitoring": "监测已验证排序", "promoted": "升级排序",
              "rolled_back": "回滚原排序", "history_gap": "历史有缺口，使用原排序", "ruleset_changed": "规则已变更，重新验证",
              "cooldown": "等待新的独立观察数据", "waiting_weekly_review": "等待周度验证",
              "no_improvement": "未发现可靠改善", "validation_failed": "验证未通过，保留原排序"}.get(policy["status"], policy["status"])
    lines.append(f"自动优化：{status}，成熟记录{policy['mature_days']}/{MIN_DAYS}个交易日；"
                 f"当前排序：{RULES.get(policy['rule'], '原换手率提升排序')}。底部硬条件及看空否决保持。")
    lines.append(review["note"])
    return "\n\n".join(lines)


def policy_digest(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]
