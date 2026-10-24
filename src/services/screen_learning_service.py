"""学习档案的持久化、审计和筛选入口适配。"""
import gzip
import html
import json
import math
import os
import re
from pathlib import Path

from src.core.screen_learning import VERSION, RULES, HARD_FILTER_DIGEST, RULESET_VERSION, build_review, order_codes, policy_digest, summary_markdown

MAX_BYTES = 80 * 1024 * 1024
MAX_DAYS = 120


def empty_state():
    return {"version": VERSION, "days": [], "active": {"rule": "baseline"}}


def validate_state(state):
    if not isinstance(state, dict) or state.get("version") != VERSION or not isinstance(state.get("days"), list):
        raise ValueError("学习档案格式无效")
    days = state["days"]
    if not isinstance(state.get("active"), dict) or state["active"].get("rule") not in {*RULES, "baseline"}:
        raise ValueError("学习档案策略无效")
    if len(days) > MAX_DAYS or [d["as_of"] for d in days] != sorted({d["as_of"] for d in days}):
        raise ValueError("学习档案日期重复或越界")
    from datetime import date
    for day in days:
        if date.fromisoformat(day["as_of"]).isoformat() != day["as_of"]:
            raise ValueError("学习档案日期无效")
        if not 0 <= day["coverage"] <= 1 or len(day["rows"]) > 10000:
            raise ValueError("学习档案覆盖无效")
        total = day.get("universe_count")
        if (type(total) is not int or not 1 <= total <= 15000 or len(day["rows"]) > total
                or not math.isclose(day["coverage"], len(day["rows"])/total, abs_tol=1e-9)):
            raise ValueError("学习档案覆盖与股票数不一致")
        for code, row in day["rows"].items():
            if (not re.fullmatch(r"\d{6}", code) or not isinstance(row, list) or len(row) != 8
                    or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in row)):
                raise ValueError("学习档案量价无效")
            if row[4] > min(row[0], row[2]) or row[3] < max(row[0], row[2]):
                raise ValueError("学习档案OHLC无效")
        for key in ("eligible", "selected", "published", "rejected", "baseline_pool"):
            codes = day.get(key, [])
            if not isinstance(codes, list) or len(codes) != len(set(codes)) or any(not re.fullmatch(r"\d{6}", x) for x in codes):
                raise ValueError("学习档案股票列表无效")
        if day.get("policy", {}).get("rule", "baseline") not in {*RULES, "baseline"}:
            raise ValueError("未知学习排序")
    return state


def load_state(path):
    path = Path(path)
    if not path.exists():
        return empty_state()
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("学习档案过大")
    return validate_state(json.loads(raw))


def save_state(state, path):
    validate_state(state)
    raw = json.dumps(state, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("学习档案过大")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    compressed = gzip.compress(raw)
    if len(compressed) > 20 * 1024 * 1024:
        raise ValueError("学习压缩档案过大")
    temporary.write_bytes(compressed)
    temporary.replace(path)


def prepare(state, snapshot, output_dir):
    validate_state(state)
    as_of = snapshot["as_of"]
    if state["days"] and state["days"][-1]["as_of"] > as_of:
        raise ValueError("不能用未来档案为过去选股训练")
    existing = next((day for day in state["days"] if day["as_of"] == as_of), None)
    if existing:
        # 同日重跑保留首次快照、排序和模型结论，不用重新取数覆盖历史。
        snapshot = existing
    else:
        state["days"].append(snapshot)
        state["days"] = state["days"][-MAX_DAYS:]
    review, feature_map = build_review(state["days"], state.get("active", {"rule": "baseline"}))
    if existing:
        review["policy"] = existing.get("policy", {"rule": "baseline", "status": "collecting", "mature_days": 0})
    else:
        snapshot["policy"] = review["policy"]
        state["active"] = review["policy"]
    snapshot["policy_digest"] = policy_digest(snapshot["policy"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    review["summary_md"] = summary_markdown(review)
    (output / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "review.md").write_text(review["summary_md"], encoding="utf-8")
    (output / "review.html").write_text(render_review_html(review), encoding="utf-8")
    context = {"version": VERSION, "as_of": as_of, "coverage": snapshot["coverage"],
               "policy": snapshot["policy"], "policy_digest": snapshot["policy_digest"], "features": feature_map}
    (output / "context.json").write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
    return review


def render_review_html(review):
    from src.stock_links import get_a_share_quote_url
    esc = html.escape
    rows = []
    for day in reversed(review["history"]):
        for group, label in (("published", "实际推送"), ("selected", "技术入选"), ("rejected", "模型否决")):
            for horizon in (1, 3, 5, 20):
                value = day["groups"][group][str(horizon)]
                hit = f"{value['hit_rate']:.1%}" if value["hit_rate"] is not None else "待核验"
                ret = f"{value['mean_return_pct']:+.2f}%" if value["mean_return_pct"] is not None else "—"
                rows.append(f"<tr><td>{esc(day['as_of'])}</td><td>{label}</td><td>{horizon}</td>"
                            f"<td>{value['evaluated']}/{value['total']}</td><td>{hit}</td><td>{ret}</td>"
                            f"<td>{value['pending']} / {value['unavailable']}</td></tr>")
    patterns = "".join(f"<tr><td>{esc(RULES[rule])}</td><td>{row['matched']}</td><td>{row['others']}</td>"
                       f"<td>{row['winner_rate']:.1%}</td><td>{row['control_rate']:.1%}</td></tr>"
                       for rule, row in review["patterns"].items()
                       if row["winner_rate"] is not None and row["control_rate"] is not None)
    winners = "".join(f'<li><a href="{get_a_share_quote_url(item["code"])}" target="_blank" rel="noopener noreferrer">'
                       f'{esc(item["code"])}</a> · 后续5日 {item["return_pct"]:+.2f}% · '
                       f'启动前观察日 {esc(item["before_date"])}</li>' for item in review["winner_examples"])
    summary = esc(review["summary_md"])
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>选股自我复盘 · {esc(review['as_of'])}</title><style>body{{font:16px/1.7 system-ui;background:#f3f6f4;color:#203b32;margin:0}}main{{max-width:1150px;margin:auto;padding:32px 20px}}section{{background:white;border:1px solid #dce5e0;border-radius:14px;padding:22px;margin:20px 0}}pre{{white-space:pre-wrap;font:inherit}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid #dae4df;padding:10px}}.scroll{{overflow:auto}}a{{color:#147652}}</style>
<main><h1>选股自我复盘</h1><p>行情日 {esc(review['as_of'])} · 先记录、后验证、再升级</p>
<section><pre>{summary}</pre></section><section><h2>历史选股表现</h2><div class="scroll"><table><tr><th>选股日</th><th>分组</th><th>交易日后</th><th>已核验/总数</th><th>上涨比例</th><th>平均涨跌</th><th>未到期/缺数据</th></tr>{''.join(rows)}</table></div><p>空表表示尚无后续交易日，不能伪造准确率。实际推送只认完整成功回执。</p></section>
<section><h2>上涨前的特征与对照</h2><p>赢家定义为随后5个交易日涨幅至少5%；特征取启动前，未上涨股票也进入对照。相关性不是因果，也不是未来收益承诺。</p><div class="scroll"><table><tr><th>事前特征</th><th>符合样本</th><th>对照样本</th><th>符合组赢家比例</th><th>对照组赢家比例</th></tr>{patterns}</table></div></section>
<section><h2>近期上涨股票样本</h2><ul>{winners}</ul><p>未到5个后续交易日时不列出赢家。名称链接继续使用雪球，页面不绘制股票曲线。</p></section></main></html>'''


def apply_learned_ranking(candidates, context, as_of):
    # 仅在既有硬条件之后改变顺序；未验证/日期不符/快照覆盖不足时沿用基线。
    baseline = [item["code"] for item in candidates]
    fallback = {"applied": False, "rule": "baseline", "baseline_pool": baseline}
    try:
        if (not isinstance(context, dict) or context.get("version") != VERSION
                or context.get("as_of") != as_of.isoformat()
                or not .95 <= context.get("coverage", 0) <= 1
                or context.get("policy_digest") != policy_digest(context.get("policy", {}))):
            return fallback
        policy = context["policy"]
        if (policy.get("rule") not in {*RULES, "baseline"}
                or policy.get("promoted_at", as_of.isoformat()) > as_of.isoformat()
                or policy.get("rule") != "baseline" and policy.get("status") not in {"promoted", "monitoring"}):
            return fallback
        if (policy.get("rule") in RULES and (policy.get("hard_filter_digest") != HARD_FILTER_DIGEST
                                            or policy.get("ruleset_version") != RULESET_VERSION)):
            return fallback
        for code in baseline:
            value = context["features"][code]
            if (not math.isfinite(value["return"]) or not 0 <= value["close_position"] <= 1
                    or any(value.get(key) is not None and (not math.isfinite(value[key]) or value[key] <= 0)
                           for key in ("turnover_growth", "volume_ratio"))):
                return fallback
            if value.get("prior_five_return") is not None and not math.isfinite(value["prior_five_return"]):
                return fallback
    except (KeyError, ValueError, TypeError, AttributeError):
        return fallback
    rule = policy.get("rule", "baseline")
    ordered = order_codes(baseline, context.get("features", {}), rule)
    lookup = {code: index for index, code in enumerate(ordered)}
    candidates.sort(key=lambda item: lookup[item["code"]])
    return {"applied": rule in RULES, "rule": rule, "policy_digest": context["policy_digest"],
            "baseline_pool": baseline}


def record_selection(state, screen, audit=None, receipt=None):
    day = state["days"][-1]
    if screen.get("as_of") != day["as_of"] or screen.get("status") != "complete":
        return
    selected = [item["code"] for item in screen["selected"]]
    pool = screen.get("learning", {}).get("baseline_pool", [item["code"] for item in screen["candidates"]])
    if "selected" in day:
        if day["selected"] != selected or day["baseline_pool"] != pool:
            raise ValueError("同日股票池已冻结，不能覆盖")
        if day.get("delivery_status") == "acknowledged":
            return
    else:
        day.update(selected=selected, baseline_pool=pool, profile=screen.get("profile"),
                   ruleset_version=screen.get("ruleset_version"), parameters=screen.get("parameters"),
                   top_n=screen.get("parameters", {}).get("top_n", 10), published=[], rejected=[],
                   delivery_status="unverified")
        limits = (screen.get("parameters") or {}).get("bottom_reversal")
        if isinstance(limits, dict):
            import hashlib
            day["hard_filter_digest"] = hashlib.sha256(json.dumps(limits, sort_keys=True).encode()).hexdigest()[:16]
    if not audit or audit.get("as_of") != day["as_of"] or set(audit.get("input_codes", [])) != set(selected):
        return
    if not set(audit.get("accepted", [])).issubset(selected) or not set(audit.get("rejected", [])).issubset(selected):
        raise ValueError("模型审计不属于本次股票池")
    if set(audit.get("accepted", [])) & set(audit.get("rejected", [])):
        raise ValueError("模型接收与否决记录冲突")
    day["rejected"] = audit.get("rejected", [])
    day["model_accepted"] = audit.get("accepted", [])
    if not audit.get("accepted"):
        day["delivery_status"] = "no_recommendations"
    if receipt:
        from src.core.daily_delivery import validate_receipt
        try:
            validate_receipt(receipt)
        except (ValueError, TypeError, KeyError):
            return
    delivery = (receipt or {}).get("components", {}).get("stocks", {})
    identity = (receipt or {}).get("identity", {})
    if (identity.get("as_of") == day["as_of"] and str(identity.get("run_id")) == str(audit.get("run_id"))
            and delivery.get("status") == "acknowledged" and delivery.get("finalized") is True
            and not delivery.get("active_call") and not delivery.get("failed_calls")
            and delivery.get("posts", {}).get("ack", 0) > 0
            and not delivery.get("posts", {}).get("unknown") and not delivery.get("posts", {}).get("rejected")):
        day["published"] = audit["accepted"]
        day["delivery_status"] = "acknowledged"


def notification_summary(as_of):
    target = os.getenv("SCREEN_LEARNING_REVIEW")
    if not target:
        return ""
    try:
        review = json.loads(Path(target).read_text(encoding="utf-8"))
        if not isinstance(review, dict) or review.get("as_of") != as_of:
            return ""
        summary = review.get("summary_md", "")
        return summary if isinstance(summary, str) else ""
    except (OSError, ValueError, TypeError):
        return ""


def write_model_audit(results, exclusions, input_codes):
    target = os.getenv("SCREEN_LEARNING_MODEL_AUDIT")
    if not target:
        return
    as_of = os.getenv("PATTERN_COMPLETED_SESSION")
    if not as_of:
        return
    output = Path(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"as_of": as_of, "run_id": os.getenv("GITHUB_RUN_ID", "local"),
                                  "input_codes": input_codes, "accepted": [r.code for r in results],
                                  "rejected": [item["code"] for item in exclusions]}, ensure_ascii=False), encoding="utf-8")
