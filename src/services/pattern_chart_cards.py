"""原生飞书趋势卡片：复用已核验日K，不依赖图片上传凭据或外链。"""
import json
import logging
import re
import time
from pathlib import Path

from src.pattern_screener import PATTERN_NAMES
from src.services.pattern_chart_renderer import get_chart_bars


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def build_stock_chart_card(candidate):
    code = str(candidate.get("code", ""))
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("股票代码无效")
    bars = get_chart_bars(candidate)
    if not bars or not candidate.get("data_date"):
        raise ValueError("缺少已核验的日K历史")
    patterns = " · ".join(signal["name"] for signal in candidate.get("signals", [])
                          if signal.get("name") in PATTERN_NAMES) or "见选股报告"
    for count in (60, 50, 40, 30):
        shown = bars[-count:]
        prices = [
            {"d": bar["date"], "v": round(bar[key], 3), "s": name}
            for bar in shown
            for key, name in (("close", "收盘"), ("ma5", "MA5"), ("ma10", "MA10"),
                              ("ma20", "MA20"), ("ma30", "MA30"))
            if bar[key] is not None
        ]
        volumes = [{"d": bar["date"], "v": round(bar["volume"] / 10000, 2),
                    "s": "上涨" if bar["close"] >= bar["open"] else "下跌"} for bar in shown]
        price_spec = {
            "type": "line", "data": {"values": prices},
            "xField": "d", "yField": "v", "seriesField": "s",
            "color": ["#dc4848", "#b98a19", "#437abd", "#9968b1", "#4f8263"],
            "point": {"visible": False},
            "axes": [{"orient": "bottom", "label": {"autoHide": True}},
                     {"orient": "left", "zero": False, "title": {"visible": True, "text": "价格（元）"}}],
            "legends": {"visible": True, "orient": "bottom"},
        }
        volume_spec = {
            "type": "bar", "data": {"values": volumes},
            "xField": "d", "yField": "v", "seriesField": "s",
            "color": {"type": "ordinal", "domain": ["上涨", "下跌"], "range": ["#dc4848", "#218869"]},
            "axes": [{"orient": "bottom", "label": {"autoHide": True}},
                     {"orient": "left", "title": {"visible": True, "text": "成交量（万股）"}}],
        }
        card = {
            "schema": "2.0",
            "header": {"template": "green", "title": {"tag": "plain_text",
                       "content": f"{candidate.get('name', code)} {code} · 候选股走势图"}},
            "body": {"elements": [
                {"tag": "markdown", "content": f"行情截至 {candidate['data_date']} · 前复权日K收盘价\n"
                 f"形态：{patterns}\n"
                 f"最近 {len(shown)} 个交易日 · 收盘 {bars[-1]['close']:.3f} 元\n"
                 "点击图表可在飞书内放大查看；这是收盘后快照，不是实时分时图。"},
                {"tag": "chart", "preview": True, "aspect_ratio": "4:3", "chart_spec": price_spec},
                {"tag": "chart", "preview": True, "aspect_ratio": "2:1", "chart_spec": volume_spec},
            ]},
        }
        # Reserve room for Webhook wrapper, configured keyword and signature.
        # The sender checks the exact final request bytes again.
        if len(compact_json(card).encode("utf-8")) <= 18000:
            return card
    raise ValueError("走势图卡片超过字节预算")


def deliver_chart_cards(result, service, output_dir, *, send=False):
    """保存图表卡片并按既有report路由/降噪策略发送；失败不冒充图表成功。"""
    from src.notification import NotificationChannel

    logger = logging.getLogger(__name__)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    attempts = []
    if result.get("status") != "complete":
        raise ValueError("筛选未完成，不生成推送图表")
    allowed = send and service.should_broadcast_static_channels() and (
        NotificationChannel.FEISHU in service.get_channels_for_route("report")
    )
    for candidate in result.get("selected", []):
        code = str(candidate.get("code", ""))
        attempt = {"code": code, "status": "preview" if not send else "route_skipped"}
        try:
            if candidate.get("data_date") != result.get("as_of"):
                raise ValueError("候选日期与报告日期不一致")
            card = build_stock_chart_card(candidate)
            payload = compact_json(card)
            (directory / f"{code}.json").write_text(payload, encoding="utf-8")
            if allowed:
                key = f"pattern-chart:{result['as_of']}:{code}"
                decision = service.evaluate_noise_control(
                    payload, route_type="report", dedup_key=key, cooldown_key=f"pattern-chart:{code}",
                )
                if not decision.should_send:
                    attempt.update(status="noise_suppressed", reason=decision.reason_code)
                else:
                    try:
                        success = service.send_card_to_feishu(card)
                    except Exception:
                        service.release_noise_control(decision)
                        raise
                    if success:
                        service.record_noise_control(decision)
                        attempt["status"] = "sent"
                    else:
                        service.release_noise_control(decision)
                        attempt["status"] = "failed"
                    time.sleep(1)  # stay below the existing bot's per-second rate limit
        except Exception as exc:
            attempt.update(status="failed", error=type(exc).__name__)
        attempts.append(attempt)
        logger.info("走势图 %s：%s", code, attempt["status"])
    (directory / "delivery.json").write_text(compact_json(attempts), encoding="utf-8")
    return attempts
