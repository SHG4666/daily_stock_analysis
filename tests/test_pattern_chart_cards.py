"""Real chart contents, byte budgets, delivery policy and failure accounting."""
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from scripts.send_pattern_charts import main
from src.notification import NotificationChannel
from src.services.pattern_chart_cards import build_stock_chart_card, compact_json, deliver_chart_cards


def candidate(code="001391"):
    return {
        "code": code, "name": "国货航", "data_date": "2026-10-12",
        "chart_bars": [{"date": f"2026-08-{day:02d}", "open": 10, "low": 9,
                        "high": 12, "close": 11, "volume": 20000,
                        "ma5": 10.5, "ma10": 10.25, "ma20": 10.125, "ma30": None}
                       for day in range(1, 31)] +
                      [{"date": "2026-10-12", "open": 11, "low": 10,
                        "high": 13, "close": 12, "volume": 30000,
                        "ma5": 11, "ma10": 10.5, "ma20": 10.2, "ma30": 10.1}],
    }


def service(channels=None, should_send=True, outcome=True):
    return SimpleNamespace(
        should_broadcast_static_channels=Mock(return_value=True),
        get_channels_for_route=Mock(return_value=[NotificationChannel.FEISHU] if channels is None else channels),
        evaluate_noise_control=Mock(return_value=SimpleNamespace(should_send=should_send, reason_code="quiet_hours")),
        send_card_to_feishu=Mock(return_value=outcome),
        record_noise_control=Mock(), release_noise_control=Mock(),
    )


def report(*stocks):
    return {"status": "complete", "as_of": "2026-10-12", "selected": list(stocks or [candidate()])}


def test_chart_payload_keeps_date_prices_mas_and_volume_units_under_budget():
    card = build_stock_chart_card(candidate())
    price, volume = card["body"]["elements"][1:]
    assert price["preview"] and volume["preview"]
    values = price["chart_spec"]["data"]["values"]
    assert {x["s"]: x["v"] for x in values if x["d"] == "2026-10-12"} == {
        "收盘": 12, "MA5": 11, "MA10": 10.5, "MA20": 10.2, "MA30": 10.1,
    }
    assert volume["chart_spec"]["data"]["values"][-1]["v"] == 3
    assert len(compact_json(card).encode("utf-8")) <= 18000


@pytest.mark.parametrize("change", ["code", "stale", "nan"])
def test_invalid_chart_never_generates_or_sends_misleading_data(change):
    stock = candidate()
    if change == "code":
        stock["code"] = "../unsafe"
    elif change == "stale":
        stock["data_date"] = "2026-10-13"
    else:
        stock["chart_bars"][-1]["close"] = float("nan")
    with pytest.raises(ValueError):
        build_stock_chart_card(stock)


@pytest.mark.parametrize("mode", ["preview", "excluded", "quiet", "context"])
def test_delivery_respects_preview_route_quiet_hours_and_context(mode, tmp_path):
    notification = service(channels=[] if mode == "excluded" else None, should_send=mode != "quiet")
    if mode == "context":
        notification.should_broadcast_static_channels.return_value = False
    results = deliver_chart_cards(report(), notification, tmp_path, send=mode != "preview")
    notification.send_card_to_feishu.assert_not_called()
    assert results[0]["status"] in ("preview", "route_skipped", "noise_suppressed")
    assert (tmp_path / "001391.json").exists()


def test_each_stock_gets_its_own_noise_key_and_failed_send_releases_reservation(tmp_path):
    notification = service()
    notification.send_card_to_feishu.side_effect = [True, False]
    with patch("src.services.pattern_chart_cards.time.sleep"):
        results = deliver_chart_cards(report(candidate(), candidate("300879")), notification, tmp_path, send=True)
    assert [x["status"] for x in results] == ["sent", "failed"]
    assert notification.record_noise_control.call_count == 1
    assert notification.release_noise_control.call_count == 1
    assert notification.evaluate_noise_control.call_args_list[0].kwargs["cooldown_key"] != (
        notification.evaluate_noise_control.call_args_list[1].kwargs["cooldown_key"])
    assert json.loads((tmp_path / "delivery.json").read_text()) == results


def test_incomplete_report_cannot_push_charts(tmp_path):
    incomplete = deepcopy(report())
    incomplete["status"] = "incomplete"
    with pytest.raises(ValueError):
        deliver_chart_cards(incomplete, service(), tmp_path, send=True)


def test_cli_refuses_to_send_an_old_report_before_initializing_notification(tmp_path):
    old = report()
    old["as_of"] = "2000-01-01"
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    with patch("src.notification.NotificationService") as notification:
        assert main(["--report", str(path), "--send"]) == 1
    notification.assert_not_called()
