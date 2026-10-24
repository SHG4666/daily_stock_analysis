"""事前记录、事后标签、市场对照、训练隔离、发送证据与故障降级的回归。"""
import copy
import json
from datetime import date
from types import SimpleNamespace

import pytest

from data_provider.learning_market_data import parse_batch
from src.core import screen_learning as core
from src.services import screen_learning_service as service


def day(d, codes=("000001", "000002")):
    return {"as_of": d, "rows": {code: [10., 10., 9.9, 10.1, 9.8, 10000., 100000., 1.] for code in codes},
            "eligible": list(codes), "coverage": 1., "universe_count": len(codes),
            "profile": "bottom_reversal", "ruleset_version": core.RULESET_VERSION,
            "hard_filter_digest": core.HARD_FILTER_DIGEST, "baseline_pool": list(codes), "selected": list(codes),
            "top_n": 1, "published": [codes[0]], "rejected": [], "delivery_status": "acknowledged"}


def test_next_session_is_monday_not_saturday_and_missing_monday_is_not_tuesday():
    days = [day("2026-10-23")]
    assert core.forward_result(days, 0, "000001", 1) == {"status": "pending", "expected_end": "2026-10-26"}
    days.append(day("2026-10-27"))
    assert core.forward_result(days, 0, "000001", 1)["status"] == "unavailable"


def test_known_calendar_boundary_does_not_break_matured_results_or_guess_future_dates():
    import exchange_calendars as xcals
    calendar = xcals.get_calendar("XSHG")
    last_dates = [d.date().isoformat() for d in calendar.sessions[-2:]]
    days = [day(d) for d in last_dates]
    assert core.forward_result(days, 0, "000001", 1)["status"] == "complete"
    assert core.forward_result(days, 0, "000001", 3) == {"status": "pending", "expected_end": None}


def test_directional_evaluation_and_market_control_include_losing_stocks():
    days = [day("2026-10-22"), day("2026-10-23")]
    days[1]["rows"]["000001"][:5] = [11., 10., 10., 11., 9.9]
    days[1]["rows"]["000002"][:5] = [9., 10., 9.5, 10., 8.9]
    metric = core.group_metrics(days, 0, ["000001", "000002"], 1)
    assert metric["evaluated"] == 2 and metric["hit_rate"] == .5
    assert metric["mean_return_pct"] == pytest.approx(0)
    review, _ = core.build_review(days, {"rule": "baseline"})
    assert review["history"][0]["groups"]["published"]["1"]["mean_return_pct"] == pytest.approx(10.)
    assert review["history"][0]["market_mean_return_pct"] == pytest.approx(0)
    assert "非可成交收益" in review["note"]


def test_features_do_not_change_when_future_prices_or_volumes_change():
    days = [day("2026-10-22"), day("2026-10-23"), day("2026-10-26")]
    before = core.features(days, 1)
    days[2]["rows"]["000001"] = [999., 1., 20., 1000., 1., 1e9, 1e12, 99.]
    assert core.features(days, 1) == before
    assert core.features(days, 1)["000001"]["turnover_growth"] == 1
    assert core.features(days, 0)["000001"]["turnover_growth"] is None


def test_growth_rules_require_real_history_not_filled_values():
    days = [day("2026-10-22"), day("2026-10-26")]
    value = core.features(days, 1)["000001"]
    assert value["turnover_growth"] is None and value["volume_ratio"] is None
    assert not core.matches("turnover_rise", value) and not core.matches("volume_rise", value)


def quote(daystamp="20261023150000", code="000001"):
    fields = [""] * 50
    for idx, value in {2: code, 3: "10.5", 4: "10", 5: "10.1", 6: "100", 30: daystamp,
                       33: "10.6", 34: "10", 35: "10.5/100/105000", 38: "1", 44: "0.105"}.items():
        fields[idx] = value
    return 'v_sz000001="' + "~".join(fields) + '";'


@pytest.mark.parametrize("stamp", ["20261022150000", "20261023145959", "bad", "20261026150000"])
def test_snapshot_rejects_stale_intraday_invalid_and_future_quotes(stamp):
    assert parse_batch(quote(stamp), {"000001": "sz000001"}, date(2026, 10, 23)) == {}


def test_snapshot_reuses_volume_and_amount_parser_and_checks_stock_identity():
    rows = parse_batch(quote(), {"000001": "sz000001"}, date(2026, 10, 23))
    assert rows["000001"][5] == 10000 and rows["000001"][6] == 105000
    assert parse_batch(quote(code="600519"), {"000001": "sz000001"}, date(2026, 10, 23)) == {}


def test_winners_use_pre_move_features_and_include_nonwinners():
    dates = core.sessions("2026-10-19", "2026-10-26")
    days = [day(d) for d in dates]
    days[0]["rows"]["000001"][:5] = [10., 10., 9.9, 10., 9.]
    days[0]["rows"]["000002"][:5] = [10., 10., 9.9, 11., 9.]
    for item in days[1:]:
        item["rows"]["000001"][:2] = [10.2, 10.]
        item["rows"]["000002"][:2] = [9.9, 10.]
    maps = {i: core.features(days, i) for i in range(len(days))}
    result = core.winner_patterns(days, [0], maps, minimum=1)["close_strong"]
    assert result["matched"] == result["winners"] == 1
    assert result["others"] == 1 and result["other_winners"] == 0
    assert result["lift"] == 1
    # 改赢家最后一天形态，不应改启动前的特征分组。
    days[-1]["rows"]["000001"][3] = 50
    assert core.winner_patterns(days, [0], maps, minimum=1)["close_strong"] == result


def test_cold_start_and_same_day_rerun_keep_policy_and_original_snapshot(tmp_path):
    state = service.empty_state()
    snapshot = day("2026-10-23")
    review = service.prepare(state, snapshot, tmp_path)
    assert review["policy"]["rule"] == "baseline" and review["policy"]["status"] == "collecting"
    assert "不能伪造准确率" in (tmp_path / "review.html").read_text()
    old = copy.deepcopy(state["days"][0]["rows"])
    changed = day("2026-10-23")
    changed["rows"]["000001"][0] = 20
    service.prepare(state, changed, tmp_path)
    assert state["days"][0]["rows"] == old and len(state["days"]) == 1
    with pytest.raises(ValueError, match="未来"):
        service.prepare(state, day("2026-10-22"), tmp_path)


def context():
    policy = {"rule": "close_strong", "status": "promoted", "promoted_at": "2026-10-22",
              "hard_filter_digest": core.HARD_FILTER_DIGEST, "ruleset_version": core.RULESET_VERSION}
    return {"version": 1, "as_of": "2026-10-23", "coverage": 1., "policy": policy,
            "policy_digest": core.policy_digest(policy), "features": {
                "000001": {"return": .03, "close_position": .5},
                "000002": {"return": .04, "close_position": .9}}}


def test_learning_only_reorders_approved_candidates_and_preserves_original_order_for_audit():
    candidates = [{"code": "000001"}, {"code": "000002"}]
    result = service.apply_learned_ranking(candidates, context(), date(2026, 10, 23))
    assert [v["code"] for v in candidates] == ["000002", "000001"]
    assert result["baseline_pool"] == ["000001", "000002"] and result["applied"]


@pytest.mark.parametrize("invalid", [None, [], {"version": 99}, "bad"])
def test_broken_learning_context_cannot_interrupt_normal_screening(invalid):
    candidates = [{"code": "000001"}]
    assert not service.apply_learned_ranking(candidates, invalid, date(2026, 10, 23))["applied"]
    assert candidates == [{"code": "000001"}]


@pytest.mark.parametrize("change", ["stale", "future_policy", "coverage", "feature_missing", "nan", "digest"])
def test_unverified_learning_state_never_changes_ranking(change):
    data = context()
    if change == "stale": data["as_of"] = "2026-10-22"
    elif change == "future_policy":
        data["policy"]["promoted_at"] = "2026-10-26"
        data["policy_digest"] = core.policy_digest(data["policy"])
    elif change == "coverage": data["coverage"] = .9
    elif change == "feature_missing": data["features"].pop("000002")
    elif change == "nan": data["features"]["000002"]["return"] = float("nan")
    else: data["policy_digest"] = "tampered"
    candidates = [{"code": "000001"}, {"code": "000002"}]
    assert not service.apply_learned_ranking(candidates, data, date(2026, 10, 23))["applied"]
    assert candidates[0]["code"] == "000001"


def test_compressed_state_roundtrip_and_invalid_rows_are_not_silently_reset(tmp_path):
    state = service.empty_state()
    state["days"] = [day("2026-10-23")]
    path = tmp_path / "state.json.gz"
    service.save_state(state, path)
    assert service.load_state(path) == state
    state["days"][0]["rows"]["000001"][0] = float("nan")
    with pytest.raises(ValueError): service.save_state(state, path)
    path.write_bytes(b"broken")
    with pytest.raises(Exception): service.load_state(path)


def test_cold_start_does_not_optimize_even_if_a_small_sample_looks_perfect():
    days = [day(d) for d in core.sessions("2026-09-01", "2026-10-23")]
    assert core.select_policy(days, {"rule": "baseline"}, {})["status"] == "collecting"
    assert not core.block_improvement([{"delta": 100, "hit_delta": 1, "trial": 100, "baseline": 0}]*5)


def test_model_audit_records_all_vetoes_without_inventing_publication(tmp_path, monkeypatch):
    path = tmp_path / "audit.json"
    monkeypatch.setenv("SCREEN_LEARNING_MODEL_AUDIT", str(path))
    monkeypatch.setenv("PATTERN_COMPLETED_SESSION", "2026-10-23")
    monkeypatch.setenv("GITHUB_RUN_ID", "20")
    service.write_model_audit([], [{"code": "000001", "reason": "看空"}], ["000001"])
    audit = json.loads(path.read_text())
    state = service.empty_state(); state["days"] = [day("2026-10-23", ("000001",))]
    state["days"][0].pop("selected")
    screen = {"as_of": "2026-10-23", "status": "complete", "selected": [{"code": "000001"}],
              "candidates": [{"code": "000001"}]}
    service.record_selection(state, screen, audit, None)
    assert state["days"][0]["published"] == [] and state["days"][0]["rejected"] == ["000001"]
    assert state["days"][0]["delivery_status"] == "no_recommendations"


def test_market_footer_is_dated_and_does_not_duplicate(tmp_path, monkeypatch):
    from src.core.market_review import _render_market_review_payload_markdown
    p = tmp_path / "review.json"
    p.write_text(json.dumps({"as_of": "2026-10-23", "summary_md": "## 选股自我复盘\n尚未到期"}))
    monkeypatch.setenv("SCREEN_LEARNING_REVIEW", str(p))
    payload = {"completed_session": "2026-10-23", "markdown_report": "大盘正文", "generated_at": "today"}
    assert _render_market_review_payload_markdown(payload).count("选股自我复盘") == 1
    payload["markdown_report"] += "\n## 选股自我复盘\n尚未到期"
    assert _render_market_review_payload_markdown(payload).count("选股自我复盘") == 1
    p.write_text(json.dumps({"as_of": "2026-10-22", "summary_md": "wrong-date-footer"}))
    assert "wrong-date-footer" not in _render_market_review_payload_markdown(payload)


def long_history(end="2026-10-23"):
    days = [day(d, ("000001", "000002", "000003")) for d in core.sessions("2026-01-01", end)[-120:]]
    for item in days:
        item["rows"]["000001"][:5] = [9.9, 10., 9.95, 10.2, 9.8]
        item["rows"]["000002"][:5] = [10.3, 10., 10.1, 10.3, 9.9]
        item["rows"]["000003"][:5] = [9.9, 10., 9.95, 10.2, 9.8]
    return days


def test_real_train_validate_holdout_can_promote_and_purges_five_day_label_overlap():
    days = long_history()
    maps = {i: core.features(days, i) for i in range(len(days))}
    result = core.select_policy(days, {"rule": "baseline"}, maps)
    assert result["status"] == "promoted" and result["rule"] in core.RULES
    assert result["validation_days"] == result["holdout_days"] == 25
    assert len(core.sessions(result["train_end"], result["validation_start"])) >= 7
    assert result["promoted_at"] == days[-1]["as_of"]


def test_training_success_is_not_enough_when_future_holdout_stops_improving():
    days = long_history()
    for item in days[-30:]:
        item["rows"]["000002"][:5] = [9.8, 10., 9.7, 9.8, 9.6]
    maps = {i: core.features(days, i) for i in range(len(days))}
    result = core.select_policy(days, {"rule": "baseline"}, maps)
    assert result["rule"] == "baseline" and result["status"] == "validation_failed"


def test_degraded_active_policy_rolls_back_and_does_not_reactivate_immediately():
    days = long_history()
    for item in days[-40:]:
        item["rows"]["000002"][:5] = [9.8, 10., 9.7, 9.8, 9.6]
    maps = {i: core.features(days, i) for i in range(len(days))}
    active = {"rule": "close_strong", "promoted_at": days[-45]["as_of"],
              "hard_filter_digest": core.HARD_FILTER_DIGEST, "ruleset_version": core.RULESET_VERSION}
    result = core.select_policy(days, active, maps)
    assert result["rule"] == "baseline" and result["status"] == "rolled_back"


def test_weekly_review_and_twenty_session_cooldown_limit_repeated_tuning():
    days = long_history("2026-10-22")
    assert core.select_policy(days, {"rule": "baseline"}, {})["status"] == "waiting_weekly_review"
    days = long_history()
    active = {"rule": "baseline", "reviewed_at": days[-10]["as_of"]}
    assert core.select_policy(days, active, {})["status"] == "cooldown"


def test_only_real_acknowledgements_become_published_and_retry_enrichment_is_monotonic(tmp_path, monkeypatch):
    from src.core import daily_delivery as delivery
    identity = {"repository": "owner/repo", "workflow_id": 123,
                "workflow_path": ".github/workflows/00-daily-analysis.yml", "branch": "main",
                "run_id": "20", "run_attempt": 1, "event_name": "workflow_dispatch", "as_of": "2026-10-23",
                "target_fingerprint": delivery.feishu_target_fingerprint("https://example.test/noncredential")}
    _, path = delivery.create_delivery_files(tmp_path, identity, delivery.COMPONENTS)
    monkeypatch.setenv("DAILY_DELIVERY_FILE", str(path))
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    delivery._GUARDS.clear()
    state = service.empty_state(); state["days"] = [day("2026-10-23")]
    state["days"][0].pop("selected")
    screen = {"as_of": "2026-10-23", "status": "complete", "profile": "bottom_reversal",
              "selected": [{"code": "000001"}, {"code": "000002"}],
              "candidates": [{"code": "000001"}, {"code": "000002"}]}
    audit = {"as_of": "2026-10-23", "run_id": "20", "input_codes": ["000001", "000002"],
             "accepted": ["000001"], "rejected": ["000002"]}
    service.record_selection(state, screen, audit, json.loads(path.read_text()))
    assert state["days"][0]["published"] == []
    def accepted():
        delivery.record_daily_post("ack")
        return True
    assert delivery.run_daily_delivery(identity["target_fingerprint"], ("stocks",), accepted)
    delivery.finish_daily_components("stocks")
    service.record_selection(state, screen, audit, json.loads(path.read_text()))
    assert state["days"][0]["published"] == ["000001"]
    service.record_selection(state, screen, dict(audit, accepted=["000002"]), None)
    assert state["days"][0]["published"] == ["000001"]


def test_bad_receipt_is_not_delivery_evidence():
    state = service.empty_state(); state["days"] = [day("2026-10-23")]
    state["days"][0].pop("selected")
    screen = {"as_of": "2026-10-23", "status": "complete", "selected": [{"code": "000001"}],
              "candidates": [{"code": "000001"}]}
    audit = {"as_of": "2026-10-23", "run_id": "20", "input_codes": ["000001"], "accepted": ["000001"]}
    service.record_selection(state, screen, audit, {"identity": {"as_of": "2026-10-23", "run_id": "20"},
                                                  "components": {"stocks": {"status": "acknowledged"}}})
    assert state["days"][0]["published"] == []


@pytest.mark.parametrize("payload", [[], "bad", {"as_of": "2026-10-23", "summary_md": []}])
def test_malformed_learning_report_cannot_break_market_notifications(tmp_path, monkeypatch, payload):
    path = tmp_path / "review.json"; path.write_text(json.dumps(payload))
    monkeypatch.setenv("SCREEN_LEARNING_REVIEW", str(path))
    assert service.notification_summary("2026-10-23") == ""


def test_changed_hard_rules_or_missing_previous_session_suspend_automatic_ranking():
    days = long_history()
    active = {"rule": "close_strong", "hard_filter_digest": "old-parameters", "ruleset_version": core.RULESET_VERSION}
    assert core.select_policy(days, active, {})["status"] == "ruleset_changed"
    active["hard_filter_digest"] = core.HARD_FILTER_DIGEST
    days.pop(-2)
    result = core.select_policy(days, active, {})
    assert result["rule"] == "baseline" and result["status"] == "history_gap"
