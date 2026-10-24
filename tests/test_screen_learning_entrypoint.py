"""跨Actions恢复档案的来源核验和learning-only的不发送约束。"""
import copy
import gzip
import io
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile

import pytest
import yaml

from scripts import screen_learning as entry
from src.core import daily_delivery
from src.services.screen_learning_service import empty_state
from tests.test_screen_learning import day


def archive(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as target:
        for name, payload in files.items():
            target.writestr(name, payload)
    return output.getvalue()


def run(run_id=10):
    return {"id": run_id, "repository": {"full_name": "owner/repo", "id": 1}, "workflow_id": 123,
            "path": entry.WORKFLOW_PATH, "head_branch": "main", "event": "workflow_dispatch",
            "head_sha": "abc", "created_at": "2026-10-23T14:00:00Z"}


class Client:
    repository = "owner/repo"

    def __init__(self, raw, name="screen-learning-state-9-1"):
        self.raw = raw
        self.artifact = {"id": 99, "name": name, "created_at": "2026-10-23T15:00:00Z",
                         "workflow_run": {"id": 9, "head_branch": "main", "head_sha": "abc", "repository_id": 1}}
        self.history_run = run(9)

    def get_json(self, path, params=None):
        return {"artifacts": [self.artifact]} if path == "/actions/artifacts" else self.history_run

    def download(self, artifact):
        return self.raw


def test_durable_history_is_restored_from_expected_main_workflow_only():
    state = empty_state(); state["days"] = [day("2026-10-23")]
    state.update(repository="owner/repo", workflow_path=entry.WORKFLOW_PATH)
    client = Client(archive({"state.json.gz": gzip.compress(json.dumps(state).encode())}))
    assert entry.restore_remote(client, run()) == state


@pytest.mark.parametrize("fault", ["branch", "workflow", "sha", "repository", "name", "corrupt", "extra_entry"])
def test_untrusted_or_corrupt_learning_artifact_cannot_activate_a_policy(fault):
    state = empty_state(); state.update(repository="owner/repo", workflow_path=entry.WORKFLOW_PATH)
    client = Client(archive({"state.json.gz": gzip.compress(json.dumps(state).encode())}))
    if fault == "branch": client.history_run["head_branch"] = "attacker-branch"
    elif fault == "workflow": client.history_run["path"] = "another.yml"
    elif fault == "sha": client.artifact["workflow_run"]["head_sha"] = "wrong"
    elif fault == "repository": client.history_run["repository"]["full_name"] = "another/repo"
    elif fault == "name": client.artifact["name"] = "screen-learning-state-8-1"
    elif fault == "corrupt": client.raw = archive({"state.json.gz": b"broken"})
    else: client.raw = archive({"state.json.gz": b"x", "unexpected": b"x"})
    with pytest.raises(Exception): entry.restore_remote(client, run())


def old_report(tmp_path):
    identity = {"repository": "owner/repo", "workflow_id": 123, "workflow_path": entry.WORKFLOW_PATH,
                "branch": "main", "run_id": "9", "run_attempt": 1, "event_name": "workflow_dispatch",
                "as_of": "2026-10-23", "target_fingerprint": daily_delivery.feishu_target_fingerprint("https://example.test/x")}
    _, path = daily_delivery.create_delivery_files(tmp_path, identity, daily_delivery.COMPONENTS)
    receipt = json.loads(path.read_text())
    receipt["components"]["stocks"].update(status="acknowledged", finalized=True, calls=1,
                                            posts={"ack": 1, "unknown": 0, "rejected": 0})
    screen = {"as_of": "2026-10-23", "status": "complete", "profile": "bottom_reversal",
              "selected": [{"code": "000001"}, {"code": "000002"}],
              "candidates": [{"code": "000001"}, {"code": "000002"}]}
    files = {"reports/screen.json": json.dumps(screen), "reports/receipt.json": json.dumps(receipt),
             "reports/report_20261023.md": "## ⚪ [样本](https://xueqiu.com/S/SZ000001) (000001)\n\n正文",
             "logs/stock_analysis_20261023.log": "[000002] 不进入每日推荐：模型趋势看空"}
    return Client(archive(files), "analysis-reports-62")


def test_verified_previous_close_can_seed_real_published_and_rejected_records(tmp_path, monkeypatch):
    client = old_report(tmp_path)
    monkeypatch.setattr(entry, "get_market_now", lambda market: datetime(2026, 10, 24, 21, tzinfo=ZoneInfo("Asia/Shanghai")))
    state = empty_state(); state["days"] = [day("2026-10-23")]
    for key in ("selected", "published", "rejected", "baseline_pool"):
        state["days"][0].pop(key)
    entry.bootstrap_same_close(client, run(), state)
    assert state["days"][0]["published"] == ["000001"] and state["days"][0]["rejected"] == ["000002"]
    assert state["days"][0]["selection_origin"]["run_id"] == 9


def test_no_reconstruction_of_old_features_after_next_market_open(tmp_path, monkeypatch):
    client = old_report(tmp_path)
    monkeypatch.setattr(entry, "get_market_now", lambda market: datetime(2026, 10, 26, 10, tzinfo=ZoneInfo("Asia/Shanghai")))
    state = empty_state(); state["days"] = [day("2026-10-23")]
    state["days"][0].pop("selected")
    entry.bootstrap_same_close(client, run(), state)
    assert "selected" not in state["days"][0]


def test_unknown_next_session_cannot_seed_old_report(tmp_path, monkeypatch):
    client = old_report(tmp_path)
    monkeypatch.setattr(entry, "following_sessions", lambda start, count: [])
    state = empty_state(); state["days"] = [day("2026-10-23")]
    state["days"][0].pop("selected")
    entry.bootstrap_same_close(client, run(), state)
    assert "selected" not in state["days"][0]


def test_learning_only_workflow_cannot_call_models_or_send_messages():
    workflow = yaml.safe_load(Path(".github/workflows/00-daily-analysis.yml").read_text())
    steps = workflow["jobs"]["analyze"]["steps"]
    analyze = next(s for s in steps if s["name"] == "执行股票分析")
    prepare = next(s for s in steps if s.get("id") == "delivery")
    screen = next(s for s in steps if s.get("id") == "patterns")
    assert analyze["if"] == "github.event.inputs.mode != 'learning-only'"
    assert "!= 'learning-only'" in prepare["if"] and "!= 'learning-only'" in screen["if"]
    assert any(s.get("id") == "learning_upload" and s["with"]["retention-days"] == 30 for s in steps)
    assert next(s for s in steps if s.get("id") == "learning_prepare")["continue-on-error"] is True
