"""GitHub evidence fixtures exercise migration, missing receipts and read failures."""
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from scripts import daily_delivery as cli
from src.core.daily_delivery import COMPONENTS, create_delivery_files, feishu_target_fingerprint


NOW = datetime(2026, 10, 21, 20, 42, tzinfo=ZoneInfo("Asia/Shanghai"))


def zipped(name, payload):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr(name, json.dumps(payload))
    return result.getvalue()


@pytest.fixture
def identity():
    return {"repository": "owner/repo", "workflow_id": 123, "workflow_path": cli.WORKFLOW_PATH,
            "branch": "main", "run_id": "20", "run_attempt": 1, "event_name": "workflow_dispatch",
            "as_of": "2026-10-21", "target_fingerprint": feishu_target_fingerprint("https://example.test/noncredential")}


def run_for(identity, run_id=10, **overrides):
    return {"id": run_id, "repository": {"full_name": identity["repository"], "id": 1},
            "workflow_id": identity["workflow_id"], "path": cli.WORKFLOW_PATH, "head_branch": "main",
            "head_sha": "abc", "event": "workflow_dispatch", "run_attempt": 1, **overrides}


class FakeEvidence:
    def __init__(self, identity):
        self.identity = identity
        self.runs = {}
        self.artifacts = {}
        self.bodies = {}
        self.jobs = {}

    def get_json(self, path):
        return self.runs[int(path.split("/")[-1])]

    def pages(self, path, key, params=None):
        if key == "workflow_runs":
            return list(self.runs.values())
        parts = path.split("/")
        if key == "artifacts":
            return self.artifacts.get(int(parts[3]), [])
        if key == "jobs":
            return self.jobs.get((int(parts[3]), int(parts[5])), [])
        raise AssertionError(path)

    def download(self, artifact):
        return self.bodies[artifact["id"]]

    def add_artifact(self, run, name, member, payload):
        artifact_id = len(self.bodies) + 1
        raw = zipped(member, payload)
        artifact = {"id": artifact_id, "name": name, "size_in_bytes": len(raw), "expired": False,
                    "workflow_run": {"id": run["id"], "head_branch": "main", "head_sha": run["head_sha"], "repository_id": 1}}
        self.artifacts.setdefault(run["id"], []).append(artifact)
        self.bodies[artifact_id] = raw
        return artifact


def add_modern(client, tmp_path, identity, states=None, *, run_id=10, attempt=1, terminal=True,
               prior=None, as_of=None, target=None, conclusion="success"):
    run = run_for(identity, run_id, run_attempt=attempt, conclusion=conclusion)
    client.runs[run_id] = run
    old = {**identity, "run_id": str(run_id), "run_attempt": attempt,
           "as_of": as_of or identity["as_of"], "target_fingerprint": target or identity["target_fingerprint"]}
    intent, receipt = create_delivery_files(tmp_path / f"{run_id}-{attempt}", old, COMPONENTS, prior)
    client.add_artifact(run, cli.artifact_name("intent", old), "intent.json", json.loads(intent.read_text()))
    if terminal:
        payload = json.loads(receipt.read_text())
        for state in payload["components"].values():
            state["finalized"] = True
        for component, status in (states or {}).items():
            state = payload["components"][component]
            state["status"] = status
            state["calls"] = 1
            state["failed_calls"] = int(status != "acknowledged")
            state["posts"] = {"ack": int(status in {"acknowledged", "partial"}),
                              "rejected": int(status in {"rejected", "partial"}), "unknown": int(status == "unknown")}
        client.add_artifact(run, cli.artifact_name("receipt", old), "receipt.json", payload)
    return client


def test_failed_run_actual_ack_suppresses_only_its_component(tmp_path, identity):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, {"stocks": "acknowledged"}, conclusion="failure")
    assert cli.collect_prior(client, identity) == {"stocks": "delivered", "market": "allow", "document": "allow"}


@pytest.mark.parametrize("status", ["unknown", "partial"])
def test_partial_or_unknown_cannot_be_auto_replayed(tmp_path, identity, status):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, {"stocks": status})
    assert cli.collect_prior(client, identity)["stocks"] == "blocked"


def test_intent_without_terminal_is_unknown_even_if_job_success(tmp_path, identity):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, terminal=False)
    assert set(cli.collect_prior(client, identity).values()) == {"blocked"}


def test_inherited_skip_without_original_ack_is_not_proof_of_unsent(tmp_path, identity):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, prior={"stocks": "delivered"})
    assert cli.collect_prior(client, identity)["stocks"] == "blocked"
    add_modern(client, tmp_path, identity, {"stocks": "acknowledged"}, run_id=9)
    assert cli.collect_prior(client, identity)["stocks"] == "delivered"


def test_explicit_rejections_are_safe_to_retry(tmp_path, identity):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, {"stocks": "rejected"})
    assert cli.collect_prior(client, identity)["stocks"] == "allow"


def test_unfinished_ack_is_unknown(tmp_path, identity):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, {"stocks": "acknowledged"})
    artifact = client.artifacts[10][-1]
    payload = cli.artifact_json(client.bodies[artifact["id"]], "receipt.json")
    payload["components"]["stocks"]["finalized"] = False
    client.bodies[artifact["id"]] = zipped("receipt.json", payload)
    assert cli.collect_prior(client, identity)["stocks"] == "blocked"


def test_other_date_or_destination_does_not_block(tmp_path, identity):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, {"stocks": "acknowledged"}, as_of="2026-10-20")
    add_modern(client, tmp_path, identity, {"stocks": "acknowledged"}, run_id=9, target="0" * 64)
    assert set(cli.collect_prior(client, identity).values()) == {"allow"}


def test_previous_attempt_of_same_run_is_checked(tmp_path, identity):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, {"stocks": "acknowledged"}, run_id=20)
    identity["run_attempt"] = 2
    client.runs[20]["run_attempt"] = 2
    assert cli.collect_prior(client, identity)["stocks"] == "delivered"


@pytest.mark.parametrize("old_day,expected", [("2026-10-20", "allow"), ("2026-10-21", "blocked")])
def test_legacy_uses_screen_date_instead_of_run_creation_date(tmp_path, identity, old_day, expected):
    client = FakeEvidence(identity)
    run = run_for(identity)
    client.runs[10] = run
    client.jobs[(10, 1)] = [{"steps": [{"name": "执行股票分析", "status": "completed", "conclusion": "success"}]}]
    client.add_artifact(run, "analysis-reports-56", "reports/pattern_screen/screen.json", {"as_of": old_day})
    assert cli.collect_prior(client, identity)["stocks"] == expected


def test_legacy_attempt_without_artifact_is_unknown(identity):
    client = FakeEvidence(identity)
    client.runs[10] = run_for(identity, conclusion="failure")
    client.jobs[(10, 1)] = [{"steps": [{"name": "执行股票分析", "status": "completed", "conclusion": "failure"}]}]
    assert set(cli.collect_prior(client, identity).values()) == {"blocked"}


def test_legacy_skipped_analysis_is_not_a_send(identity):
    client = FakeEvidence(identity)
    client.runs[10] = run_for(identity, conclusion="success")
    client.jobs[(10, 1)] = [{"steps": [{"name": "执行股票分析", "status": "completed", "conclusion": "skipped"}]}]
    assert set(cli.collect_prior(client, identity).values()) == {"allow"}


def test_legacy_market_only_without_screen_is_unknown_not_an_unrecoverable_parse_error(identity):
    client = FakeEvidence(identity)
    run = run_for(identity)
    client.runs[10] = run
    client.jobs[(10, 1)] = [{"steps": [{"name": "执行股票分析", "status": "completed", "conclusion": "success"}]}]
    client.add_artifact(run, "analysis-reports-56", "reports/market_review.json", {"report": "old market"})
    assert cli.collect_prior(client, identity)["market"] == "blocked"


def test_ambiguous_legacy_rerun_is_not_dated_from_only_first_attempt_artifact(identity):
    client = FakeEvidence(identity)
    run = run_for(identity, run_attempt=2)
    client.runs[10] = run
    client.jobs[(10, 2)] = [{"steps": [{"name": "执行股票分析", "status": "completed", "conclusion": "success"}]}]
    client.add_artifact(run, "analysis-reports-56", "screen.json", {"as_of": "2026-10-20"})
    assert cli.collect_prior(client, identity)["stocks"] == "blocked"


def test_untrusted_artifact_provenance_fails_closed(tmp_path, identity):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, {"stocks": "acknowledged"})
    client.artifacts[10][0]["workflow_run"]["head_sha"] = "different"
    with pytest.raises(cli.EvidenceError):
        cli.collect_prior(client, identity)


def test_json_status_without_matching_post_ack_is_rejected(tmp_path, identity):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, {"stocks": "acknowledged"})
    artifact = client.artifacts[10][-1]
    data = cli.artifact_json(client.bodies[artifact["id"]], "receipt.json")
    data["components"]["stocks"]["posts"]["ack"] = 0
    client.bodies[artifact["id"]] = zipped("receipt.json", data)
    with pytest.raises(cli.EvidenceError):
        cli.collect_prior(client, identity)


def test_corrupt_or_path_traversal_zip_is_rejected():
    for raw in (b"not-a-zip", zipped("../receipt.json", {})):
        with pytest.raises(cli.EvidenceError):
            cli.artifact_json(raw, "receipt.json")


@pytest.fixture
def github_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    values = {"GITHUB_REPOSITORY": "owner/repo", "GITHUB_REF": "refs/heads/main",
              "GITHUB_WORKFLOW_REF": "owner/repo/" + cli.WORKFLOW_PATH + "@refs/heads/main",
              "GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_RUN_ID": "20", "GITHUB_RUN_ATTEMPT": "1",
              "GITHUB_ENV": str(tmp_path / "env"), "GITHUB_OUTPUT": str(tmp_path / "output"),
              "FEISHU_WEBHOOK_URL": "https://example.test/noncredential"}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    for key in ("FEISHU_APP_ID", "FEISHU_CHAT_ID", "DAILY_DELIVERY_FILE"):
        monkeypatch.delenv(key, raising=False)
    return values


def test_prepare_sets_workflow_contract_without_sending(github_env, identity):
    client = FakeEvidence(identity)
    client.runs[20] = run_for(identity, 20)
    args = SimpleNamespace(as_of="2026-10-21", mode="stocks-only", allow_resend="false")
    assert cli.prepare(args, client=client, now=NOW) == 0
    data = json.loads(Path("reports/daily_delivery/receipt.json").read_text())
    assert set(data["components"]) == {"stocks", "document"}
    assert "DAILY_DELIVERY_FILE=" in Path(github_env["GITHUB_ENV"]).read_text()
    output = Path(github_env["GITHUB_OUTPUT"]).read_text()
    assert "intent_name=daily-delivery-intent-2026-10-21-20-1" in output and "enabled=true" in output


def test_no_feishu_keeps_other_channels_and_does_not_read_history(github_env, monkeypatch):
    monkeypatch.delenv("FEISHU_WEBHOOK_URL")
    assert cli.prepare(SimpleNamespace(as_of="", mode="full", allow_resend="false"),
                       client=object(), now=NOW.replace(hour=1)) == 0
    assert Path(github_env["GITHUB_OUTPUT"]).read_text() == "enabled=false\n"
    assert not Path(github_env["GITHUB_ENV"]).exists()
    assert not Path("reports/daily_delivery").exists()


@pytest.mark.parametrize("mode", ["full", "market-only"])
def test_overnight_without_verified_session_cannot_guess_new_delivery_day(github_env, mode):
    with pytest.raises(cli.EvidenceError, match="verified completed-session"):
        cli.prepare(SimpleNamespace(as_of="", mode=mode, allow_resend="false"),
                    client=object(), now=NOW.replace(hour=8, minute=59, second=59))
    assert not Path(github_env["GITHUB_ENV"]).exists()
    assert not Path("reports/daily_delivery").exists()


def test_daytime_unscreened_mode_preserves_natural_date_contract(github_env, identity):
    client = FakeEvidence(identity)
    client.runs[20] = run_for(identity, 20)
    assert cli.prepare(SimpleNamespace(as_of="", mode="market-only", allow_resend="false"),
                       client=client, now=NOW.replace(hour=9, minute=0, second=0)) == 0
    data = json.loads(Path("reports/daily_delivery/receipt.json").read_text())
    assert data["identity"]["as_of"] == "2026-10-21"


def test_overnight_schedule_retains_explicit_verified_screen_day(github_env, identity, monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    client = FakeEvidence(identity)
    client.runs[20] = run_for(identity, 20, event="schedule")
    assert cli.prepare(SimpleNamespace(as_of="2026-10-21", mode="full", allow_resend="false"),
                       client=client, now=NOW.replace(day=22, hour=1)) == 0
    data = json.loads(Path("reports/daily_delivery/receipt.json").read_text())
    assert data["identity"]["as_of"] == "2026-10-21"


@pytest.mark.parametrize("field,value", [
    ("GITHUB_REF", "refs/heads/feature"), ("GITHUB_EVENT_NAME", "pull_request"),
    ("GITHUB_WORKFLOW_REF", "owner/repo/.github/workflows/another.yml@refs/heads/main"),
])
def test_prepare_refuses_wrong_workflow_without_network(github_env, monkeypatch, field, value):
    monkeypatch.setenv(field, value)
    with pytest.raises(cli.EvidenceError):
        cli.prepare(SimpleNamespace(as_of="2026-10-21", mode="full", allow_resend="false"), client=object(), now=NOW)


@pytest.mark.parametrize("as_of", ["20261021", "2026-10-19", "2026-10-22", "2026-10-20"])
def test_prepare_date_limits_block_stale_manual_or_future(github_env, as_of):
    with pytest.raises(cli.EvidenceError):
        cli.prepare(SimpleNamespace(as_of=as_of, mode="full", allow_resend="false"), client=object(), now=NOW)


def test_resend_requires_new_dispatch_and_preserves_prior_files(github_env, identity, tmp_path, monkeypatch):
    client = add_modern(FakeEvidence(identity), tmp_path, identity, {"stocks": "acknowledged"})
    client.runs[20] = run_for(identity, 20)
    args = SimpleNamespace(as_of="2026-10-21", mode="full", allow_resend="true")
    assert cli.prepare(args, client=client, now=NOW) == 0
    data = json.loads(Path("reports/daily_delivery/receipt.json").read_text())
    assert data["components"]["stocks"]["prior"] == "allow" and data["explicit_resend"] is True
    assert len(client.artifacts[10]) == 2
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    with pytest.raises(cli.EvidenceError):
        cli.prepare(args, client=client, now=NOW)


class Response:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    def iter_content(self, chunk_size):
        yield self.content

    def close(self):
        pass


def test_artifact_redirect_never_receives_authorization(monkeypatch):
    client = cli.GitHubEvidence("owner/repo", "fake-test-token")
    calls = []

    def authenticated(*args, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer fake-test-token"
        return Response(302, headers={"Location": "https://example.blob.core.windows.net/artifact?fake-signature=1"})

    def anonymous_get(url, **kwargs):
        calls.append(url)
        assert "Authorization" not in kwargs["headers"]
        assert kwargs["allow_redirects"] is False
        assert anonymous.trust_env is False
        return Response(200, b"zip-data")

    anonymous = SimpleNamespace(trust_env=True, get=anonymous_get, close=lambda: None)
    monkeypatch.setattr(client.session, "get", authenticated)
    monkeypatch.setattr(cli.requests, "Session", lambda: anonymous)
    assert client.download({"id": 1, "expired": False, "size_in_bytes": 10}) == b"zip-data"
    assert len(calls) == 1


def test_api_failure_and_pagination_do_not_become_empty_history(monkeypatch):
    client = cli.GitHubEvidence("owner/repo", "fake-test-token")
    monkeypatch.setattr(client.session, "get", lambda *a, **kw: Response(403))
    with pytest.raises(cli.EvidenceError):
        client.get_json("/actions/runs")
    monkeypatch.setattr(client, "get_json", lambda *a, **kw: {"total_count": 2, "artifacts": [{"id": 1}]})
    with pytest.raises(cli.EvidenceError):
        client.pages("/actions/artifacts", "artifacts")
