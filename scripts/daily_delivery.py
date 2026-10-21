#!/usr/bin/env python3
"""Prepare evidence-backed daily delivery, then finalize its local receipt.

No messages or repository writes are performed here. The workflow uploads the
intent before main.py and uploads the receipt afterwards, using Actions' own
artifact runtime. Missing or unreadable sending evidence is never 'not sent'.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.daily_delivery import (
    COMPONENTS, create_delivery_files, feishu_target_fingerprint, finalize_daily_delivery, validate_receipt,
)

WORKFLOW_PATH = ".github/workflows/00-daily-analysis.yml"
MAX_PAGES = 10
MAX_ATTEMPTS = 20
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_EXPANDED_BYTES = 80 * 1024 * 1024
MAX_ARTIFACT_JSON_BYTES = 256 * 1024


class EvidenceError(RuntimeError):
    """A deliberately credential-free error that prevents automatic sending."""


def _read_bounded(response, limit):
    result = bytearray()
    for chunk in response.iter_content(chunk_size=65536):
        result.extend(chunk)
        if len(result) > limit:
            raise EvidenceError("GitHub evidence exceeds its size limit")
    return bytes(result)


class GitHubEvidence:
    def __init__(self, repository, token, api_url="https://api.github.com"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise EvidenceError("Invalid GitHub repository")
        if api_url.rstrip("/") != "https://api.github.com" or not token:
            raise EvidenceError("GitHub read access is required")
        self.repository = repository
        self.base = api_url.rstrip("/") + "/repos/" + repository
        self.headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28"}
        self.session = requests.Session()

    def get_json(self, path, params=None):
        try:
            response = self.session.get(self.base + path, params=params, headers=self.headers,
                                        timeout=(5, 20), allow_redirects=False, stream=True)
            if response.status_code != 200:
                raise EvidenceError("GitHub evidence read failed (HTTP %s)" % response.status_code)
            raw = _read_bounded(response, MAX_JSON_BYTES)
            value = json.loads(raw)
        except EvidenceError:
            raise
        except Exception as exc:
            raise EvidenceError("GitHub evidence read failed (%s)" % type(exc).__name__) from None
        finally:
            if "response" in locals():
                response.close()
        if not isinstance(value, dict):
            raise EvidenceError("GitHub returned an unexpected evidence document")
        return value

    def pages(self, path, key, params=None):
        results = []
        for page in range(1, MAX_PAGES + 1):
            payload = self.get_json(path, {**(params or {}), "per_page": 100, "page": page})
            items = payload.get(key)
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise EvidenceError("Invalid GitHub evidence collection")
            results.extend(items)
            if len(items) < 100:
                if type(payload.get("total_count")) is int and payload["total_count"] > len(results):
                    raise EvidenceError("GitHub evidence collection was incomplete")
                return results
        raise EvidenceError("GitHub evidence pagination limit reached")

    def download(self, artifact):
        artifact_id = artifact.get("id")
        size = artifact.get("size_in_bytes")
        if (type(artifact_id) is not int or artifact_id < 1 or artifact.get("expired")
                or type(size) is not int or size < 1 or size > MAX_ARCHIVE_BYTES):
            raise EvidenceError("Artifact is missing, expired or too large")
        try:
            response = self.session.get(self.base + f"/actions/artifacts/{artifact_id}/zip", headers=self.headers,
                                        timeout=(5, 20), allow_redirects=False, stream=True)
            if response.status_code != 302:
                raise EvidenceError("Artifact download redirect is unavailable")
            location = response.headers.get("Location", "")
            parsed = urlparse(location)
            host = (parsed.hostname or "").lower()
            if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in {None, 443}
                    or not any(host.endswith(suffix) for suffix in {
                        ".blob.core.windows.net", ".githubusercontent.com", ".amazonaws.com"})):
                raise EvidenceError("Artifact download destination is not trusted")
            response.close()
            # A separate request is essential: signed blob URLs must never receive GITHUB_TOKEN.
            anonymous = requests.Session()
            anonymous.trust_env = False  # Also prevents .netrc auth on the blob host.
            response = anonymous.get(location, headers={"Accept": "application/zip"}, timeout=(5, 20),
                                     allow_redirects=False, stream=True)
            if response.status_code != 200:
                raise EvidenceError("Artifact download failed")
            return _read_bounded(response, MAX_ARCHIVE_BYTES)
        except EvidenceError:
            raise
        except Exception as exc:
            raise EvidenceError("Artifact download failed (%s)" % type(exc).__name__) from None
        finally:
            if "response" in locals():
                response.close()
            if "anonymous" in locals():
                anonymous.close()


def artifact_json(raw, basename, *, required=True):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) > 500 or sum(entry.file_size for entry in entries) > MAX_EXPANDED_BYTES:
                raise EvidenceError("Artifact expansion limit exceeded")
            for entry in entries:
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                    raise EvidenceError("Invalid artifact member path")
            matches = [entry for entry in entries if PurePosixPath(entry.filename).name == basename]
            if not matches and not required:
                return None
            if len(matches) != 1:
                raise EvidenceError("Artifact does not contain exactly one " + basename)
            if matches[0].file_size > (MAX_JSON_BYTES if basename == "screen.json" else MAX_ARTIFACT_JSON_BYTES):
                raise EvidenceError("Artifact JSON exceeds its size limit")
            payload = json.loads(archive.read(matches[0]))
            if not isinstance(payload, dict):
                raise EvidenceError("Invalid artifact JSON")
            return payload
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError("Artifact cannot be verified (%s)" % type(exc).__name__) from None


def _same_repository(value, repository):
    return isinstance(value, dict) and str(value.get("full_name", "")).casefold() == repository.casefold()


def _check_run(run, identity):
    if (not _same_repository(run.get("repository"), identity["repository"])
            or run.get("workflow_id") != identity["workflow_id"] or run.get("path") != WORKFLOW_PATH
            or run.get("head_branch") != "main" or run.get("event") not in {"schedule", "workflow_dispatch"}):
        raise EvidenceError("Run does not belong to the expected daily workflow")


def _verified_artifact(artifact, run, identity):
    metadata = artifact.get("workflow_run")
    if (not isinstance(metadata, dict) or metadata.get("id") != run["id"]
            or metadata.get("head_branch") != "main"
            or metadata.get("head_sha") != run.get("head_sha")):
        raise EvidenceError("Artifact provenance cannot be verified")
    repository_id = run.get("repository", {}).get("id")
    if metadata.get("repository_id") is not None and metadata["repository_id"] != repository_id:
        raise EvidenceError("Artifact belongs to another repository")


def _validate_artifact_payload(payload, kind, run, attempt, identity):
    try:
        validate_receipt(payload, kind=kind)
    except (ValueError, TypeError, KeyError):
        raise EvidenceError("Daily delivery artifact schema is invalid") from None
    found = payload["identity"]
    for key in ("repository", "workflow_id", "workflow_path", "branch"):
        if found[key] != identity[key]:
            raise EvidenceError("Daily delivery artifact identity differs from its workflow")
    if (str(found["run_id"]) != str(run["id"]) or found["run_attempt"] != attempt
            or found["event_name"] != run["event"]):
        raise EvidenceError("Daily delivery artifact attempt cannot be verified")
    if kind == "intent" and any(state["status"] != "not_attempted" or any(state["posts"].values())
                                or state["active_call"] for state in payload["components"].values()):
        raise EvidenceError("Daily intent contains invalid sending evidence")


def _merge_prior(prior, component, state):
    if state == "delivered" or (state == "blocked" and prior[component] != "delivered"):
        prior[component] = state


def _component_evidence(state):
    if state["active_call"] or not state["finalized"]:
        return "blocked"
    if state["status"] == "acknowledged" and state["posts"]["ack"] and not state["posts"]["unknown"]:
        return "delivered"
    if (state["prior"] == "allow" and state["status"] in {"not_attempted", "rejected"}
            and not state["posts"]["ack"] and not state["posts"]["unknown"]):
        return "allow"
    return "blocked"


def _analysis_attempted(client, run, attempt):
    jobs = client.pages(f"/actions/runs/{run['id']}/attempts/{attempt}/jobs", "jobs")
    for job in jobs:
        for step in job.get("steps", []):
            if step.get("name") == "执行股票分析" and step.get("conclusion") != "skipped":
                return step.get("status") in {"in_progress", "completed"}
    return False


def collect_prior(client, identity):
    """Inspect all relevant attempts, including failed runs and legacy evidence."""
    prior = dict.fromkeys(COMPONENTS, "allow")
    as_of = date.fromisoformat(identity["as_of"])
    # Delayed runs may be created the next UTC/local date; artifact as_of decides identity.
    created = f"{(as_of - timedelta(days=1)).isoformat()}..{(as_of + timedelta(days=2)).isoformat()}"
    runs = client.pages(f"/actions/workflows/{identity['workflow_id']}/runs", "workflow_runs", {"created": created})
    seen = set()
    for summary in runs:
        run_id = summary.get("id")
        if type(run_id) is not int or run_id in seen:
            raise EvidenceError("Invalid or duplicate run metadata")
        seen.add(run_id)
        if summary.get("head_branch") != "main" or summary.get("event") not in {"schedule", "workflow_dispatch"}:
            continue
        run = client.get_json(f"/actions/runs/{run_id}")
        _check_run(run, identity)
        attempts = run.get("run_attempt")
        if type(attempts) is not int or not 1 <= attempts <= MAX_ATTEMPTS:
            raise EvidenceError("Run attempt evidence exceeds its bound")
        artifacts = client.pages(f"/actions/runs/{run_id}/artifacts", "artifacts")
        for attempt in range(1, attempts + 1):
            if str(run_id) == str(identity["run_id"]) and attempt >= identity["run_attempt"]:
                continue
            suffix = f"-{run_id}-{attempt}"
            modern = [item for item in artifacts if str(item.get("name", "")).startswith("daily-delivery-")
                      and str(item["name"]).endswith(suffix)]
            if modern:
                parsed = {}
                for artifact in modern:
                    _verified_artifact(artifact, run, identity)
                    kind = "intent" if artifact["name"].startswith("daily-delivery-intent-") else "receipt"
                    if not artifact["name"].startswith(f"daily-delivery-{kind}-") or kind in parsed:
                        raise EvidenceError("Ambiguous daily delivery artifact names")
                    payload = artifact_json(client.download(artifact), kind + ".json")
                    _validate_artifact_payload(payload, kind, run, attempt, identity)
                    if artifact["name"] != artifact_name(kind, payload["identity"]):
                        raise EvidenceError("Daily delivery artifact name is inconsistent")
                    parsed[kind] = payload
                source = parsed.get("intent") or parsed["receipt"]
                if source["identity"]["as_of"] != identity["as_of"]:
                    continue
                if source["identity"]["target_fingerprint"] != identity["target_fingerprint"]:
                    continue
                if ("intent" not in parsed or "receipt" not in parsed
                        or parsed["intent"]["identity"] != parsed["receipt"]["identity"]
                        or parsed["intent"]["components"].keys() != parsed["receipt"]["components"].keys()):
                    for component in source["components"]:
                        _merge_prior(prior, component, "blocked")
                    continue
                for component, state in parsed["receipt"]["components"].items():
                    _merge_prior(prior, component, _component_evidence(state))
                continue
            if not _analysis_attempted(client, run, attempt):
                continue
            # Old successful jobs are not proof of a successful webhook. Determine
            # which day they processed, then conservatively mark that day unknown.
            legacy = [item for item in artifacts if str(item.get("name", "")).startswith("analysis-reports-")]
            if legacy:
                same_day = False
                for artifact in legacy:
                    _verified_artifact(artifact, run, identity)
                    payload = artifact_json(client.download(artifact), "screen.json", required=False)
                    if payload is None:
                        # A valid legacy market-only archive has no screening
                        # JSON. Its delivery date cannot be disproved, so keep
                        # it unknown; explicit resend can still override this.
                        same_day = True
                        continue
                    try:
                        day = date.fromisoformat(payload["as_of"])
                        if day.isoformat() != payload["as_of"]:
                            raise ValueError
                    except (ValueError, TypeError, KeyError):
                        raise EvidenceError("Legacy report has no reliable market date") from None
                    same_day = same_day or day == as_of
                if not same_day and attempts == 1:
                    continue
            for component in COMPONENTS:
                _merge_prior(prior, component, "blocked")
    return prior


def artifact_name(kind, identity):
    return f"daily-delivery-{kind}-{identity['as_of']}-{identity['run_id']}-{identity['run_attempt']}"


def prepare(args, *, client=None, now=None):
    now = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    now = now.astimezone(ZoneInfo("Asia/Shanghai"))
    env = os.environ
    webhook, app_id, chat_id = (env.get(key, "").strip() for key in ("FEISHU_WEBHOOK_URL", "FEISHU_APP_ID", "FEISHU_CHAT_ID"))
    if not webhook and not (app_id and chat_id):
        if not env.get("GITHUB_OUTPUT"):
            raise EvidenceError("GitHub workflow output file is required")
        with Path(env["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write("enabled=false\n")
        print("No daily Feishu destination; other notification channels retain their normal behavior")
        return 0
    repository = env.get("GITHUB_REPOSITORY", "")
    event = env.get("GITHUB_EVENT_NAME", "")
    if event not in {"schedule", "workflow_dispatch"}:
        raise EvidenceError("Only the daily schedule or manual workflow may prepare delivery")
    ref = env.get("GITHUB_REF", "")
    workflow_ref = env.get("GITHUB_WORKFLOW_REF", "")
    if ref != "refs/heads/main" or workflow_ref != f"{repository}/{WORKFLOW_PATH}@refs/heads/main":
        raise EvidenceError("Daily delivery is restricted to the main daily workflow")
    if not args.as_of and now.hour < 9:
        raise EvidenceError("Overnight daily delivery requires a verified completed-session as_of")
    try:
        run_id, attempt = int(env.get("GITHUB_RUN_ID", "")), int(env.get("GITHUB_RUN_ATTEMPT", ""))
        if run_id < 1 or attempt < 1:
            raise ValueError
        as_of = date.fromisoformat(args.as_of or now.date().isoformat())
        if args.as_of and as_of.isoformat() != args.as_of:
            raise ValueError
    except (ValueError, TypeError):
        raise EvidenceError("Invalid daily delivery run identity or market date") from None
    if as_of not in {now.date(), now.date() - timedelta(days=1)}:
        raise EvidenceError("Daily delivery date is outside the completed-session window")
    if as_of != now.date():
        from src.core.screening_session import validate_completed_session
        try:
            validate_completed_session(as_of.isoformat(), now, event)
        except ValueError:
            raise EvidenceError("Previous-day delivery requires the validated overnight schedule window") from None
    if args.allow_resend not in {"true", "false"}:
        raise EvidenceError("allow-resend must be true or false")
    resend = args.allow_resend == "true"
    if resend and not (event == "workflow_dispatch" and attempt == 1):
        raise EvidenceError("Explicit resend is accepted only on a new manual workflow dispatch")
    client = client or GitHubEvidence(repository, env.get("GITHUB_TOKEN", ""), env.get("GITHUB_API_URL", "https://api.github.com"))
    run = client.get_json(f"/actions/runs/{run_id}")
    identity = {"repository": repository, "workflow_id": run.get("workflow_id"), "workflow_path": WORKFLOW_PATH,
                "branch": "main", "run_id": str(run_id), "run_attempt": attempt, "event_name": event,
                "as_of": as_of.isoformat(), "target_fingerprint": feishu_target_fingerprint(
                    webhook, app_id, env.get("FEISHU_RECEIVE_ID_TYPE", "chat_id"), chat_id, env.get("FEISHU_DOMAIN", "feishu"))}
    _check_run(run, identity)
    if run.get("id") != run_id or run.get("run_attempt") != attempt or run.get("event") != event:
        raise EvidenceError("Current run identity cannot be verified")
    prior = collect_prior(client, identity)
    if resend:
        prior = dict.fromkeys(COMPONENTS, "allow")
    planned = {"full": COMPONENTS, "stocks-only": ("stocks", "document"), "market-only": ("market",)}[args.mode]
    directory = Path("reports/daily_delivery")
    intent, receipt = create_delivery_files(directory, identity, planned, prior)
    if resend:
        # Retain the override in both artifacts; never delete prior delivery evidence.
        for path in (intent, receipt):
            from src.core.daily_delivery import atomic_write_json
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["explicit_resend"] = True
            atomic_write_json(path, payload)
    for destination, lines in ((env.get("GITHUB_ENV"), ["DAILY_DELIVERY_FILE=" + str(receipt.resolve())]),
                               (env.get("GITHUB_OUTPUT"), ["intent_name=" + artifact_name("intent", identity),
                                                          "receipt_name=" + artifact_name("receipt", identity),
                                                          "enabled=true"])):
        if not destination:
            raise EvidenceError("GitHub workflow output files are required")
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    print("Daily delivery prepared for " + identity["as_of"] + ": " + ", ".join(f"{key}={prior[key]}" for key in planned))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("prepare")
    command.add_argument("--as-of", default="")
    command.add_argument("--mode", choices=("full", "stocks-only", "market-only"), required=True)
    command.add_argument("--allow-resend", choices=("true", "false"), default="false")
    command = subparsers.add_parser("finish")
    command.add_argument("--file", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            return prepare(args)
        okay = finalize_daily_delivery(args.file or None)
        print("Daily delivery receipt finalized" if okay else "Daily delivery is blocked, partial, rejected or uncertain")
        return 0 if okay else 1
    except EvidenceError as exc:
        print("Daily delivery evidence unavailable: " + str(exc), file=sys.stderr)
    except Exception as exc:
        print("Daily delivery stopped (%s); no automatic send is permitted" % type(exc).__name__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
