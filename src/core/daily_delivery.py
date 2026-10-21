"""Opt-in, component-scoped evidence for a daily Feishu delivery.

The workflow persists an intent before it enables this file. A local receipt is
not a distributed lock: workflow concurrency and the uploaded artifacts provide
the cross-run boundary. A successful callback alone never proves delivery.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

logger = logging.getLogger(__name__)
COMPONENTS = ("stocks", "market", "document")
STATUSES = {"not_attempted", "active_call", "acknowledged", "rejected", "partial", "unknown"}
_GUARDS: dict[tuple[str, str, str], "_Guard"] = {}
_GUARDS_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict) -> None:
    """Replace a complete JSON document; never leave a truncated receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def feishu_target_fingerprint(webhook_url, app_id="", receive_id_type="chat_id", chat_id="", domain="feishu") -> str:
    """Hash routing identity, never credentials such as app/webhook signing secrets."""
    webhook = str(webhook_url or "").strip()
    if webhook:
        target = ["feishu_webhook", webhook]
    else:
        receive_type = str(receive_id_type or "chat_id").strip().lower()
        receive_type = receive_type if receive_type in {"chat_id", "open_id"} else "chat_id"
        raw_domain = str(domain or "feishu").strip().lower()
        raw_domain = raw_domain if raw_domain in {"feishu", "lark"} else "feishu"
        target = ["feishu_app", str(app_id or "").strip(), receive_type, str(chat_id or "").strip(), raw_domain]
    return hashlib.sha256(json.dumps(target, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _components(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(values))
    if not result or any(value not in COMPONENTS for value in result):
        raise ValueError("Daily delivery requires known components")
    return result


def validate_receipt(payload: dict, *, kind="receipt") -> dict:
    """Validate the small artifact schema before using it as sending evidence."""
    if (not isinstance(payload, dict) or type(payload.get("version")) is not int
            or payload["version"] != 1 or payload.get("kind") != kind):
        raise ValueError("Invalid daily delivery schema")
    identity = payload.get("identity")
    required = {"repository", "workflow_id", "workflow_path", "branch", "run_id", "run_attempt",
                "event_name", "as_of", "target_fingerprint"}
    if not isinstance(identity, dict) or not required <= identity.keys():
        raise ValueError("Missing daily delivery identity")
    if (any(not isinstance(identity[key], str) or not identity[key] for key in {
            "repository", "workflow_path", "branch", "event_name", "as_of", "target_fingerprint"})
            or not str(identity["run_id"]).isdigit() or int(identity["run_id"]) < 1
            or type(identity["run_attempt"]) is not int or identity["run_attempt"] < 1
            or type(identity["workflow_id"]) is not int or identity["workflow_id"] < 1):
        raise ValueError("Invalid daily delivery identity types")
    if (date.fromisoformat(identity["as_of"]).isoformat() != identity["as_of"]
            or len(identity["target_fingerprint"]) != 64
            or any(c not in "0123456789abcdef" for c in identity["target_fingerprint"])):
        raise ValueError("Invalid daily delivery date or target")
    components = payload.get("components")
    if not isinstance(components, dict) or not components or not set(components) <= set(COMPONENTS):
        raise ValueError("Invalid daily delivery components")
    for state in components.values():
        if (not isinstance(state, dict) or state.get("prior") not in {"allow", "delivered", "blocked"}
                or state.get("status") not in STATUSES or type(state.get("finalized")) is not bool
                or type(state.get("active_call")) is not bool):
            raise ValueError("Invalid daily delivery state")
        posts = state.get("posts")
        if (not isinstance(posts, dict) or set(posts) != {"ack", "rejected", "unknown"}
                or any(type(v) is not int or v < 0 for v in posts.values())):
            raise ValueError("Invalid daily delivery POST evidence")
        if type(state.get("calls")) is not int or state["calls"] < 0:
            raise ValueError("Invalid daily delivery call count")
        if (type(state.get("failed_calls")) is not int
                or not 0 <= state["failed_calls"] <= state["calls"]):
            raise ValueError("Invalid daily delivery logical failure count")
        if (state["active_call"] != (state["status"] == "active_call")
                or (state["active_call"] and state["finalized"])
                or (state["status"] == "acknowledged" and (
                    not posts["ack"] or posts["ack"] < state["calls"] or posts["unknown"] or state["failed_calls"]))
                or (state["status"] == "not_attempted" and any(posts.values()))
                or (any(posts.values()) and not state["calls"])
                or (state["status"] == "rejected" and (not posts["rejected"] or posts["ack"] or posts["unknown"]))
                or (state["status"] in {"rejected", "partial"} and not state["failed_calls"])
                or (state["status"] == "partial" and not posts["ack"])):
            raise ValueError("Inconsistent daily delivery evidence")
    if not isinstance(payload.get("refused", []), list) or any(
            not isinstance(value, str) for value in payload.get("refused", [])):
        raise ValueError("Invalid daily delivery refusal evidence")
    return payload


def create_delivery_files(directory: Path, identity: dict, components: Sequence[str], prior=None) -> tuple[Path, Path]:
    """Pure local preparation, also used by offline integration fixtures.

    ``prior`` maps each component to allow/delivered/blocked. Production callers
    must obtain that map from GitHub artifact evidence before calling this.
    """
    prior = prior or {}
    states = {
        component: {"prior": prior.get(component, "allow"), "status": "not_attempted",
                    "finalized": False, "active_call": False, "posts": {"ack": 0, "rejected": 0, "unknown": 0},
                    "calls": 0, "failed_calls": 0}
        for component in _components(components)
    }
    payload = {"version": 1, "kind": "receipt", "identity": dict(identity), "components": states, "refused": [],
               "prepared_at": _now(), "updated_at": _now()}
    validate_receipt(payload)
    directory = Path(directory)
    intent, receipt = directory / "intent.json", directory / "receipt.json"
    atomic_write_json(intent, {**payload, "kind": "intent"})
    atomic_write_json(receipt, payload)
    return intent, receipt


@dataclass
class _Call:
    guard: "_Guard"
    components: tuple[str, ...]
    posts: dict[str, int] = field(default_factory=lambda: {"ack": 0, "rejected": 0, "unknown": 0})


_CURRENT_CALL: ContextVar[_Call | None] = ContextVar("daily_delivery_call", default=None)


class _Guard:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.data = validate_receipt(json.loads(path.read_text(encoding="utf-8")))
        identity = self.data["identity"]
        for env, key in (("GITHUB_RUN_ID", "run_id"), ("GITHUB_RUN_ATTEMPT", "run_attempt"),
                         ("GITHUB_REPOSITORY", "repository")):
            if os.getenv(env) and str(identity[key]) != os.environ[env]:
                raise ValueError("Daily delivery belongs to another run")

    def save(self):
        self.data["updated_at"] = _now()
        atomic_write_json(self.path, self.data)

    def refuse(self, code):
        if code not in self.data.setdefault("refused", []):
            self.data["refused"].append(code)
        self.save()


def _guard(path=None) -> _Guard | None:
    value = str(path or os.getenv("DAILY_DELIVERY_FILE", "")).strip()
    if not value:
        return None
    resolved = str(Path(value).resolve())
    key = (resolved, os.getenv("GITHUB_RUN_ID", ""), os.getenv("GITHUB_RUN_ATTEMPT", ""))
    with _GUARDS_LOCK:
        if key not in _GUARDS:
            _GUARDS[key] = _Guard(Path(resolved))
        return _GUARDS[key]


def daily_delivery_kwargs(components: Sequence[str]) -> dict:
    return {"daily_components": _components(components)} if os.getenv("DAILY_DELIVERY_FILE") else {}


def record_daily_post(outcome: str) -> None:
    """Called only at the actual Feishu transport outcome, in the callback context."""
    call = _CURRENT_CALL.get()
    if call is None:
        return
    if outcome not in call.posts:
        raise ValueError("Unknown daily POST outcome")
    with call.guard.lock:
        call.posts[outcome] += 1
        for component in call.components:
            call.guard.data["components"][component]["posts"][outcome] += 1
        call.guard.save()


def daily_post_is_uncertain() -> bool:
    call = _CURRENT_CALL.get()
    return bool(call and call.posts["unknown"])


def run_daily_delivery(target: str, components: Sequence[str], send: Callable[[], bool]) -> bool:
    """Execute a logical delivery only if its component evidence permits it."""
    try:
        guard = _guard()
        if guard is None:
            return bool(send())
        components = _components(components)
        with guard.lock:
            if target != guard.data["identity"]["target_fingerprint"]:
                logger.error("Daily Feishu destination differs from the prepared intent")
                guard.refuse("destination_mismatch")
                return False
            states = [guard.data["components"].get(component) for component in components]
            if any(state is None for state in states):
                logger.error("Daily delivery component was not declared")
                guard.refuse("undeclared_component")
                return False
            delivered = [state["prior"] == "delivered" or (
                state["status"] == "acknowledged" and state["finalized"]) for state in states]
            if all(delivered):
                logger.info("Daily Feishu report already acknowledged; skipping delivery")
                return True
            if any(delivered) or any(state["prior"] == "blocked" or state["active_call"]
                                     or state["status"] in {"partial", "unknown"} for state in states):
                logger.error("Daily delivery evidence requires an explicit manual resend")
                guard.refuse("mixed_or_uncertain_components")
                return False
            for state in states:
                state.update(status="active_call", active_call=True, finalized=False, calls=state["calls"] + 1)
            guard.save()
            call = _Call(guard, components)
            token = _CURRENT_CALL.set(call)
            try:
                result = bool(send())
            except Exception as exc:
                logger.error("Daily delivery callback failed (%s)", type(exc).__name__)
                record_daily_post("unknown")
                result = False
            finally:
                _CURRENT_CALL.reset(token)
            for state in states:
                posts = state["posts"]
                if not result or call.posts["unknown"] or not call.posts["ack"]:
                    state["failed_calls"] += 1
                if state["status"] == "unknown" or posts["unknown"] or (result and not call.posts["ack"]):
                    status = "unknown"
                elif posts["ack"] and state["failed_calls"]:
                    status = "partial"
                elif result:
                    status = "acknowledged"
                elif posts["rejected"]:
                    status = "rejected"
                else:
                    status = "not_attempted"
                state.update(status=status, active_call=False)
            if not result and not any(call.posts.values()):
                guard.refuse("callback_not_attempted")
            guard.save()
            return result and all(state["status"] == "acknowledged" for state in states)
    except Exception as exc:
        logger.error("Daily delivery evidence unavailable (%s); refusing to send", type(exc).__name__)
        return False


def finish_daily_components(*components: str) -> None:
    """Seal a completed phase; an active call or incomplete evidence is not success."""
    guard = _guard()
    if guard is None:
        return
    with guard.lock:
        for component in _components(components):
            state = guard.data["components"].get(component)
            if state is None:
                raise ValueError("Daily delivery component was not declared")
            if state["active_call"]:
                state.update(status="unknown", active_call=False)
            state["finalized"] = True
        guard.save()


def record_daily_component_failure(component: str) -> None:
    """Remember a report/pre-dispatch failure without inventing a POST outcome.

    One logical stock report can fail before Feishu's guarded callback starts.
    Subsequent stock ACKs must not erase that missing report from the phase.
    """
    guard = _guard()
    if guard is None:
        return
    with guard.lock:
        state = guard.data["components"].get(component)
        if state is None:
            guard.refuse("undeclared_failed_component")
            return
        if state["prior"] == "delivered" or (state["status"] == "acknowledged" and state["finalized"]):
            return
        state["calls"] += 1
        state["failed_calls"] += 1
        state["finalized"] = False
        if not state["active_call"]:
            if state["status"] == "unknown" or state["posts"]["unknown"]:
                state["status"] = "unknown"
            elif state["posts"]["ack"]:
                state["status"] = "partial"
            elif state["posts"]["rejected"]:
                state["status"] = "rejected"
            else:
                state["status"] = "not_attempted"
        guard.save()


def finalize_daily_delivery(path=None) -> bool:
    """Finish a workflow, sealing only phases where no POST was attempted."""
    guard = _guard(path)
    if guard is None:
        raise ValueError("Daily receipt is required")
    with guard.lock:
        for state in guard.data["components"].values():
            if state["active_call"] or (not state["finalized"] and state["status"] != "not_attempted"):
                state.update(status="unknown", active_call=False)
            elif state["status"] == "not_attempted":
                state["finalized"] = True
        guard.save()
        return not guard.data.get("refused") and all(not state["failed_calls"] and state["prior"] != "blocked" and state["status"] not in {
            "active_call", "rejected", "partial", "unknown"} for state in guard.data["components"].values())
