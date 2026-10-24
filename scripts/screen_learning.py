#!/usr/bin/env python3
"""每日学习档案：只读恢复凭证、收盘建档、模型/发送结果定稿；本脚本不发送消息。"""
import argparse
import gzip
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.daily_delivery import EvidenceError, GitHubEvidence, WORKFLOW_PATH, _check_run, _verified_artifact, artifact_json
from data_provider.learning_market_data import collect_close_snapshot
from data_provider.pattern_market_data import PatternMarketData
from src.core.trading_calendar import get_effective_trading_date, get_market_now, infer_market_phase, MarketPhase, is_market_open
from src.services.screen_learning_service import (
    MAX_BYTES, empty_state, load_state, prepare, record_selection, save_state, validate_state,
)
from src.core.screen_learning import sessions

logger = logging.getLogger(__name__)


def restore_remote(client, current_run):
    """只信任本仓库main每日工作流的不可变artifact，不从cache推定完整历史。"""
    identity = {"repository": client.repository, "workflow_id": current_run["workflow_id"]}
    _check_run(current_run, identity)
    found = None
    for page in range(1, 4):
        payload = client.get_json("/actions/artifacts", {"per_page": 100, "page": page})
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("无法确认学习档案列表")
        matches = [a for a in artifacts if re.fullmatch(r"screen-learning-state-\d+-\d+", a.get("name", ""))]
        if matches:
            found = max(matches, key=lambda a: a["created_at"])
            break
        if len(artifacts) < 100:
            return None
    if found is None:
        raise ValueError("学习档案检索超过预算，不能静默重新开始")
    run_id = found["workflow_run"]["id"]
    run = client.get_json(f"/actions/runs/{run_id}")
    _check_run(run, identity)
    _verified_artifact(found, run, identity)
    if not found["name"].startswith(f"screen-learning-state-{run_id}-"):
        raise ValueError("学习档案身份不符")
    with zipfile.ZipFile(io.BytesIO(client.download(found))) as archive:
        entries = archive.infolist()
        if len(entries) != 1 or Path(entries[0].filename).name != "state.json.gz" or entries[0].file_size > 20*1024*1024:
            raise ValueError("学习档案压缩格式无效")
        with gzip.GzipFile(fileobj=io.BytesIO(archive.read(entries[0]))) as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("学习档案展开过大")
        state = validate_state(json.loads(raw))
    if state.get("repository") != client.repository or state.get("workflow_path") != WORKFLOW_PATH:
        raise ValueError("学习档案仓库不符")
    return state


def read_optional(path):
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def bootstrap_same_close(client, current_run, state):
    """只引入同一收盘日的旧日报，不用事后走势重建过去的特征。"""
    day = state["days"][-1]
    if "selected" in day:
        return
    from datetime import date, timedelta
    next_session = sessions(day["as_of"], date.fromisoformat(day["as_of"]) + timedelta(days=20))[1]
    now = get_market_now("cn")
    if now.date().isoformat() > next_session or (now.date().isoformat() == next_session and now.hour >= 9):
        return
    artifacts = client.get_json("/actions/artifacts", {"per_page": 100}).get("artifacts", [])
    for artifact in sorted(artifacts, key=lambda a: a["created_at"], reverse=True):
        if not re.fullmatch(r"analysis-reports-\d+", artifact.get("name", "")) or artifact.get("expired"):
            continue
        run = client.get_json(f"/actions/runs/{artifact['workflow_run']['id']}")
        identity = {"repository": client.repository, "workflow_id": current_run["workflow_id"]}
        _check_run(run, identity)
        _verified_artifact(artifact, run, identity)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")).astimezone(ZoneInfo("Asia/Shanghai"))
        if created.date().isoformat() < day["as_of"]:
            return
        raw = client.download(artifact)
        screen = artifact_json(raw, "screen.json", required=False)
        if not screen or screen.get("as_of") != day["as_of"] or screen.get("status") != "complete":
            continue
        receipt = artifact_json(raw, "receipt.json", required=False)
        if not receipt:
            continue
        from src.core.daily_delivery import validate_receipt
        validate_receipt(receipt)
        if (receipt["identity"]["repository"] != client.repository
                or str(receipt["identity"]["run_id"]) != str(run["id"])):
            raise ValueError("旧日报发送凭证身份不符")
        published, rejected = [], []
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            target = "report_" + day["as_of"].replace("-", "") + ".md"
            reports = [entry for entry in archive.infolist() if Path(entry.filename).name == target]
            if len(reports) != 1 or reports[0].file_size > 2*1024*1024:
                continue
            text = archive.read(reports[0]).decode("utf-8")
            published = re.findall(r"^## [^\n]*\((\d{6})\)\s*$", text, re.MULTILINE)
            for entry in archive.infolist():
                if re.fullmatch(r"stock_analysis_\d{8}\.log", Path(entry.filename).name) and entry.file_size < 10*1024*1024:
                    rejected += re.findall(r"\[(\d{6})\] 不进入每日推荐：", archive.read(entry).decode("utf-8"))
        if not published or len(published) != len(set(published)):
            continue
        audit = {"as_of": day["as_of"], "run_id": str(run["id"]),
                 "input_codes": [item["code"] for item in screen["selected"]],
                 "accepted": published, "rejected": sorted(set(rejected))}
        record_selection(state, screen, audit, receipt)
        day["selection_origin"] = {"run_id": run["id"], "artifact_id": artifact["id"], "source": "verified_legacy_report"}
        return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "finalize"))
    parser.add_argument("--state", default="data/screen_learning/state.json.gz")
    parser.add_argument("--output-dir", default="reports/screen_learning")
    parser.add_argument("--screen", default="reports/pattern_screen/screen.json")
    parser.add_argument("--audit", default="reports/screen_learning/model_audit.json")
    parser.add_argument("--receipt", default="reports/daily_delivery/receipt.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from src.config import setup_env
        setup_env()
        if args.mode == "prepare":
            now = get_market_now("cn")
            as_of = get_effective_trading_date("cn", current_time=now)
            if now.date() == as_of and infer_market_phase("cn", current_time=now) != MarketPhase.POSTMARKET:
                raise ValueError("尚未完成收盘，不能建立学习快照")
            if now.date() != as_of and is_market_open("cn", now.date()) and now.hour >= 9:
                raise ValueError("盘中不能冒充昨日收盘快照")
            # GitHub永远由artifact恢复；本地才从显式路径继续。获取失败不可清空已有学习状态。
            if os.getenv("GITHUB_RUN_ID"):
                client = GitHubEvidence(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
                run = client.get_json("/actions/runs/" + os.environ["GITHUB_RUN_ID"])
                state = restore_remote(client, run)
                if state is None:
                    logger.info("没有可恢复的学习档案，从今天重新积累，历史准确率不作推定")
                    state = empty_state()
                state["repository"] = client.repository
                state["workflow_path"] = WORKFLOW_PATH
            else:
                state = load_state(args.state)
            snapshot = collect_close_snapshot(PatternMarketData(Path("data/pattern_screen_cache")), as_of)
            snapshot["captured_at"] = now.isoformat()
            review = prepare(state, snapshot, args.output_dir)
            if os.getenv("GITHUB_RUN_ID"):
                try:
                    bootstrap_same_close(client, run, state)
                    if "selected" in state["days"][-1]:
                        review = prepare(state, state["days"][-1], args.output_dir)
                except (EvidenceError, ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
                    logger.warning("旧日报未能作为首次记录引入：%s", type(exc).__name__)
            save_state(state, args.state)
            if os.getenv("GITHUB_OUTPUT"):
                with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
                    stream.write(f"as_of={as_of.isoformat()}\n")
            logger.info("学习快照 %s 覆盖 %.2f%%，%s", as_of, 100*snapshot["coverage"], review["policy"]["status"])
        else:
            state = load_state(args.state)
            if not state["days"]:
                raise ValueError("没有本次学习快照")
            screen = read_optional(args.screen)
            if screen:
                record_selection(state, screen, read_optional(args.audit), read_optional(args.receipt))
            prepare(state, state["days"][-1], args.output_dir)
            save_state(state, args.state)
        return 0
    except Exception as exc:
        logger.error("学习模块未就绪，正常选股沿用原排序：%s", type(exc).__name__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
