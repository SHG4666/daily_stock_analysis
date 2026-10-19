#!/usr/bin/env python3
"""离线逐案复算原八形态与扩展规则，输出 review.json / review.csv / review.md。

退出码：0=全部完成且期望一致；1=至少一个 expected_extension 不符；
2=manifest、输入数据或输出错误（优先于期望不符）。不联网、不调用AI、
不发送通知，不计算收益、全市场回测或候选排名。CSV路径相对manifest解析。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pattern_screener as detector


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def strict_date(value):
    if not isinstance(value, str):
        raise ValueError("as_of必须是YYYY-MM-DD字符串")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("as_of必须是YYYY-MM-DD字符串")
    return parsed


def load_cases(manifest_path, manifest):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("manifest必须是包含非空cases数组的对象")
    cases, seen = [], set()
    for index, source in enumerate(manifest["cases"], start=1):
        if not isinstance(source, dict):
            raise ValueError(f"cases第{index}项必须是对象")
        for key in ("code", "name", "csv", "as_of"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f"cases第{index}项的{key}必须是非空字符串")
        as_of = strict_date(source["as_of"])
        code = source["code"]
        identity = (code, as_of)
        if identity in seen:
            raise ValueError(f"重复案例(code,as_of)：{code}, {as_of.isoformat()}")
        seen.add(identity)
        if "expected_extension" in source and not isinstance(source["expected_extension"], bool):
            raise ValueError(f"cases第{index}项的expected_extension必须是布尔值")
        csv_path = (manifest_path.parent / source["csv"]).resolve()
        cases.append({
            "code": code, "name": source["name"], "csv": source["csv"],
            "csv_path": str(csv_path), "as_of": as_of.isoformat(),
            "expected_extension": source.get("expected_extension"),
        })
    return cases


def review_case(case):
    record = dict(case)
    record["status"] = "error"
    try:
        raw = Path(case["csv_path"]).read_bytes()
        record["csv_sha256"] = sha256(raw)
        frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
        as_of = strict_date(case["as_of"])
        # The real validation cuts all future bars before numeric/OHLC checks.
        bars = detector.prepare_bars(frame, as_of)
        record.update(input_rows=len(frame), reviewed_rows=len(bars),
                      first_reviewed_date=bars.date.iloc[0].date().isoformat(),
                      last_reviewed_date=bars.date.iloc[-1].date().isoformat())
        old_patterns = [signal.name for signal in detector.detect_patterns(bars, as_of, include_extensions=False)]
        new_patterns = [signal.name for signal in detector.detect_patterns(bars, as_of, include_extensions=True)]
        extension = detector.inspect_volume_breakout(bars, as_of)
        if (not isinstance(extension.get("matched"), bool)
                or not isinstance(extension.get("checks"), dict)
                or any(not isinstance(value, bool) for value in extension["checks"].values())
                or not isinstance(extension.get("evidence"), dict)):
            raise ValueError("扩展规则解释结构不符合matched/checks/evidence契约")
        expected = case["expected_extension"]
        expectation_met = None if expected is None else extension["matched"] == expected
        record.update(
            status="expectation_failed" if expectation_met is False else "ok",
            old_patterns=old_patterns, new_patterns=new_patterns,
            added=[name for name in new_patterns if name not in old_patterns],
            extension_matched=extension["matched"], checks=extension["checks"], evidence=extension["evidence"],
            expectation_met=expectation_met,
        )
    except Exception as exc:
        # Keep the exact error visible; a broken case must not look like no match.
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, detector.ScreeningDataError):
            record["error"]["reason"] = exc.reason
    return record


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def markdown_cell(value):
    text = str(value).replace("\\", "\\\\").replace("|", "\\|")
    return text.replace("[", "\\[").replace("]", "\\]").replace("\n", " ").replace("\r", " ")


def render_markdown(result):
    summary = result["summary"]
    lines = [
        "# 技术形态样本复算", "",
        "这是指定CSV样本的规则核对；未计算收益、未执行全市场回测或候选排序。", "",
        f"规则版本：{result['ruleset_version']}",
        f"检测器SHA256：{result['detector_sha256']}",
        f"manifest SHA256：{result['manifest_sha256'] or '未读取'}", "",
        f"案例 {summary['total']}；成功 {summary['ok']}；期望不符 {summary['expectation_failures']}；"
        f"数据错误 {summary['data_errors']}；manifest错误 {summary['manifest_errors']}。", "",
        "## 参数快照", "", "```json", json.dumps(result["parameters"], ensure_ascii=False, indent=2), "```", "",
    ]
    if result.get("manifest_error"):
        error = result["manifest_error"]
        lines.extend([f"manifest错误：{markdown_cell(error['type'])}：{markdown_cell(error['message'])}", ""])
    lines.extend([
        "| 代码标签 | 名称标签 | 目标日 | 原八形态 | 新规则形态 | 新增 | 期望匹配 | 结果 |", "|---|---|---|---|---|---|---|---|",
    ])
    for case in result["cases"]:
        unavailable = "未计算" if case["status"] == "error" else "无"
        values = [case["code"], case["name"], case["as_of"], "、".join(case.get("old_patterns", [])) or unavailable,
                  "、".join(case.get("new_patterns", [])) or unavailable, "、".join(case.get("added", [])) or unavailable,
                  str(case["expected_extension"]) if case["expected_extension"] is not None else "未设定", case["status"]]
        lines.append("| " + " | ".join(markdown_cell(value) for value in values) + " |")
    for index, case in enumerate(result["cases"], start=1):
        lines.extend(["", f"## 案例{index}：{markdown_cell(case['code'])} {markdown_cell(case['name'])}", "",
                      f"目标日：{case['as_of']}；CSV：{markdown_cell(case['csv_path'])}",
                      f"CSV SHA256：{case.get('csv_sha256', '未读取')}"])
        if case.get("error"):
            error = case["error"]
            lines.append(f"错误：{markdown_cell(error['type'])}：{markdown_cell(error['message'])}")
            continue
        lines.extend([f"输入{case['input_rows']}行，截断并核验后{case['reviewed_rows']}行；"
                      f"区间{case['first_reviewed_date']}至{case['last_reviewed_date']}。", "",
                      f"扩展匹配：{case['extension_matched']}；期望一致：{case['expectation_met']}", "",
                      "| 检查条件 | 通过 |", "|---|---|"])
        lines.extend(f"| {markdown_cell(key)} | {'真' if value else '假'} |" for key, value in case["checks"].items())
        lines.extend(["", "证据：", "", "```json", json.dumps(case["evidence"], ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines) + "\n"


def write_review(result, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    fields = ["code", "name", "as_of", "csv_path", "csv_sha256", "manifest_sha256", "ruleset_version", "detector_sha256", "parameters",
              "status", "old_patterns", "new_patterns", "added", "extension_matched", "expected_extension",
              "expectation_met", "checks_true", "checks_false", "checks", "evidence", "error"]
    with (output_dir / "review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            row = {key: case.get(key, "") for key in fields}
            row.update(ruleset_version=result["ruleset_version"], detector_sha256=result["detector_sha256"],
                       manifest_sha256=result["manifest_sha256"], parameters=compact(result["parameters"]))
            row["checks_true"] = "；".join(key for key, value in case.get("checks", {}).items() if value)
            row["checks_false"] = "；".join(key for key, value in case.get("checks", {}).items() if not value)
            for key in ("old_patterns", "new_patterns", "added", "checks", "evidence", "error"):
                row[key] = compact(case.get(key))
            writer.writerow(row)
    (output_dir / "review.md").write_text(render_markdown(result), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path, output_dir = args.manifest.resolve(), args.output_dir.resolve()
    outputs = {(output_dir / f"review.{suffix}").resolve() for suffix in ("json", "csv", "md")}
    protected = {manifest_path, Path(__file__).resolve(), Path(detector.__file__).resolve()}
    if protected & outputs:
        print("输出路径与manifest或工具源码冲突，未写入任何结果", file=sys.stderr)
        return 2
    result = {
        "ruleset_version": detector.RULESET_VERSION,
        "detector_file": str(Path(detector.__file__).resolve()),
        "detector_sha256": sha256(Path(detector.__file__).read_bytes()),
        "manifest_path": str(manifest_path), "manifest_sha256": None,
        "parameters": {"volume_breakout": dict(detector.VOLUME_BREAKOUT_LIMITS),
                       "old8_include_extensions": False, "new_include_extensions": True},
        "cases": [], "summary": {"total": 0, "ok": 0, "expectation_failures": 0, "data_errors": 0, "manifest_errors": 0},
    }
    try:
        raw = manifest_path.read_bytes()
        result["manifest_sha256"] = sha256(raw)
        manifest = json.loads(raw.decode("utf-8-sig"))
        # Protect declared CSVs even if a later manifest field is malformed.
        sources = manifest.get("cases") if isinstance(manifest, dict) else None
        if isinstance(sources, list):
            protected.update(
                (manifest_path.parent / case["csv"]).resolve()
                for case in sources if isinstance(case, dict) and isinstance(case.get("csv"), str)
            )
        if protected & outputs:
            print("输出路径与manifest或输入CSV冲突，未写入任何结果", file=sys.stderr)
            return 2
        cases = load_cases(manifest_path, manifest)
        result["cases"] = [review_case(case) for case in cases]
        result["summary"].update(
            total=len(cases), ok=sum(case["status"] == "ok" for case in result["cases"]),
            expectation_failures=sum(case["status"] == "expectation_failed" for case in result["cases"]),
            data_errors=sum(case["status"] == "error" for case in result["cases"]),
        )
    except (OSError, ValueError, TypeError) as exc:
        result["manifest_error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["summary"]["manifest_errors"] = 1
    try:
        write_review(result, output_dir)
    except (OSError, ValueError, TypeError) as exc:
        print(f"结果写入失败：{type(exc).__name__}：{exc}", file=sys.stderr)
        return 2
    summary = result["summary"]
    print(f"案例{summary['total']}：成功{summary['ok']}，期望不符{summary['expectation_failures']}，"
          f"数据错误{summary['data_errors']}，manifest错误{summary['manifest_errors']}")
    if summary["manifest_errors"] or summary["data_errors"]:
        return 2
    return 1 if summary["expectation_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
