"""Offline case-review CLI: real rules, historical cutoffs and auditable outputs."""
import csv
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from scripts.review_pattern_cases import main
from src import pattern_screener as detector
from tests.test_pattern_screener import AS_OF, cross_multiple_mas, make_bars, set_candle


@pytest.fixture(autouse=True)
def no_network():
    with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("case review must stay offline")):
        yield


def volume_breakout():
    bars = make_bars(np.r_[np.linspace(10, 10.3, 40), 10.9],
                     np.r_[np.full(40, 1000), 3000])
    set_candle(bars, -1, 10.31, 10.9, 3000)
    return bars


def write_case(directory, bars, filename="inputs/sample.csv", **labels):
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(path, index=False)
    case = {"code": "002727", "name": "标签名称", "csv": filename, "as_of": AS_OF.isoformat()}
    case.update(labels)
    return case, path


def write_manifest(directory, cases):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "cases.json"
    path.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    return path


def invoke(manifest, directory):
    code = main(["--manifest", str(manifest), "--output-dir", str(directory)])
    result = json.loads((directory / "review.json").read_text(encoding="utf-8"))
    return code, result


def test_real_old8_and_extension_results_have_hashes_checks_and_three_readable_outputs(tmp_path):
    bundle = tmp_path / "bundle"
    old_case, old_csv = write_case(bundle, cross_multiple_mas(), "inputs/old.csv", expected_extension=False)
    new_case, new_csv = write_case(bundle, volume_breakout(), "inputs/new.csv", code="extension", expected_extension=True)
    manifest = write_manifest(bundle, [old_case, new_case])
    before = {path: path.read_bytes() for path in (manifest, old_csv, new_csv)}
    exit_code, result = invoke(manifest, tmp_path / "output")

    assert exit_code == 0
    assert result["summary"] == {"total": 2, "ok": 2, "expectation_failures": 0, "data_errors": 0, "manifest_errors": 0}
    assert result["ruleset_version"] == detector.RULESET_VERSION
    assert result["detector_sha256"] == hashlib.sha256(Path(detector.__file__).read_bytes()).hexdigest()
    assert result["manifest_sha256"] == hashlib.sha256(before[manifest]).hexdigest()
    assert result["parameters"]["volume_breakout"] == detector.VOLUME_BREAKOUT_LIMITS
    old, new = result["cases"]
    assert old["old_patterns"] == ["一阳穿多线"]
    assert old["new_patterns"] == old["old_patterns"]
    assert old["added"] == [] and old["extension_matched"] is False
    assert new["old_patterns"] == []
    assert new["new_patterns"] == new["added"] == ["均线上方放量突破"]
    assert new["extension_matched"] is True and new["expectation_met"] is True
    assert all(new["checks"].values()) and not all(old["checks"].values())
    assert new["evidence"]["量比（此前20日）"] == 3.0
    for item, path in zip(result["cases"], (old_csv, new_csv)):
        assert item["csv_sha256"] == hashlib.sha256(before[path]).hexdigest()
        assert item["last_reviewed_date"] == item["as_of"] == AS_OF.isoformat()
        assert item["reviewed_rows"] == 200
    for path, original in before.items():
        assert path.read_bytes() == original

    with (tmp_path / "output/review.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert json.loads(rows[1]["checks"]) == new["checks"]
    assert json.loads(rows[1]["evidence"]) == new["evidence"]
    assert json.loads(rows[1]["parameters"]) == result["parameters"]
    assert rows[1]["detector_sha256"] == result["detector_sha256"]
    assert rows[0]["checks_false"] and rows[1]["checks_true"]
    markdown = (tmp_path / "output/review.md").read_text(encoding="utf-8")
    assert "| 放量确认 | 真 |" in markdown and "| 今开站上昨日四线 | 假 |" in markdown
    assert result["detector_sha256"] in markdown and new["csv_sha256"] in markdown
    assert "未计算收益、未执行全市场回测或候选排序" in markdown


def test_relative_csv_is_resolved_from_manifest_and_labels_do_not_affect_rules(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    case, source = write_case(bundle, volume_breakout(), expected_extension=True)
    relabeled = {**case, "code": "任意标签不是股票代码", "name": "另一名称"}
    manifest = write_manifest(bundle, [case, relabeled])
    unrelated = tmp_path / "elsewhere"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    exit_code, result = invoke(manifest, tmp_path / "output")
    assert exit_code == 0
    first, second = result["cases"]
    assert first["csv_path"] == second["csv_path"] == str(source.resolve())
    assert second["code"] == relabeled["code"]
    for field in ("old_patterns", "new_patterns", "checks", "evidence", "extension_matched"):
        assert first[field] == second[field]


def test_each_target_date_excludes_future_bars_even_when_future_ohlcv_is_invalid(tmp_path):
    bars = volume_breakout()
    previous_date = bars.date.iloc[-2].date()
    future = pd.DataFrame([{
        "date": AS_OF + timedelta(days=3), "open": -1, "high": -2,
        "low": 1000000, "close": -10, "volume": -100,
    }])
    with_future = pd.concat([bars, future], ignore_index=True)
    bundle = tmp_path / "bundle"
    first, _ = write_case(bundle, with_future, as_of=previous_date.isoformat(), expected_extension=False)
    second = {**first, "as_of": AS_OF.isoformat(), "expected_extension": True}
    manifest = write_manifest(bundle, [first, second])
    exit_code, result = invoke(manifest, tmp_path / "output")
    assert exit_code == 0
    before, target = result["cases"]
    assert before["last_reviewed_date"] == previous_date.isoformat()
    assert before["reviewed_rows"] == 199 and before["extension_matched"] is False
    assert target["last_reviewed_date"] == AS_OF.isoformat()
    assert target["reviewed_rows"] == 200 and target["input_rows"] == 201
    expected = detector.inspect_volume_breakout(bars, AS_OF)
    assert target["checks"] == expected["checks"] and target["evidence"] == expected["evidence"]


def test_repeated_run_is_identical_and_does_not_modify_configuration_or_github_outputs(tmp_path, monkeypatch):
    case, _ = write_case(tmp_path, volume_breakout())
    manifest = write_manifest(tmp_path, [case])
    config_path = tmp_path / ".env"
    github_output = tmp_path / "github-output.txt"
    config_path.write_text("STOCK_LIST=unchanged\n", encoding="utf-8")
    github_output.write_text("existing-output\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.chdir(tmp_path)
    first_code, first = invoke(manifest, tmp_path / "first")
    second_code, second = invoke(manifest, tmp_path / "second")
    assert first_code == second_code == 0
    assert first == second and first["cases"][0]["expectation_met"] is None
    for suffix in ("json", "csv", "md"):
        assert (tmp_path / f"first/review.{suffix}").read_bytes() == (tmp_path / f"second/review.{suffix}").read_bytes()
    assert config_path.read_text() == "STOCK_LIST=unchanged\n"
    assert github_output.read_text() == "existing-output\n"


def test_expectation_failure_returns_one_but_keeps_actual_match_and_evidence(tmp_path):
    case, _ = write_case(tmp_path, volume_breakout(), expected_extension=False)
    exit_code, result = invoke(write_manifest(tmp_path, [case]), tmp_path / "output")
    assert exit_code == 1
    assert result["summary"]["expectation_failures"] == 1
    reviewed = result["cases"][0]
    assert reviewed["status"] == "expectation_failed"
    assert reviewed["extension_matched"] is True and reviewed["expectation_met"] is False
    assert reviewed["checks"] and reviewed["evidence"]


def test_data_errors_are_not_no_matches_and_take_precedence_over_expectation_failure(tmp_path):
    good, _ = write_case(tmp_path, volume_breakout(), expected_extension=False)
    bad, bad_csv = write_case(tmp_path, pd.DataFrame({"date": [AS_OF.isoformat()], "close": [10]}),
                              "inputs/bad.csv", code="bad-columns")
    missing = {**good, "code": "missing-file", "csv": "inputs/missing.csv"}
    last = {**good, "code": "last-valid", "expected_extension": True}
    exit_code, result = invoke(write_manifest(tmp_path, [good, bad, missing, last]), tmp_path / "output")
    assert exit_code == 2
    assert result["summary"] == {"total": 4, "ok": 1, "expectation_failures": 1, "data_errors": 2, "manifest_errors": 0}
    reviewed = result["cases"]
    assert reviewed[1]["error"]["reason"] == "missing_columns"
    assert reviewed[1]["csv_sha256"] == hashlib.sha256(bad_csv.read_bytes()).hexdigest()
    assert reviewed[2]["error"]["type"] == "FileNotFoundError"
    assert "extension_matched" not in reviewed[1] and "old_patterns" not in reviewed[2]
    assert reviewed[3]["status"] == "ok"
    markdown = (tmp_path / "output/review.md").read_text(encoding="utf-8")
    assert "未计算" in markdown and "missing_columns" in markdown and "FileNotFoundError" in markdown


@pytest.mark.parametrize("manifest", [
    [], {"cases": []}, {"cases": [None]},
    {"cases": [{"code": "x", "name": "x", "csv": "x.csv", "as_of": "20261009"}]},
    {"cases": [{"code": "x", "name": "x", "csv": "x.csv", "as_of": "2026-10-09", "expected_extension": "true"}]},
])
def test_bad_manifest_is_reported_without_reading_cases(tmp_path, manifest):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    exit_code, result = invoke(path, tmp_path / "output")
    assert exit_code == 2 and result["summary"]["manifest_errors"] == 1
    assert result["cases"] == [] and result["manifest_error"]["message"]
    assert (tmp_path / "output/review.csv").exists() and (tmp_path / "output/review.md").exists()


def test_duplicate_code_and_date_is_rejected_even_when_name_or_csv_differs(tmp_path):
    case, _ = write_case(tmp_path, volume_breakout())
    duplicate = {**case, "name": "不同名称", "csv": "another.csv"}
    exit_code, result = invoke(write_manifest(tmp_path, [case, duplicate]), tmp_path / "output")
    assert exit_code == 2
    assert "重复案例" in result["manifest_error"]["message"]
    assert result["cases"] == []


@pytest.mark.parametrize("content", [b"{not-json", b"\xff\xfe"])
def test_unreadable_manifest_content_has_explicit_error_artifacts(tmp_path, content):
    path = tmp_path / "cases.json"
    path.write_bytes(content)
    exit_code, result = invoke(path, tmp_path / "output")
    assert exit_code == 2 and result["summary"]["manifest_errors"] == 1
    assert result["manifest_sha256"] == hashlib.sha256(content).hexdigest()


def test_unexpected_detector_exception_is_visible_instead_of_becoming_no_match(tmp_path):
    case, _ = write_case(tmp_path, volume_breakout())
    with mock.patch("scripts.review_pattern_cases.detector.inspect_volume_breakout", side_effect=RuntimeError("detector exploded")):
        exit_code, result = invoke(write_manifest(tmp_path, [case]), tmp_path / "output")
    assert exit_code == 2
    assert result["cases"][0]["error"] == {"type": "RuntimeError", "message": "detector exploded"}
    assert "extension_matched" not in result["cases"][0]


@pytest.mark.parametrize("collision", ["bad-manifest", "csv", "symlink"])
def test_output_paths_never_overwrite_manifest_or_input_csv(tmp_path, collision, capsys):
    if collision == "bad-manifest":
        protected = tmp_path / "review.json"
        protected.write_bytes(b"{invalid manifest")
        manifest = protected
    else:
        filename = "review.csv" if collision == "csv" else "inputs/sample.csv"
        case, protected = write_case(tmp_path, volume_breakout(), filename)
        manifest = write_manifest(tmp_path, [case])
        if collision == "symlink":
            (tmp_path / "review.csv").symlink_to(protected)
    before = protected.read_bytes()
    assert main(["--manifest", str(manifest), "--output-dir", str(tmp_path)]) == 2
    assert protected.read_bytes() == before
    assert "冲突" in capsys.readouterr().err
