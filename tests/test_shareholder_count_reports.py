"""Disclosed shareholder counts render consistently without fetching or guessing."""
from copy import deepcopy
from datetime import date
from unittest import mock

import pytest

from src.analyzer import AnalysisResult
from src.config import Config
from src.formatters import format_feishu_markdown
from src.notification import NotificationService
from src.services.report_renderer import render
from src.shareholder_count import format_shareholder_count


ZH_TEXT = "最新披露股东户数：39,518 户（截至 2026-06-30，公告 2026-08-27；来源：东方财富）"
EN_TEXT = ("Latest disclosed shareholder accounts: 39,518 "
           "(as of 2026-06-30, announced 2026-08-27; source: Eastmoney)")
METHODS = (
    "generate_daily_report", "generate_dashboard_report", "generate_brief_report",
    "generate_single_stock_report", "generate_wechat_dashboard", "generate_wechat_summary",
)


@pytest.fixture(autouse=True)
def offline_fixed_today():
    with mock.patch("src.shareholder_count._today", return_value=date(2026, 10, 18)), \
         mock.patch("requests.sessions.Session.request", side_effect=AssertionError("render must not fetch")):
        yield


def payload(**changes):
    value = {
        "code": "001391", "count": 39518, "end_date": "2026-06-30",
        "announcement_date": "2026-08-27", "source": "东方财富",
        "source_url": "https://data.eastmoney.com/gdhs/detail/001391.html",
        "fetched_at": "2026-10-18T08:00:00+08:00",
    }
    value.update(changes)
    return value


def stock(record, language="zh"):
    return AnalysisResult(
        code="001391", name="国货航", sentiment_score=72, trend_prediction="看多",
        operation_advice="持有", analysis_summary="观察量价", report_language=language,
        fundamental_context={"shareholder_count": record},
    )


def legacy_report(method, result, summary_only=False, renderer_enabled=False):
    config = Config(stock_list=[], report_renderer_enabled=renderer_enabled,
                    report_summary_only=summary_only, report_templates_dir="templates",
                    report_language=result.report_language)
    with mock.patch("src.notification.get_config", return_value=config), \
         mock.patch("src.services.report_renderer.get_config", return_value=config):
        service = NotificationService()
        return getattr(service, method)(result if method == "generate_single_stock_report" else [result])


@pytest.mark.parametrize("language,expected", [("zh", ZH_TEXT), ("en", EN_TEXT)])
def test_formatter_has_explicit_dates_source_and_language(language, expected):
    record = payload()
    before = deepcopy(record)
    assert format_shareholder_count(record, language, expected_code="SZ001391") == expected
    assert record == before


@pytest.mark.parametrize("count", [None, False, True, 0, -1, 1.25, float("nan"),
                                   float("inf"), float("-inf"), "39518", object()])
def test_invalid_counts_are_hidden_not_presented_as_zero(count):
    assert format_shareholder_count(payload(count=count)) == ""


def test_integral_finite_numeric_count_can_be_formatted_without_decimal_fraction():
    assert format_shareholder_count(payload(count=39518.0)) == ZH_TEXT


@pytest.mark.parametrize("changes", [
    {"end_date": "20260630"}, {"end_date": "2026-6-30"},
    {"end_date": "2026-02-30"}, {"end_date": date(2026, 6, 30)},
    {"announcement_date": "2026-08-27T00:00:00"},
    {"end_date": "2026-08-28"},
    {"announcement_date": "2026-10-19"},
    {"end_date": "2026-10-19", "announcement_date": "2026-10-20"},
    {"fetched_at": "2026-10-19T08:00:00+08:00"},
    {"fetched_at": "2026-08-26T08:00:00+08:00"},
    {"fetched_at": "not-a-date"}, {"fetched_at": "2026-10-18"},
    {"source": "其他来源"}, {"code": "AAPL"},
])
def test_invalid_dates_future_records_or_unsupported_provenance_are_hidden(changes):
    assert format_shareholder_count(payload(**changes)) == ""


def test_fetch_date_is_validated_in_china_time_instead_of_utc_date_prefix():
    record = payload(end_date="2026-10-18", announcement_date="2026-10-18",
                     fetched_at="2026-10-17T17:00:00+00:00")
    assert "截至 2026-10-18，公告 2026-10-18" in format_shareholder_count(record)


@pytest.mark.parametrize("record", [None, {}, [], "missing"])
def test_missing_records_have_no_placeholder(record):
    assert format_shareholder_count(record) == ""


@pytest.mark.parametrize("expected_code", ["600519", "SH000001", "AAPL"])
def test_record_must_belong_to_the_displayed_stock(expected_code):
    assert format_shareholder_count(payload(), expected_code=expected_code) == ""


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("summary_only", [False, True])
@pytest.mark.parametrize("language,expected", [("zh", ZH_TEXT), ("en", EN_TEXT)])
def test_all_legacy_report_entries_show_count_once_in_full_and_summary_modes(
    method, summary_only, language, expected,
):
    result = stock(payload(), language)
    report = legacy_report(method, result, summary_only)
    assert report.count(expected) == 1
    assert expected in format_feishu_markdown(report)
    assert "[国货航](https://xueqiu.com/S/SZ001391)" in report
    assert result.name == "国货航"
    assert "shareholder_count" not in result.to_dict()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("summary_only", [False, True])
def test_missing_count_does_not_add_zero_or_unavailable_hint(method, summary_only):
    report = legacy_report(method, stock(None), summary_only)
    assert "股东户数" not in report
    assert "shareholder accounts" not in report
    assert "[国货航](https://xueqiu.com/S/SZ001391)" in report


@pytest.mark.parametrize("platform", ["markdown", "wechat", "brief"])
@pytest.mark.parametrize("summary_only", [False, True])
@pytest.mark.parametrize("language,expected", [("zh", ZH_TEXT), ("en", EN_TEXT)])
def test_actual_templates_show_count_in_both_branches(platform, summary_only, language, expected):
    config = Config(stock_list=[], report_templates_dir="templates", report_language=language)
    with mock.patch("src.services.report_renderer.get_config", return_value=config):
        report = render(platform, [stock(payload(), language)], summary_only=summary_only)
    assert report is not None
    assert report.count(expected) == 1
    assert "[国货航](https://xueqiu.com/S/SZ001391)" in report


@pytest.mark.parametrize("platform", ["markdown", "wechat", "brief"])
@pytest.mark.parametrize("record", [None, {"code": "600519"}])
def test_templates_hide_missing_or_wrong_stock_records(platform, record):
    config = Config(stock_list=[], report_templates_dir="templates")
    if record:
        record = payload(code=record["code"])
    with mock.patch("src.services.report_renderer.get_config", return_value=config):
        report = render(platform, [stock(record)], summary_only=True)
    assert report is not None
    assert "股东户数" not in report


@pytest.mark.parametrize("method", ["generate_dashboard_report", "generate_wechat_dashboard", "generate_brief_report"])
@pytest.mark.parametrize("summary_only", [False, True])
def test_notification_template_enabled_path_carries_count(method, summary_only):
    report = legacy_report(method, stock(payload()), summary_only, renderer_enabled=True)
    assert report.count(ZH_TEXT) == 1


def test_count_stays_separate_from_stock_link_and_invalid_name_cannot_change_target():
    result = stock(payload())
    result.name = "名称](https://untrusted.example) ["
    report = legacy_report("generate_single_stock_report", result)
    assert ZH_TEXT in report
    assert report.count("](https://xueqiu.com/S/SZ001391)") == 1
    assert "\\](https://untrusted.example) \\[" in report
    assert ZH_TEXT not in report.split("](https://xueqiu.com/S/SZ001391)", 1)[0]
