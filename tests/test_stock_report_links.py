"""Quote links survive report variants and Feishu conversion without name injection."""
from html.parser import HTMLParser
from unittest import mock

import markdown2
import pytest

from src.analyzer import AnalysisResult
from src.config import Config
from src.formatters import format_feishu_markdown
from src.notification import NotificationService
from src.services.report_renderer import render
from src.stock_links import format_stock_name_markdown, get_a_share_quote_url


@pytest.mark.parametrize("code,symbol", [
    ("001391", "sz001391"), ("300879", "sz300879"),
    ("600519", "sh600519"), ("688001", "sh688001"),
    ("430047", "bj430047"), ("830799", "bj830799"),
    ("870001", "bj870001"), ("880001", "bj880001"), ("920978", "bj920978"),
    ("SH600519", "sh600519"), ("sz.001391", "sz001391"),
    ("688001.SH", "sh688001"), ("BJ920978", "bj920978"),
    ("SH.600519", "sh600519"), ("SS.600519", "sh600519"), ("600519.SS", "sh600519"),
    ("SZ300879", "sz300879"), ("SZ.300879", "sz300879"), ("300879.SZ", "sz300879"),
    ("BJ.920978", "bj920978"), ("920978.BJ", "bj920978"),
    ("  600519.sh  ", "sh600519"), ("bj.920978", "bj920978"),
])
def test_quote_url_maps_only_known_a_share_codes(code, symbol):
    assert get_a_share_quote_url(code) == f"https://gu.qq.com/{symbol}"


@pytest.mark.parametrize("code", [
    "AAPL", "HK00700", "00700", "0700.HK", "600519.HK", "HK600519",
    "510300", "900001", "123456", "60051", "6005191", "60１２３４", None,
    "600519](https://untrusted.example)", "https://untrusted.example/600519",
    "SS600519", "SH600519.SZ", "SH.600519.SH", "600519.XSHG", "SZ-000001",
])
def test_unknown_market_or_invalid_code_never_gets_a_guessed_link(code):
    assert get_a_share_quote_url(code) is None


@pytest.mark.parametrize("code", [
    "SH000001", "SH.000001", "000001.SH", "SS.000001", "000001.SS",
    "SH300879", "SH.300879", "300879.SH", "300879.SS",
    "SZ600519", "SZ.600519", "600519.SZ",
    "BJ600519", "BJ.600519", "600519.BJ",
    "SH920978", "SH.920978", "920978.SH", "920978.SS",
    "SZ920978", "SZ.920978", "920978.SZ",
    "BJ300879", "BJ.300879", "300879.BJ", "sh000001", "000001.sh",
])
def test_explicit_exchange_conflict_never_links_another_security(code):
    assert get_a_share_quote_url(code) is None
    assert format_stock_name_markdown("原始标的", code) == "原始标的"


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.urls.append(dict(attrs).get("href"))


@pytest.mark.parametrize("name", [
    "*ST国华", "名称[示例]", r"名称\](https://untrusted.example)",
    "名称](https://untrusted.example) [", "名称\n[其他](https://untrusted.example)",
    '<a href="https://untrusted.example">伪装名称</a>',
])
def test_stock_name_cannot_inject_another_markdown_or_html_link(name):
    formatted = format_stock_name_markdown(name, "001391")
    parser = Anchors()
    parser.feed(markdown2.markdown(formatted))
    assert parser.urls == ["https://gu.qq.com/sz001391"]


def test_unknown_market_remains_plain_name_and_legacy_underscore_choices_are_kept():
    assert format_stock_name_markdown("Apple", "AAPL") == "Apple"
    assert format_stock_name_markdown("*ST_示例", "AAPL") == r"\*ST_示例"
    assert format_stock_name_markdown("*ST_示例", "AAPL", escape_underscores=True) == r"\*ST\_示例"
    assert format_stock_name_markdown("", "001391") == ""


def result():
    return AnalysisResult(code="001391", name="国货航", sentiment_score=72,
                          trend_prediction="看多", operation_advice="持有",
                          analysis_summary="观察量价", dashboard={})


@pytest.mark.parametrize("method", [
    "generate_daily_report", "generate_dashboard_report", "generate_brief_report",
    "generate_single_stock_report", "generate_wechat_dashboard", "generate_wechat_summary",
])
@pytest.mark.parametrize("summary_only", [False, True])
def test_legacy_reports_link_stock_names_in_full_and_summary_modes(method, summary_only):
    config = Config(stock_list=[], report_renderer_enabled=False, report_summary_only=summary_only)
    stock = result()
    with mock.patch("src.notification.get_config", return_value=config):
        service = NotificationService()
        report = getattr(service, method)(stock if method == "generate_single_stock_report" else [stock])
    link = "[国货航](https://gu.qq.com/sz001391)"
    assert link in report
    assert link in format_feishu_markdown(report)
    assert stock.name == "国货航"
    assert stock.to_dict()["name"] == "国货航"


@pytest.mark.parametrize("platform", ["markdown", "wechat", "brief"])
@pytest.mark.parametrize("summary_only", [False, True])
def test_jinja_templates_keep_linked_names(platform, summary_only):
    config = Config(stock_list=[], report_templates_dir="templates")
    with mock.patch("src.services.report_renderer.get_config", return_value=config):
        report = render(platform, [result()], summary_only=summary_only)
    assert report is not None
    assert "[国货航](https://gu.qq.com/sz001391)" in report
    assert "[国货航](https://gu.qq.com/sz001391)" in format_feishu_markdown(report)


@pytest.mark.parametrize("method", ["generate_dashboard_report", "generate_brief_report"])
def test_notification_renderer_enabled_path_uses_linked_template_names(method):
    config = Config(stock_list=[], report_renderer_enabled=True, report_templates_dir="templates")
    with mock.patch("src.notification.get_config", return_value=config), \
         mock.patch("src.services.report_renderer.get_config", return_value=config):
        service = NotificationService()
        report = getattr(service, method)([result()])
    assert "[国货航](https://gu.qq.com/sz001391)" in report


def test_actual_feishu_card_keeps_linked_title_without_sending_a_second_chart_card():
    config = Config(stock_list=[], feishu_webhook_url="https://feishu.example/hook")
    response = mock.Mock(status_code=200)
    response.json.return_value = {"code": 0}
    with mock.patch("src.notification.get_config", return_value=config):
        service = NotificationService()
        report = service.generate_single_stock_report(result())
    with mock.patch("src.notification_sender.feishu_sender.requests.post", return_value=response) as post:
        assert service.send_to_feishu(report)
    post.assert_called_once()
    card = post.call_args.kwargs["json"]["card"]
    assert "[国货航](https://gu.qq.com/sz001391)" in card["elements"][0]["text"]["content"]
    assert all(element["tag"] != "chart" for element in card["elements"])
