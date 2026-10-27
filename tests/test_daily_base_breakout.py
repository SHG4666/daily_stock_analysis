"""一心堂企稳启动由兼容模式接入每日真实换手率路径；不以代码放行。"""
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.bottom_reversal import PROFILE, RULESET_VERSION, inspect_bottom_reversal
from src.pattern_screener import VOLUME_BREAKOUT_NAME, inspect_volume_breakout, prepare_bars
from src.services.pattern_screening_service import screen_market, write_result

FIXTURES = Path(__file__).parent / 'fixtures/patterns'
AS_OF = date(2026, 10, 14)


class FrozenProvider:
    source_name = '腾讯冻结前复权日K'
    universe_source = '历史回归报价'
    universe_url = history_url = turnover_url = 'https://example.invalid'
    turnover_source = '腾讯历史实际换手率'

    def __init__(self, as_of=AS_OF, turnover_problem=None, code='002727'):
        self.as_of = as_of
        self.turnover_problem = turnover_problem
        self.code = code
        self.frame = pd.read_csv(FIXTURES / 'yxt_002727_20261014.csv')
        self.turnover = pd.read_csv(FIXTURES / 'yxt_002727_20261014_turnover.csv')

    def get_universe(self):
        rows = self.frame[self.frame.date <= self.as_of.isoformat()]
        latest = rows.iloc[-1]
        return [{'code': self.code, 'name': '一心堂' if self.code == '002727' else '其他结构样本',
                 'open': float(latest.open), 'price': float(latest.close), 'volume': float(latest.volume),
                 'amount': 405_798_470.16}]

    def get_daily_data(self, *args):
        return self.frame

    def get_turnover_data(self, *args):
        if self.turnover_problem == 'error': raise TimeoutError()
        data = self.turnover.copy()
        if self.turnover_problem == 'flat': data['turnover'] = 1.
        elif self.turnover_problem == 'stale': data = data[data.date < self.as_of.isoformat()]
        return data


def test_actual_yxt_october14_enters_daily_pool_with_turnover_and_shared_detector(tmp_path):
    provider = FrozenProvider()
    frame = prepare_bars(provider.frame, AS_OF)
    metrics, rejection = inspect_bottom_reversal(frame)
    assert metrics['prior_decline'] > 0 and rejection is None
    assert metrics['volume_breakout'] == inspect_volume_breakout(provider.frame, AS_OF)
    assert metrics['setup'] == VOLUME_BREAKOUT_NAME
    result = screen_market(provider, AS_OF, profile=PROFILE)
    assert result['status'] == 'complete' and result['turnover_coverage'] == 1
    assert result['ruleset_version'] == RULESET_VERSION
    assert result['parameters']['bottom_reversal']['volume_breakout'] == result['parameters']['volume_breakout']
    stock = result['selected'][0]
    assert stock['code'] == '002727' and len(stock['signals']) == 1
    assert stock['signals'][0]['name'] == VOLUME_BREAKOUT_NAME
    assert stock['bottom_reversal']['turnover'] == 8.33
    assert stock['bottom_reversal']['turnover_previous_ratio'] == pytest.approx(8.33 / 4.72)
    assert stock['bottom_reversal']['turnover_five_mean_ratio'] == pytest.approx(8.33 / 2.822)
    write_result(result, tmp_path)
    rendered = (tmp_path / 'report.html').read_text()
    assert VOLUME_BREAKOUT_NAME in rendered and '突破前20日高点：11.98' in rendered
    assert 'https://xueqiu.com/S/SZ002727' in rendered and '<svg' not in rendered
    assert (tmp_path / 'stocks.txt').read_text() == '002727'


@pytest.mark.parametrize('day,failed_check', [(12, '突破前MA20走平或抬升'), (13, '阳线实体足够')])
def test_earlier_real_dates_are_not_backdated_into_a_later_breakout(day, failed_check):
    provider = FrozenProvider(date(2026, 10, day))
    result = screen_market(provider, provider.as_of, profile=PROFILE)
    metrics, _ = inspect_bottom_reversal(prepare_bars(provider.frame, provider.as_of))
    assert not metrics['volume_breakout']['checks'][failed_check]
    assert result['status'] == 'complete' and result['selected'] == []


@pytest.mark.parametrize('problem,status', [('flat', 'complete'), ('error', 'incomplete'), ('stale', 'incomplete')])
def test_breakout_cannot_substitute_volume_for_real_turnover(problem, status):
    result = screen_market(FrozenProvider(turnover_problem=problem), AS_OF, profile=PROFILE)
    assert result['status'] == status and result['selected'] == []
    if status == 'incomplete':
        assert result['turnover_coverage'] == 0 and result['data_issues']['turnover_data'] == 1


@pytest.mark.parametrize('case,check', [
    ('no_breakout', '收盘有效突破20日高点'), ('weak_volume', '放量确认'),
    ('long_wick', '收盘接近当日最高'), ('large_gap', '正跳空幅度受控'),
    ('ma_deviation', 'MA20乖离受控'),
])
def test_daily_integration_keeps_existing_launch_quality_checks(case, check):
    provider = FrozenProvider()
    if case == 'no_breakout': provider.frame.loc[238, 'high'] = 13.
    elif case == 'weak_volume': provider.frame.loc[239, 'volume'] *= .25
    elif case == 'long_wick': provider.frame.loc[239, 'high'] = 14.
    elif case == 'large_gap': provider.frame.loc[239, 'open'] = 12.4
    elif case == 'ma_deviation': provider.frame.loc[239, ['close', 'high']] = [13., 13.1]
    metrics, rejection = inspect_bottom_reversal(prepare_bars(provider.frame, AS_OF))
    assert not metrics['volume_breakout']['checks'][check] and rejection is not None
    assert screen_market(provider, AS_OF, profile=PROFILE)['selected'] == []


def test_shape_detection_does_not_whitelist_the_reference_stock():
    result = screen_market(FrozenProvider(code='000001'), AS_OF, profile=PROFILE)
    assert result['selected'][0]['code'] == '000001'
    assert result['selected'][0]['signals'][0]['name'] == VOLUME_BREAKOUT_NAME


def test_future_prices_and_turnover_do_not_change_the_daily_historical_result():
    provider = FrozenProvider()
    baseline = screen_market(provider, AS_OF, profile=PROFILE)
    provider.frame = pd.concat([provider.frame, provider.frame.iloc[-1:].assign(
        date='2026-10-15', open=1., close=1_000., high=2_000., low=.1, volume=0.)], ignore_index=True)
    provider.turnover = pd.concat([provider.turnover, provider.turnover.iloc[-1:].assign(
        date='2026-10-15', turnover=999.)], ignore_index=True)
    extended = screen_market(provider, AS_OF, profile=PROFILE)
    assert extended['selected'] == baseline['selected']


def test_new_launch_reaches_daily_cli_and_github_candidate_handoff(tmp_path, monkeypatch):
    from scripts import screen_eight_patterns as entry
    from tests.test_pattern_entrypoint import set_clock
    set_clock(monkeypatch, datetime(2026, 10, 14, 18, tzinfo=ZoneInfo('Asia/Shanghai')), AS_OF, True)
    monkeypatch.setenv('GITHUB_OUTPUT', str(tmp_path / 'handoff'))
    monkeypatch.setattr(entry, 'PatternMarketData', lambda *args: FrozenProvider())
    monkeypatch.setattr(entry, 'enrich_shareholder_counts', lambda *a, **k: None)
    assert entry.main(['--profile', PROFILE, '--output-dir', str(tmp_path)]) == 0
    result = json.loads((tmp_path / 'screen.json').read_text())
    assert result['ruleset_version'] == RULESET_VERSION
    assert result['selected'][0]['signals'][0]['name'] == VOLUME_BREAKOUT_NAME
    assert 'stock_codes=002727\n' in (tmp_path / 'handoff').read_text()


def test_previous_ranking_policy_is_not_reused_after_launch_eligibility_changes():
    from src.core import screen_learning as core
    from src.services import screen_learning_service as learning
    from tests.test_screen_learning import context
    data = context()
    data['policy']['ruleset_version'] = '5.0-pullback-continuation'
    data['policy_digest'] = core.policy_digest(data['policy'])
    candidates = [{'code': '000001'}, {'code': '000002'}]
    assert not learning.apply_learned_ranking(candidates, data, date(2026, 10, 23))['applied']
    assert [x['code'] for x in candidates] == ['000001', '000002']
