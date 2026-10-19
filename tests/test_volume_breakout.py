"""Frozen-rule regression cases, not a claim of future investment returns.

The 240-row fixture preserves the October 15 Tencent qfq download ending on
2026-10-14. Twelve/thirteen October are evaluated by truncating that same file;
no candles after October 14 or subsequent price outcomes are used here.
"""
from datetime import date
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.pattern_screener import (
    BASE_PATTERN_NAMES,
    VOLUME_BREAKOUT_NAME,
    ScreeningDataError,
    detect_patterns,
    inspect_volume_breakout,
)
from tests.test_pattern_screener import AS_OF as SYNTHETIC_AS_OF, POSITIVE_CASES


AS_OF = date(2026, 10, 14)
FIXTURE = Path(__file__).parent / "fixtures" / "patterns" / "yxt_002727_20261014.csv"


def synthetic_bars(tail=None, open_=10.51, close=11.0, volume=3000.0):
    tail = np.linspace(10, 10.5, 40) if tail is None else np.asarray(tail)
    closes = np.r_[np.full(199 - len(tail), 10.0), tail, close]
    opens = closes * .997
    frame = pd.DataFrame({
        "date": pd.bdate_range(end=AS_OF, periods=len(closes)),
        "open": opens, "high": np.maximum(opens, closes) + .03,
        "low": np.minimum(opens, closes) - .03, "close": closes,
        "volume": np.full(len(closes), 1000.0),
    })
    frame.loc[frame.index[-1], ["open", "high", "low", "volume"]] = (
        open_, max(open_, close) + .03, min(open_, close) - .03, volume,
    )
    return frame


class VolumeBreakoutRealHistoryTest(unittest.TestCase):
    def setUp(self):
        self.bars = pd.read_csv(FIXTURE)

    def test_fixture_is_the_full_dated_original_export_of_qfq_bars(self):
        # Export values are not perturbed to make the new signal pass.
        self.assertEqual(len(self.bars), 240)
        self.assertEqual(self.bars.date.iloc[0], "2025-10-17")
        self.assertEqual(self.bars.date.iloc[-1], "2026-10-14")
        last = self.bars.iloc[-1]
        self.assertEqual((last.open, last.high, last.low, last.close, last.volume),
                         (11.67, 12.60, 11.62, 12.46, 33122800.0))

    def test_october14_passes_the_frozen_rules_and_is_a_low_open_not_a_gap_up(self):
        details = inspect_volume_breakout(self.bars, AS_OF)
        self.assertTrue(details["matched"], details)
        self.assertTrue(all(details["checks"].values()))
        self.assertEqual(details["evidence"]["突破前20日高点"], 11.98)
        self.assertEqual(details["evidence"]["突破幅度百分比"], 4.01)
        self.assertEqual(details["evidence"]["量比（此前20日）"], 5.37)
        self.assertEqual(details["evidence"]["MA20乖离百分比"], 14.01)
        self.assertEqual(details["evidence"]["5日累计涨幅百分比"], 16.23)
        self.assertEqual(details["evidence"]["开盘跳空百分比"], -.77)
        self.assertEqual([signal.name for signal in detect_patterns(self.bars, AS_OF)],
                         [VOLUME_BREAKOUT_NAME])

    def test_october12_is_rejected_only_for_unrepaired_prior_ma20_background(self):
        details = inspect_volume_breakout(self.bars, date(2026, 10, 12))
        self.assertFalse(details["matched"])
        self.assertEqual([key for key, passed in details["checks"].items() if not passed],
                         ["突破前MA20走平或抬升"])
        self.assertEqual(details["evidence"]["突破前MA20三日变化百分比"], -.37)

    def test_october13_is_not_a_close_breakout_or_strong_candle(self):
        details = inspect_volume_breakout(self.bars, date(2026, 10, 13))
        self.assertFalse(details["matched"])
        for key in ("阳线实体足够", "当日涨幅足够", "收盘有效突破20日高点",
                    "收盘接近当日最高", "实体占比足够"):
            with self.subTest(check=key):
                self.assertFalse(details["checks"][key])
        self.assertEqual(details["evidence"]["突破前20日高点"], 11.79)
        self.assertEqual(details["evidence"]["阳线实体百分比"], 1.82)

    def test_original_eight_results_are_preserved_when_extension_is_disabled(self):
        self.assertEqual(BASE_PATTERN_NAMES, (
            "老鸭头", "黄金坑", "空中加油", "底部红三兵",
            "杯柄形态", "一阳穿多线", "多方炮", "圆弧底",
        ))
        for day in (12, 13, 14):
            with self.subTest(day=day):
                as_of = date(2026, 10, day)
                original = detect_patterns(self.bars, as_of, include_extensions=False)
                combined = detect_patterns(self.bars, as_of)
                self.assertEqual(original, [])
                self.assertEqual([signal for signal in combined if signal.name != VOLUME_BREAKOUT_NAME], original)

    def test_historical_cutoff_matches_explicitly_truncated_input(self):
        for day in (12, 13, 14):
            with self.subTest(day=day):
                as_of = date(2026, 10, day)
                truncated = self.bars[self.bars.date <= as_of.isoformat()].copy()
                self.assertEqual(inspect_volume_breakout(self.bars, as_of),
                                 inspect_volume_breakout(truncated, as_of))

    def test_appending_future_prices_cannot_change_the_october14_signal(self):
        extended = self.bars.copy()
        future = self.bars.tail(3).copy()
        future["date"] = ["2026-10-15", "2026-10-16", "2026-10-19"]
        future[["open", "high", "low", "close", "volume"]] = [1, 1_000_000, .1, 1_000, 0]
        extended = pd.concat([extended, future], ignore_index=True)
        self.assertEqual(inspect_volume_breakout(extended, AS_OF), inspect_volume_breakout(self.bars, AS_OF))
        self.assertEqual(detect_patterns(extended, AS_OF), detect_patterns(self.bars, AS_OF))

    def test_diagnostics_share_the_existing_data_quality_guards(self):
        for frame, reason in ((self.bars.iloc[:-1], "stale"),
                              (self.bars.tail(119), "short_history")):
            with self.subTest(reason=reason):
                with self.assertRaises(ScreeningDataError) as error:
                    inspect_volume_breakout(frame, AS_OF)
                self.assertEqual(error.exception.reason, reason)
        invalid = self.bars.copy()
        invalid.loc[invalid.index[-1], "volume"] = np.nan
        with self.assertRaises(ScreeningDataError) as error:
            inspect_volume_breakout(invalid, AS_OF)
        self.assertEqual(error.exception.reason, "invalid_ohlcv")


class VolumeBreakoutCounterexampleTest(unittest.TestCase):
    def assert_blocked(self, frame, key):
        details = inspect_volume_breakout(frame, AS_OF)
        self.assertFalse(details["matched"], details)
        self.assertFalse(details["checks"][key], details)
        self.assertNotIn(VOLUME_BREAKOUT_NAME,
                         [signal.name for signal in detect_patterns(frame, AS_OF)])
        return details

    def test_generic_slow_trend_breakout_does_not_need_a_stock_code(self):
        frame = synthetic_bars()
        details = inspect_volume_breakout(frame, AS_OF)
        self.assertTrue(details["matched"], details)
        self.assertEqual(detect_patterns(frame, AS_OF, include_extensions=False), [])
        self.assertEqual([signal.name for signal in detect_patterns(frame, AS_OF)], [VOLUME_BREAKOUT_NAME])

    def test_original_eight_positive_cases_have_identical_results_with_extension_off(self):
        for name in BASE_PATTERN_NAMES:
            with self.subTest(pattern=name):
                frame = POSITIVE_CASES[name]()
                original = detect_patterns(frame, SYNTHETIC_AS_OF, include_extensions=False)
                combined = detect_patterns(frame, SYNTHETIC_AS_OF)
                self.assertEqual([signal.name for signal in original], [name])
                self.assertEqual([signal for signal in combined if signal.name != VOLUME_BREAKOUT_NAME], original)

    def test_high_volume_cannot_replace_a_close_breakout(self):
        frame = synthetic_bars()
        frame.loc[frame.index[-2], "high"] = 11.2
        self.assert_blocked(frame, "收盘有效突破20日高点")

    def test_intraday_breakout_followed_by_a_pullback_is_not_a_close_breakout(self):
        frame = synthetic_bars()
        frame.loc[frame.index[-2], "high"] = 11.02
        frame.loc[frame.index[-1], "high"] = 11.04
        self.assertGreater(frame.high.iloc[-1], frame.high.iloc[-21:-1].max())
        self.assertLess(frame.close.iloc[-1], frame.high.iloc[-21:-1].max())
        self.assert_blocked(frame, "收盘有效突破20日高点")

    def test_insufficient_volume_does_not_confirm_a_breakout(self):
        frame = synthetic_bars(volume=1999)
        self.assert_blocked(frame, "放量确认")

    def test_twice_the_prior_average_is_inclusive_and_excludes_today_from_baseline(self):
        details = inspect_volume_breakout(synthetic_bars(volume=2000), AS_OF)
        self.assertTrue(details["matched"], details)
        self.assertEqual(details["evidence"]["量比（此前20日）"], 2)

    def test_long_upper_wick_blocks_an_otherwise_large_up_day(self):
        frame = synthetic_bars()
        frame.loc[frame.index[-1], "high"] = 12
        self.assert_blocked(frame, "收盘接近当日最高")

    def test_open_below_existing_mas_is_a_different_pattern(self):
        frame = synthetic_bars(open_=10.4)
        self.assert_blocked(frame, "今开站上昨日四线")

    def test_today_cannot_repair_a_previously_falling_ma20_by_itself(self):
        prior = np.r_[np.linspace(11.2, 11.5, 10), np.linspace(11.4, 10.2, 10),
                      np.linspace(10.1, 10.9, 10)]
        frame = synthetic_bars(tail=prior, open_=11.25, close=11.8)
        ma20 = frame.close.rolling(20).mean()
        self.assertGreater(ma20.iloc[-1], ma20.iloc[-2])
        self.assertLess(ma20.iloc[-2], ma20.iloc[-5])
        self.assert_blocked(frame, "突破前MA20走平或抬升")

    def test_excessive_ma20_deviation_is_blocked(self):
        frame = synthetic_bars(open_=10.6, close=12.3)
        self.assert_blocked(frame, "MA20乖离受控")

    def test_five_day_overheating_is_a_separate_guard_from_ma20_deviation(self):
        prior = np.r_[np.full(25, 10.5), [10, 10.2, 10.5, 10.8, 11.1]]
        frame = synthetic_bars(tail=prior, open_=11.5, close=12.1)
        details = self.assert_blocked(frame, "5日涨幅受控")
        self.assertTrue(details["checks"]["MA20乖离受控"])

    def test_excessive_positive_open_gap_is_blocked(self):
        frame = synthetic_bars(open_=11.1, close=11.5)
        self.assert_blocked(frame, "正跳空幅度受控")

    def test_one_price_candle_is_not_a_tradable_breakout_confirmation(self):
        frame = synthetic_bars()
        frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = 11.0
        details = self.assert_blocked(frame, "阳线实体足够")
        self.assertFalse(details["checks"]["实体占比足够"])

    def test_small_real_body_cannot_pass_just_because_price_is_up(self):
        frame = synthetic_bars(open_=10.8, close=11)
        self.assert_blocked(frame, "阳线实体足够")

    def test_inspector_does_not_modify_cached_input(self):
        frame = synthetic_bars()
        before = frame.copy(deep=True)
        inspect_volume_breakout(frame, AS_OF)
        pd.testing.assert_frame_equal(frame, before)


if __name__ == "__main__":
    unittest.main()
