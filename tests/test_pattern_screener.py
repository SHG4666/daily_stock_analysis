"""Offline regression checks for the user's eight daily-candle patterns.

These are hand-built geometric examples, not a profitability backtest. They
exercise the real detector and validation without market APIs or notifications.
"""
from datetime import date
import unittest

import numpy as np
import pandas as pd

from src.pattern_screener import (
    PATTERN_NAMES,
    ScreeningDataError,
    detect_patterns,
    prepare_bars,
    volume_ratio,
)


AS_OF = date(2026, 10, 9)


def make_bars(tail, volumes=None):
    """Pad an explicit price path with a quiet baseline, keeping valid OHLC."""
    closes = np.r_[np.full(200 - len(tail), 10.0), tail]
    opens = closes * .997
    frame = pd.DataFrame({
        "date": pd.bdate_range(end=AS_OF, periods=len(closes)),
        "open": opens,
        "high": np.maximum(opens, closes) + .03,
        "low": np.minimum(opens, closes) - .03,
        "close": closes,
        "volume": np.full(len(closes), 1000.0),
    })
    if volumes is not None:
        frame.loc[frame.index[-len(volumes):], "volume"] = volumes
    return frame


def set_candle(frame, offset, open_, close, volume=None):
    index = frame.index[offset]
    frame.loc[index, ["open", "close", "high", "low"]] = (
        open_, close, max(open_, close) + .03, min(open_, close) - .03,
    )
    if volume is not None:
        frame.loc[index, "volume"] = volume


def duck_head():
    return make_bars(
        np.r_[np.linspace(10, 13.8, 20),
              [13.3, 13, 12.8, 12.6, 12.5, 12.45, 12.45, 12.5, 12.6, 12.8], 13.5],
        np.r_[np.full(20, 2000), np.full(10, 800), 4000],
    )


def golden_pit():
    return make_bars(
        np.r_[np.full(25, 10.0),
              [10, 9.8, 9.4, 9, 8.8, 8.5, 8.6, 8.7, 8.8, 8.9,
               9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.5, 9.5, 9.6, 9.7], 10.2],
        np.r_[np.full(45, 1000), 2000],
    )


def airborne_flag(consolidation_days=15, ignition=True):
    consolidation = [12.1, 12, 11.95, 12, 12.05, 12, 11.9, 11.95,
                     12.05, 12.1, 12, 12.05, 12.1, 12.15, 12.1]
    consolidation = consolidation[-consolidation_days:]
    frame = make_bars(
        np.r_[np.linspace(10, 11.4, 20), 12.2, consolidation, 12.6],
        np.r_[np.full(20, 1000), 3000, np.full(consolidation_days, 800), 4000],
    )
    # The image requires an initiating large bullish candle, followed by
    # two to four trading weeks of small candles before a second breakout.
    set_candle(frame, -consolidation_days - 2, 11.4 if ignition else 12.15, 12.2, 3000)
    return frame


def three_soldiers(bottom=True):
    context = np.linspace(12, 10, 20) if bottom else np.linspace(10, 12, 20)
    start = context[-1]
    frame = make_bars(np.r_[context, start + np.array([.2, .5, .8])])
    for offset, open_delta, close_delta in ((-3, 0, .2), (-2, .2, .5), (-1, .5, .8)):
        set_candle(frame, offset, start + open_delta, start + close_delta)
    return frame


def cup_and_handle():
    cup = 9 + 3 * np.linspace(-1, 1, 90) ** 2
    handle = [11.7, 11.5, 11.3, 11.2, 11.2, 11.3, 11.4, 11.5, 11.6, 11.8]
    return make_bars(
        np.r_[cup, handle, 12.3],
        np.r_[np.full(90, 2000), np.full(10, 800), 4000],
    )


def cross_multiple_mas():
    frame = make_bars([10.5], [2000])
    set_candle(frame, -1, 9.8, 10.5, 2000)
    return frame


def bullish_cannon():
    frame = make_bars([10.4, 10.2, 10.6], [2000, 1000, 3000])
    set_candle(frame, -3, 10, 10.4, 2000)
    set_candle(frame, -2, 10.35, 10.2, 1000)
    set_candle(frame, -1, 10.25, 10.6, 3000)
    return frame


def rounded_bottom():
    curve = 9 + 3 * np.linspace(-1, 1, 90) ** 2
    return make_bars(np.r_[curve, 12.4], np.r_[np.full(90, 1000), 2000])


POSITIVE_CASES = {
    "老鸭头": duck_head,
    "黄金坑": golden_pit,
    "空中加油": airborne_flag,
    "底部红三兵": three_soldiers,
    "杯柄形态": cup_and_handle,
    "一阳穿多线": cross_multiple_mas,
    "多方炮": bullish_cannon,
    "圆弧底": rounded_bottom,
}


class EightPatternDetectionTest(unittest.TestCase):
    def names(self, frame):
        return {signal.name for signal in detect_patterns(frame, AS_OF)}

    def test_each_pattern_has_a_constructed_positive_case_and_evidence(self):
        self.assertEqual(set(POSITIVE_CASES), set(PATTERN_NAMES))
        for name, builder in POSITIVE_CASES.items():
            with self.subTest(pattern=name):
                signals = detect_patterns(builder(), AS_OF)
                target = next((signal for signal in signals if signal.name == name), None)
                self.assertIsNotNone(target, [signal.name for signal in signals])
                self.assertTrue(target.evidence)
                self.assertIn("量比（此前20日）", target.to_dict()["evidence"])
                self.assertEqual(len(signals), len({signal.name for signal in signals}))

    def test_flat_market_is_not_a_pattern(self):
        self.assertEqual(detect_patterns(make_bars([10] * 200), AS_OF), [])

    def test_duck_head_requires_a_new_fast_ma_cross(self):
        frame = duck_head()
        set_candle(frame, -1, 12.6, 12.75, 4000)
        self.assertNotIn("老鸭头", self.names(frame))

    def test_duck_head_checks_ma30_support_on_each_pullback_day(self):
        frame = duck_head()
        ma30 = frame.close.rolling(30).mean()
        # The rising MA30 is materially higher late in the pullback. Comparing
        # against the whole interval's lowest MA30 would miss this support break.
        frame.loc[frame.index[-2], "low"] = 11.5
        self.assertGreater(11.5, .97 * ma30.iloc[-11:-1].min())
        self.assertLess(11.5, .97 * ma30.iloc[-2])
        self.assertNotIn("老鸭头", self.names(frame))

    def test_golden_pit_requires_reclaim(self):
        frame = golden_pit()
        set_candle(frame, -1, 9.5, 9.6, 2000)
        self.assertNotIn("黄金坑", self.names(frame))

    def test_airborne_flag_requires_breakout(self):
        frame = airborne_flag()
        set_candle(frame, -1, 11.9, 12, 4000)
        self.assertNotIn("空中加油", self.names(frame))

    def test_airborne_flag_requires_at_least_two_trading_weeks(self):
        self.assertNotIn("空中加油", self.names(airborne_flag(consolidation_days=5)))

    def test_airborne_flag_requires_an_initiating_large_bullish_candle(self):
        self.assertNotIn("空中加油", self.names(airborne_flag(ignition=False)))

    def test_airborne_flag_must_break_the_initiating_candle_high_too(self):
        frame = airborne_flag()
        frame.loc[frame.index[-17], "high"] = 12.8
        self.assertNotIn("空中加油", self.names(frame))

    def test_airborne_flag_requires_volume_on_the_initiating_candle(self):
        frame = airborne_flag()
        frame.loc[frame.index[-17], "volume"] = 1000
        self.assertNotIn("空中加油", self.names(frame))

    def test_airborne_flag_consolidation_must_hold_the_initiating_open(self):
        frame = airborne_flag()
        frame.loc[frame.index[-5], "low"] = 11.1
        self.assertNotIn("空中加油", self.names(frame))

    def test_three_bullish_candles_in_an_uptrend_are_not_bottom_soldiers(self):
        self.assertNotIn("底部红三兵", self.names(three_soldiers(bottom=False)))

    def test_three_soldiers_reject_a_long_upper_wick(self):
        frame = three_soldiers()
        frame.loc[frame.index[-1], "high"] = 12
        self.assertNotIn("底部红三兵", self.names(frame))

    def test_cup_handle_requires_breakout_above_rim(self):
        frame = cup_and_handle()
        set_candle(frame, -1, 11.8, 11.9, 4000)
        self.assertNotIn("杯柄形态", self.names(frame))

    def test_cup_handle_requires_shallow_handle(self):
        frame = cup_and_handle()
        frame.loc[frame.index[-6], "low"] = 9
        self.assertNotIn("杯柄形态", self.names(frame))

    def test_long_bullish_candle_must_start_below_the_mas(self):
        frame = cross_multiple_mas()
        set_candle(frame, -1, 10.1, 10.5, 2000)
        self.assertNotIn("一阳穿多线", self.names(frame))

    def test_cannon_third_candle_must_break_the_first_high(self):
        frame = bullish_cannon()
        set_candle(frame, -1, 10.25, 10.4, 3000)
        self.assertNotIn("多方炮", self.names(frame))

    def test_cannon_middle_candle_must_contract_volume(self):
        frame = bullish_cannon()
        frame.loc[frame.index[-2], "volume"] = 2500
        self.assertNotIn("多方炮", self.names(frame))

    def test_rounded_bottom_requires_breakout(self):
        frame = rounded_bottom()
        set_candle(frame, -1, 11.9, 12, 2000)
        self.assertNotIn("圆弧底", self.names(frame))

    def test_linear_v_recovery_is_not_a_rounded_bottom(self):
        # A sharp central cusp is not a smooth bowl even though a quadratic
        # regression may fit the entire symmetric V with a surprisingly high R².
        path = np.r_[np.linspace(12, 8, 45), np.linspace(8, 12, 45)]
        frame = make_bars(np.r_[path, 12.4], np.r_[np.full(90, 1000), 2000])
        self.assertNotIn("圆弧底", self.names(frame))

    def test_single_session_crash_recovery_is_not_a_rounded_bottom(self):
        path = np.full(90, 12.0)
        path[44:47] = [10, 8, 10]
        frame = make_bars(np.r_[path, 12.4], np.r_[np.full(90, 1000), 2000])
        self.assertNotIn("圆弧底", self.names(frame))

    def test_volume_gated_patterns_reject_unconfirmed_low_volume(self):
        for name, builder in POSITIVE_CASES.items():
            if name == "底部红三兵":
                continue
            with self.subTest(pattern=name):
                frame = builder()
                frame.loc[frame.index[-1], "volume"] = 100
                self.assertNotIn(name, self.names(frame))


class PatternDataValidationTest(unittest.TestCase):
    def assert_rejected(self, frame, reason, as_of=AS_OF):
        with self.assertRaises(ScreeningDataError) as error:
            detect_patterns(frame, as_of)
        self.assertEqual(error.exception.reason, reason)

    def test_short_history_is_not_silently_padded(self):
        self.assert_rejected(cross_multiple_mas().tail(119), "short_history")

    def test_months_old_cache_is_rejected(self):
        self.assert_rejected(cross_multiple_mas(), "stale", date(2026, 12, 31))

    def test_missing_latest_bar_is_rejected(self):
        self.assert_rejected(cross_multiple_mas().iloc[:-1], "stale")

    def test_duplicate_daily_bar_is_rejected(self):
        frame = cross_multiple_mas()
        frame = pd.concat([frame, frame.iloc[-1:]], ignore_index=True)
        self.assert_rejected(frame, "duplicate_dates")

    def test_two_timestamps_for_same_daily_session_are_duplicates(self):
        frame = cross_multiple_mas()
        repeated = frame.iloc[-1:].copy()
        repeated["date"] = repeated["date"] + pd.Timedelta(hours=15)
        self.assert_rejected(pd.concat([frame, repeated], ignore_index=True), "duplicate_dates")

    def test_shuffled_dates_are_sorted_without_changing_results(self):
        frame = cup_and_handle()
        shuffled = frame.sample(frac=1, random_state=17)
        self.assertEqual(detect_patterns(shuffled, AS_OF), detect_patterns(frame, AS_OF))

    def test_zero_negative_nan_and_infinite_values_are_rejected(self):
        for column in ("open", "high", "low", "close", "volume"):
            for value in (0, -1, np.nan, np.inf):
                with self.subTest(column=column, value=value):
                    frame = cross_multiple_mas()
                    frame.loc[frame.index[-5], column] = value
                    self.assert_rejected(frame, "invalid_ohlcv")

    def test_candle_bounds_are_validated(self):
        for column, value in (("high", 9), ("low", 11)):
            with self.subTest(column=column):
                frame = cross_multiple_mas()
                frame.loc[frame.index[-5], column] = value
                self.assert_rejected(frame, "invalid_ohlcv")

    def test_missing_column_and_invalid_date_have_explicit_reasons(self):
        frame = cross_multiple_mas()
        self.assert_rejected(frame.drop(columns="volume"), "missing_columns")
        frame["date"] = frame["date"].astype(object)
        frame.loc[0, "date"] = "not-a-date"
        self.assert_rejected(frame, "invalid_dates")

    def test_volume_baseline_excludes_the_breakout_day(self):
        frame = cross_multiple_mas()
        frame.loc[frame.index[-1], "volume"] = 20_000
        self.assertEqual(volume_ratio(prepare_bars(frame, AS_OF)), 20)

    def test_append_future_data_does_not_change_any_as_of_result(self):
        for name, builder in POSITIVE_CASES.items():
            with self.subTest(pattern=name):
                frame = builder()
                future = frame.iloc[-3:].copy()
                future["date"] = pd.bdate_range(start="2026-10-12", periods=3)
                # Even unusable OHLCV after the chosen session is irrelevant.
                future[["open", "high", "low", "close", "volume"]] = [1, 1, 0, 1, 0]
                extended = pd.concat([frame, future], ignore_index=True)
                self.assertEqual(detect_patterns(frame, AS_OF), detect_patterns(extended, AS_OF))

    def test_detection_does_not_modify_the_source_frame(self):
        frame = cup_and_handle()
        before = frame.copy(deep=True)
        detect_patterns(frame, AS_OF)
        pd.testing.assert_frame_equal(frame, before)


if __name__ == "__main__":
    unittest.main()
