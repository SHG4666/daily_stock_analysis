"""原图八形态与扩展放量突破规则，只产生待复核候选，不预测收益。"""
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd


BASE_PATTERN_NAMES = (
    "老鸭头", "黄金坑", "空中加油", "底部红三兵",
    "杯柄形态", "一阳穿多线", "多方炮", "圆弧底",
)
VOLUME_BREAKOUT_NAME = "均线上方放量突破"
PATTERN_NAMES = BASE_PATTERN_NAMES + (VOLUME_BREAKOUT_NAME,)
RULESET_VERSION = "2.0-volume-breakout"
VOLUME_BREAKOUT_LIMITS = {
    "volume_ratio_min": 2.0,
    "body_return_min": .03,
    "daily_return_min": .03,
    "resistance_sessions": 20,
    "breakout_margin_min": .005,
    "close_position_min": .75,
    "body_share_min": .50,
    "prior_ma_spread_max": .05,
    "ma20_deviation_max": .15,
    "five_session_return_max": .20,
    "positive_open_gap_max": .05,
    "prior_ma20_slope_sessions": 3,
}


class ScreeningDataError(ValueError):
    """不能用于当次筛选的数据，reason 用于覆盖统计。"""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PatternSignal:
    name: str
    evidence: dict

    def to_dict(self):
        return asdict(self)


def prepare_bars(bars: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """先截断未来数据，再严格核验日期、OHLC、量能与最新完整交易日。"""
    required = ["date", "open", "high", "low", "close", "volume"]
    if any(column not in bars for column in required):
        raise ScreeningDataError("missing_columns")
    frame = bars[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed").dt.normalize()
    if frame["date"].isna().any():
        raise ScreeningDataError("invalid_dates")
    frame = frame[frame["date"].dt.date <= as_of].sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].iloc[-1].date() != as_of:
        raise ScreeningDataError("stale")
    if frame["date"].duplicated().any():
        raise ScreeningDataError("duplicate_dates")
    if len(frame) < 120:
        raise ScreeningDataError("short_history")
    frame = frame.tail(240).reset_index(drop=True)
    numeric = required[1:]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame[numeric].to_numpy()).all() or (frame[numeric] <= 0).any().any():
        raise ScreeningDataError("invalid_ohlcv")
    if ((frame["low"] > frame[["open", "close"]].min(axis=1))
            | (frame["high"] < frame[["open", "close"]].max(axis=1))).any():
        raise ScreeningDataError("invalid_ohlcv")
    return frame


def volume_ratio(frame: pd.DataFrame) -> float:
    """当天量与此前 20 个交易日均量之比，不把当天纳入分母。"""
    return float(frame["volume"].iloc[-1] / frame["volume"].iloc[-21:-1].mean())


def _curve(values):
    smoothed = pd.Series(values).rolling(5).mean().dropna().to_numpy()
    x = np.linspace(0, 1, len(smoothed))
    coefficients = np.polyfit(x, smoothed, 2)
    fitted = np.polyval(coefficients, x)
    quadratic_error = float(np.sum((smoothed - fitted) ** 2))
    total = float(np.sum((smoothed - smoothed.mean()) ** 2))
    r2 = 1 - quadratic_error / total if total > 0 else 0
    minimum = float(np.argmin(smoothed) / (len(smoothed) - 1))
    depth = float((smoothed[:5].max() - smoothed.min()) / smoothed[:5].max())
    dwell = int(np.sum(smoothed <= smoothed.min() + .25 * (smoothed.max() - smoothed.min())))
    # 线性V同样能有很高的二次拟合R²；与尖底分段线性模型比较，不能只抬阈值。
    design = np.column_stack([np.ones(len(x)), x, np.abs(x - minimum)])
    v_fitted = design @ np.linalg.lstsq(design, smoothed, rcond=None)[0]
    v_error = float(np.sum((smoothed - v_fitted) ** 2))
    rounded = coefficients[0] > 0 and v_error >= .8 * quadratic_error
    return rounded, r2, minimum, depth, dwell


def _inspect_volume_breakout(frame, ma):
    """背景只用昨日以前，突破确认用当日完整K线；门槛为初始假设。"""
    limits = VOLUME_BREAKOUT_LIMITS
    current, previous = frame.iloc[-1], frame.iloc[-2]
    previous_mas = [series.iloc[-2] for series in ma.values()]
    today_mas = [series.iloc[-1] for series in ma.values()]
    ratio = volume_ratio(frame)
    body_return = float(current.close / current.open - 1)
    daily_return = float(current.close / previous.close - 1)
    candle_range = float(current.high - current.low)
    close_position = float((current.close - current.low) / candle_range) if candle_range > 0 else 0
    body_share = float((current.close - current.open) / candle_range) if candle_range > 0 else 0
    resistance = float(frame.high.iloc[-limits["resistance_sessions"] - 1:-1].max())
    breakout = float(current.close / resistance - 1)
    ma_spread = float(max(previous_mas) / min(previous_mas) - 1)
    slope = float(ma[20].iloc[-2] / ma[20].iloc[-2 - limits["prior_ma20_slope_sessions"]] - 1)
    deviation = float(current.close / ma[20].iloc[-1] - 1)
    five_return = float(current.close / frame.close.iloc[-6] - 1)
    open_gap = float(current.open / previous.close - 1)
    checks = {
        "前收站上昨日四线": previous.close > max(previous_mas),
        "今开站上昨日四线": current.open > max(previous_mas),
        "今收站上今日四线": current.close > max(today_mas),
        "短均线向上": (ma[5].iloc[-2] > ma[10].iloc[-2]
                     and ma[5].iloc[-1] > ma[5].iloc[-2] and ma[10].iloc[-1] > ma[10].iloc[-2]),
        "突破前MA20走平或抬升": slope >= 0,
        "突破前四线收敛": ma_spread <= limits["prior_ma_spread_max"],
        "阳线实体足够": body_return >= limits["body_return_min"],
        "当日涨幅足够": daily_return >= limits["daily_return_min"],
        "放量确认": ratio >= limits["volume_ratio_min"],
        "收盘有效突破20日高点": breakout >= limits["breakout_margin_min"],
        "收盘接近当日最高": close_position >= limits["close_position_min"],
        "实体占比足够": body_share >= limits["body_share_min"],
        "MA20乖离受控": deviation <= limits["ma20_deviation_max"],
        "5日涨幅受控": five_return <= limits["five_session_return_max"],
        "正跳空幅度受控": open_gap <= limits["positive_open_gap_max"],
    }
    return {
        "matched": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "evidence": {
            "突破前20日高点": round(resistance, 3),
            "突破幅度百分比": round(breakout * 100, 2),
            "阳线实体百分比": round(body_return * 100, 2),
            "当日涨幅百分比": round(daily_return * 100, 2),
            "量比（此前20日）": round(ratio, 2),
            "收盘位置百分比": round(close_position * 100, 2),
            "MA20乖离百分比": round(deviation * 100, 2),
            "5日累计涨幅百分比": round(five_return * 100, 2),
            "突破前MA20三日变化百分比": round(slope * 100, 2),
            "开盘跳空百分比": round(open_gap * 100, 2),
        },
    }


def inspect_volume_breakout(bars: pd.DataFrame, as_of: date) -> dict:
    """返回同一检测逻辑的逐条件解释，供离线样本复算，不联网。"""
    frame = prepare_bars(bars, as_of)
    ma = {window: frame.close.rolling(window).mean() for window in (5, 10, 20, 30)}
    return _inspect_volume_breakout(frame, ma)


def detect_patterns(bars: pd.DataFrame, as_of: date, *, include_extensions=True) -> list[PatternSignal]:
    frame = prepare_bars(bars, as_of)
    close, high, low, volume = (frame[column] for column in ("close", "high", "low", "volume"))
    current = frame.iloc[-1]
    previous = frame.iloc[-2]
    ratio = volume_ratio(frame)
    bullish = current.close > current.open
    body_return = float(current.close / current.open - 1)
    ma = {window: close.rolling(window).mean() for window in (5, 10, 20, 30)}
    matches = []

    def emit(name, **evidence):
        evidence["量比（此前20日）"] = round(ratio, 2)
        matches.append(PatternSignal(name, evidence))

    # 老鸭头：先上涨，再缩量回落，今天重新金叉，保留形成顺序。
    head = len(frame) - 60 + int(high.iloc[-60:-5].to_numpy().argmax())
    pullback = frame.iloc[head + 1:-1]
    rise = frame.iloc[max(0, head - 20):head + 1]
    retreat = 1 - float(pullback.close.min() / high.iloc[head]) if len(pullback) else 0
    prior_bull = ((ma[5].iloc[max(30, head - 20):head] > ma[30].iloc[max(30, head - 20):head])
                  & (ma[10].iloc[max(30, head - 20):head] > ma[30].iloc[max(30, head - 20):head])).any()
    first_crosses = all(
        ((ma[window].shift(1) <= ma[30].shift(1)) & (ma[window] > ma[30]))
        .iloc[max(30, head - 40):head + 1].any() for window in (5, 10)
    )
    if (bullish and ratio >= 1.3 and 5 <= len(pullback) <= 20 and prior_bull and first_crosses
            and .05 <= retreat <= .20 and high.iloc[head] / rise.close.min() >= 1.12
            and pullback.volume.mean() <= .8 * rise.volume.mean()
            and (pullback.low.to_numpy() >= .97 * ma[30].iloc[head + 1:-1].to_numpy()).all()
            and ma[30].iloc[-1] > ma[30].iloc[-11]
            and ma[5].iloc[-2] <= ma[10].iloc[-2] and ma[5].iloc[-1] > ma[10].iloc[-1]):
        emit("老鸭头", 回调天数=len(pullback), 回调幅度百分比=round(retreat * 100, 2), 确认="今天MA5重新上穿MA10")

    # 黄金坑：平台→短期下破→今天收复平台下沿或MA20。
    platform, pit = close.iloc[-46:-21], close.iloc[-21:-1]
    pit_depth = 1 - float(pit.min() / platform.median())
    reclaim = ((previous.close <= ma[20].iloc[-2] and current.close > ma[20].iloc[-1])
               or (previous.close <= platform.min() and current.close > platform.min()))
    bottom_position = int(pit.to_numpy().argmin())
    if (bullish and ratio >= 1.3 and platform.max() / platform.min() <= 1.18
            and .08 <= pit_depth <= .30 and 4 <= bottom_position <= 17
            and current.close / pit.min() >= 1.08 and reclaim):
        emit("黄金坑", 坑深百分比=round(pit_depth * 100, 2), 确认="今天收复MA20或平台下沿")

    # 原图明确为一根大阳后小K线整理2–4周，按10–20交易日近似。
    for length in range(10, 21):
        consolidation = frame.iloc[-length - 1:-1]
        impulse = frame.iloc[-length - 2]
        impulse_gain = float(impulse.close / impulse.open - 1)
        impulse_baseline = volume.iloc[-length - 22:-length - 2].mean()
        resistance = max(consolidation.high.max(), impulse.high)
        if (bullish and ratio >= 1.3 and impulse_gain >= .05
                and impulse.volume >= 1.5 * impulse_baseline
                and consolidation.close.max() / consolidation.close.min() <= 1.12
                and consolidation.low.min() >= .99 * impulse.open
                and consolidation.volume.mean() <= .8 * impulse.volume
                and (consolidation.close.to_numpy() >= ma[20].iloc[-length - 1:-1].to_numpy()).mean() >= .8
                and current.close > resistance):
            emit("空中加油", 整理天数=length, 起始大阳实体百分比=round(impulse_gain * 100, 2),
                 整理振幅百分比=round((consolidation.close.max() / consolidation.close.min() - 1) * 100, 2))
            break

    # 底部红三兵，底部背景必需，普通上涨中的三阳不算。
    soldiers = frame.iloc[-3:]
    bodies = soldiers.close - soldiers.open
    ranges = soldiers.high - soldiers.low
    context = frame.iloc[-23:-3]
    bottom_context = (context.close.iloc[-1] / context.close.iloc[0] <= .92
                      or context.close.iloc[-1] <= context.low.min() + .3 * (context.high.max() - context.low.min()))
    opens_inside = all(
        soldiers.open.iloc[i] >= soldiers.open.iloc[i - 1] * .995
        and soldiers.open.iloc[i] <= soldiers.close.iloc[i - 1] * 1.005 for i in (1, 2)
    )
    if (bottom_context and (bodies / soldiers.open >= .008).all()
            and (bodies / ranges.replace(0, np.nan) >= .5).all()
            and ((soldiers.high - soldiers.close) <= .5 * bodies).all()
            and soldiers.close.is_monotonic_increasing and opens_inside):
        emit("底部红三兵", 三日涨幅百分比=round((current.close / soldiers.open.iloc[0] - 1) * 100, 2),
             确认="底部背景、连续三阳、收盘递升")

    # 杯柄：持续U形杯身+浅柄，今天首次突破预先确定的杯沿。
    cup_found = False
    for width in (60, 90, 120, 150):
        for handle_length in (5, 10, 15, 20):
            if len(frame) < width + handle_length + 1:
                continue
            cup = frame.iloc[-width - handle_length - 1:-handle_length - 1]
            handle = frame.iloc[-handle_length - 1:-1]
            curved, r2, minimum, depth, dwell = _curve(cup.close)
            left, right = float(cup.close.iloc[:5].mean()), float(cup.close.iloc[-5:].mean())
            handle_depth = 1 - float(handle.low.min() / right)
            resistance = float(max(cup.high.iloc[:5].max(), cup.high.iloc[-5:].max()))
            if (bullish and ratio >= 1.3 and curved and r2 >= .60 and .2 <= minimum <= .65
                    and .12 <= depth <= .35 and dwell >= 10 and abs(right / left - 1) <= .08
                    and .03 <= handle_depth <= .12 and handle.low.min() >= (cup.low.min() + left) / 2
                    and handle.volume.mean() <= .8 * cup.volume.iloc[-20:].mean()
                    and previous.close <= resistance < current.close):
                emit("杯柄形态", 杯身天数=width, 柄天数=handle_length,
                     杯深百分比=round(depth * 100, 2), 柄回调百分比=round(handle_depth * 100, 2),
                     弧形拟合R2=round(r2, 2), 突破价=round(resistance, 3))
                cup_found = True
                break
        if cup_found:
            break

    # 一阳穿多线：昨日四线收敛，今天实体从线下穿到线上。
    yesterday_mas = [series.iloc[-2] for series in ma.values()]
    today_mas = [series.iloc[-1] for series in ma.values()]
    candle_range = current.high - current.low
    if (bullish and ratio >= 1.5 and body_return >= .03
            and max(yesterday_mas) / min(yesterday_mas) <= 1.05
            and current.open < min(yesterday_mas) and current.close > max(today_mas)
            and candle_range > 0 and (current.close - current.low) / candle_range >= .75):
        emit("一阳穿多线", 穿越均线="MA5/10/20/30", 阳线实体百分比=round(body_return * 100, 2),
             昨日均线离散百分比=round((max(yesterday_mas) / min(yesterday_mas) - 1) * 100, 2))

    # 多方炮：第二根缩量阴线基本在第一根阳线实体内，第三根突破。
    first, middle, last = (frame.iloc[index] for index in (-3, -2, -1))
    first_body = first.close - first.open
    middle_body = middle.open - middle.close
    if (first_body / first.open >= .02 and middle_body > 0
            and middle_body <= .5 * first_body and middle.close >= first.open * .995
            and middle.open <= first.close * 1.005 and middle.volume <= .8 * first.volume
            and body_return >= .02 and last.close > first.high and last.volume >= 1.2 * first.volume):
        emit("多方炮", 第三日量对第一日=round(last.volume / first.volume, 2),
             确认="两阳夹一阴且第三阳突破第一阳最高价")

    # 圆弧底：仅拟合截至昨日的价格，今日突破不会改变历史弧度。
    for width in (60, 90, 120):
        curve = frame.iloc[-width - 1:-1]
        curved, r2, minimum, depth, dwell = _curve(curve.close)
        resistance = float(curve.high.iloc[:5].max())
        denominator = resistance - curve.close.min()
        recovered = float((previous.close - curve.close.min()) / denominator) if denominator > 0 else 0
        if (bullish and ratio >= 1.3 and curved and r2 >= .6 and .2 <= minimum <= .65
                and .12 <= depth <= .50 and dwell >= 10 and recovered >= .8
                and previous.close <= resistance < current.close):
            emit("圆弧底", 形态天数=width, 弧形拟合R2=round(r2, 2),
                 底部持续天数=dwell, 突破价=round(resistance, 3))
            break
    if include_extensions:
        extension = _inspect_volume_breakout(frame, ma)
        if extension["matched"]:
            emit(VOLUME_BREAKOUT_NAME, **extension["evidence"])
    return matches
