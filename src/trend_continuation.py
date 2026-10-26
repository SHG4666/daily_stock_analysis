"""上涨后缩量整理、回踩支撑并再次突破的日线结构。"""
import numpy as np

SIGNAL_NAME = "上涨后缩量回踩再突破"
LIMITS = {
    "base_sessions_min": 4,
    "base_sessions_max": 12,
    "impulse_sessions": 20,
    "impulse_gain_min": .15,
    "impulse_gain_max": .50,
    "pullback_min": .02,
    "pullback_max": .12,
    "base_volume_ratio_max": .80,
    "support_tolerance": .03,
    "short_support_tolerance": .02,
    "breakout_margin_min": .005,
    "breakout_margin_max": .08,
    "volume_over_base_min": 1.5,
    "ma20_deviation_max": .25,
    "total_gain_max": .60,
    "positive_open_gap_max": .05,
}


def inspect_continuation(frame):
    """输入prepare_bars核验后的日K；先找前高，再核验其后的整理，不能倒置顺序。"""
    today = frame.iloc[-1]
    start = len(frame) - LIMITS["base_sessions_max"] - 2
    window = frame.iloc[start:-1]
    peak = float(window.high.max())
    peak_index = start + int(np.flatnonzero(window.high.to_numpy() == peak)[-1])
    base = frame.iloc[peak_index+1:-1]
    base_length_ok = LIMITS["base_sessions_min"] <= len(base) <= LIMITS["base_sessions_max"]
    if not base_length_ok:
        return {"matched": False, "checks": {"整理4至12日": False}, "base_sessions": len(base)}

    impulse = frame.iloc[peak_index-LIMITS["impulse_sessions"]:peak_index+1]
    low = float(impulse.low.min())
    low_index = impulse.index[np.flatnonzero(impulse.low.to_numpy() == low)[-1]]
    ma = {window: frame.close.rolling(window).mean() for window in (5, 10, 20, 60)}
    advance = peak / low - 1
    pullback = 1 - float(base.low.min()) / peak
    contraction = float(base.volume.mean() / frame.volume.iloc[peak_index-4:peak_index+1].mean())
    breakout = float(today.close / peak - 1)
    over_base = float(today.volume / base.volume.mean())
    deviation = float(today.close / ma[20].iloc[-1] - 1)
    checks = {
        "整理4至12日": base_length_ok,
        "上涨先于整理": (low_index < peak_index and peak >= float(impulse.high.max())
                     and LIMITS["impulse_gain_min"] <= advance <= LIMITS["impulse_gain_max"]),
        "中期趋势向上": (ma[20].iloc[-2] > ma[60].iloc[-2]
                     and ma[20].iloc[-2] > ma[20].iloc[-7] and ma[60].iloc[-2] > ma[60].iloc[-7]),
        "回调幅度受控": LIMITS["pullback_min"] <= pullback <= LIMITS["pullback_max"],
        "整理缩量": contraction <= LIMITS["base_volume_ratio_max"],
        "回踩短均线": bool((base.low <= ma[10].loc[base.index] * (1+LIMITS["short_support_tolerance"])).any()),
        "守住中期支撑": bool((base.close >= ma[20].loc[base.index] * (1-LIMITS["support_tolerance"])).all()),
        "放量突破前高": (LIMITS["breakout_margin_min"] <= breakout <= LIMITS["breakout_margin_max"]
                     and over_base >= LIMITS["volume_over_base_min"]),
        "短均线向上且收盘在线上": (ma[5].iloc[-1] > ma[5].iloc[-2] and ma[10].iloc[-1] > ma[10].iloc[-2]
                          and today.close > max(series.iloc[-1] for series in ma.values())),
        "中期乖离受控": deviation <= LIMITS["ma20_deviation_max"],
        "累计上涨受控": today.close / low - 1 <= LIMITS["total_gain_max"],
        "开盘跳空受控": today.open / frame.close.iloc[-2] - 1 <= LIMITS["positive_open_gap_max"],
    }
    return {"matched": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()},
            "base_sessions": len(base), "peak_date": frame.date.iloc[peak_index].date().isoformat(),
            "peak": peak, "impulse_gain": float(advance), "pullback": float(pullback),
            "base_volume_ratio": contraction, "volume_over_base": over_base,
            "breakout_margin": breakout, "ma20_deviation": deviation}
