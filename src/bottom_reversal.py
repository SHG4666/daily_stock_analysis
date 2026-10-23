"""每日底部放量反弹规则：可复核的筛选假设，不代表收益预测。"""
import numpy as np
import pandas as pd

from src.pattern_screener import PatternSignal, ScreeningDataError, volume_ratio

PROFILE = "bottom_reversal"
RULESET_VERSION = "3.0-bottom-reversal"
SIGNAL_NAME = "长期下跌后底部放量上涨"
LIMITS = {
    "decline_sessions": 60,
    "prior_decline_max": -.15,
    "ma20_decline_max": -.08,
    "bottom_distance_max": .20,
    "daily_return_min": .02,
    "volume_ratio_min": 1.8,
    "close_position_min": .65,
    "prior_five_session_return_max": .10,
    "turnover_previous_ratio_min": 1.2,
    "turnover_five_mean_ratio_min": 1.5,
}


def inspect_bottom_reversal(frame):
    """输入已经 prepare_bars 核验的完整日K；所有背景窗口排除当日。"""
    close = frame.close
    prior = frame.iloc[-61:-1]
    today = frame.iloc[-1]
    ma20 = close.rolling(20).mean()
    metrics = {
        "prior_decline": float(close.iloc[-2] / close.iloc[-61] - 1),
        "ma20_decline": float(ma20.iloc[-2] / ma20.iloc[-42] - 1),
        "bottom_distance": float(today.close / prior.low.min() - 1),
        "daily_return": float(today.close / close.iloc[-2] - 1),
        "volume_ratio": volume_ratio(frame),
        "close_position": float((today.close - today.low) / (today.high - today.low))
        if today.high > today.low else 0,
        "prior_five_session_return": float(close.iloc[-2] / close.iloc[-7] - 1),
    }
    checks = (
        (metrics["prior_decline"] <= LIMITS["prior_decline_max"], "此前60日跌幅不足15%"),
        (metrics["ma20_decline"] <= LIMITS["ma20_decline_max"], "缺少持续下跌背景"),
        (metrics["bottom_distance"] <= LIMITS["bottom_distance_max"], "离60日低点过远"),
        (today.close > today.open and metrics["daily_return"] >= LIMITS["daily_return_min"], "当日上涨不足2%或非阳线"),
        (metrics["volume_ratio"] >= LIMITS["volume_ratio_min"], "当日放量不足1.8倍"),
        (metrics["close_position"] >= LIMITS["close_position_min"], "收盘回落过多"),
        (metrics["prior_five_session_return"] <= LIMITS["prior_five_session_return_max"], "此前5日已涨超10%"),
    )
    return metrics, next((reason for passed, reason in checks if not passed), None)


def confirm_turnover(frame, turnover):
    """逐日精确对齐实际换手率，缺数、滞后、重复或无效数据均不能通过。"""
    try:
        values = turnover[["date", "turnover"]].copy()
        values["date"] = pd.to_datetime(values["date"], errors="raise", format="mixed").dt.normalize()
        dates = frame.date.tail(6)
        values = values[values.date <= dates.iloc[-1]]
        if values.date.duplicated().any():
            raise ValueError("duplicate turnover dates")
        aligned = values.set_index("date").reindex(dates)["turnover"]
        numbers = pd.to_numeric(aligned, errors="coerce").to_numpy(dtype=float)
        if len(numbers) != 6 or not np.isfinite(numbers).all() or (numbers <= 0).any():
            raise ValueError("missing or invalid turnover")
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ScreeningDataError("turnover_data") from exc
    current, previous, mean = numbers[-1], numbers[-2], numbers[:-1].mean()
    metrics = {"turnover": float(current), "previous_turnover": float(previous),
               "five_day_turnover_mean": float(mean),
               "turnover_previous_ratio": float(current / previous),
               "turnover_five_mean_ratio": float(current / mean)}
    passed = (current / previous >= LIMITS["turnover_previous_ratio_min"]
              and current / mean >= LIMITS["turnover_five_mean_ratio_min"])
    return metrics, passed


def reversal_signal(metrics):
    return PatternSignal(SIGNAL_NAME, {
        "此前60日跌幅": f"{metrics['prior_decline']:.1%}",
        "20日均线40日变化": f"{metrics['ma20_decline']:.1%}",
        "距离60日低点": f"{metrics['bottom_distance']:.1%}",
        "当日涨幅": f"{metrics['daily_return']:.1%}",
        "当日/此前20日均量": round(metrics["volume_ratio"], 2),
        "换手率": f"{metrics['turnover']:.2f}%",
        "较上一日换手率": f"{metrics['turnover_previous_ratio']:.2f}倍",
        "较此前5日平均换手率": f"{metrics['turnover_five_mean_ratio']:.2f}倍",
    })
