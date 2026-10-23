"""每日自动选股的模型否决层；普通手动分析仍可报告看空结论。"""
import os

from src.report_language import infer_decision_type_from_advice, is_bearish_trend


def rejection_reason(result):
    if os.getenv("PATTERN_SCREEN_ACTIVE", "").lower() != "true":
        return None
    if is_bearish_trend(result.trend_prediction):
        return "模型趋势看空"
    if (str(getattr(result, "decision_type", "")).lower() == "sell"
            or infer_decision_type_from_advice(result.operation_advice) == "sell"):
        return "模型建议减仓或卖出"
    return None
