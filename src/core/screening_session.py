"""Bound the overnight exception to the immediately preceding A-share session."""
from datetime import date, datetime, time, timedelta

from src.core.trading_calendar import MarketPhase, get_market_now, infer_market_phase


def is_delayed_schedule(now: datetime, as_of: date, event_name: str) -> bool:
    """Only a schedule delayed past midnight and before 09:00 may use yesterday.

    The calendar phase check fails closed, unlike is_market_open's legacy
    fail-open behavior. No manual or multi-day catch-up is enabled here.
    """
    local_now = get_market_now("cn", current_time=now)
    if (event_name != "schedule" or local_now.time().replace(tzinfo=None) >= time(9)
            or as_of != local_now.date() - timedelta(days=1)):
        return False
    prior_evening = datetime.combine(as_of, time(23, 59), tzinfo=local_now.tzinfo)
    return infer_market_phase("cn", current_time=prior_evening) == MarketPhase.POSTMARKET


def validate_completed_session(value: str, now: datetime, event_name: str) -> date:
    """Validate the workflow's explicit delayed-session handoff again at analysis."""
    session = date.fromisoformat(value)
    if value != session.isoformat() or not is_delayed_schedule(now, session, event_name):
        raise ValueError("延迟运行仅接受定时任务次日09:00前的上一自然交易日")
    return session
