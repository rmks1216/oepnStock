"""
백테스트 신호 규칙 중 네트워크 없이 계산할 수 있는 부분.

SimpleBacktester(examples/backtesting_example.py)가 쓴다.
"""

from __future__ import annotations

import pandas as pd


def count_recent_up_days(closes: pd.Series, window: int) -> int:
    """최근 window 개의 일간 변화 중 종가가 오른 날 수를 센다.

    보합(변화 0)은 상승으로 세지 않는다.

    Args:
        closes: 날짜순 종가.
        window: 볼 일간 변화 개수.

    Returns:
        상승일 수. 변화가 window 보다 적으면 있는 변화만 센다.
    """
    changes = closes.diff().dropna().tail(window)
    return int((changes > 0).sum())


def validate_signal_settings(backtest_config) -> None:
    """신호 설정이 성립 가능한 조합인지 확인한다.

    min_recent_up_days 가 recent_up_window 보다 크면 매수 조건이 절대 성립하지 않는다.
    이전에는 이 상태로 백테스트가 돌아 거래 0회 결과가 조용히 나왔다.

    Args:
        backtest_config: min_recent_up_days, recent_up_window 를 가진 설정 객체.

    Raises:
        ValueError: 조건이 성립할 수 없는 설정일 때.
    """
    window = backtest_config.recent_up_window
    required = backtest_config.min_recent_up_days
    if required > window:
        raise ValueError(
            f"min_recent_up_days({required}) 가 recent_up_window({window}) 보다 큽니다. "
            "최근 상승일 조건이 절대 성립하지 않습니다."
        )
