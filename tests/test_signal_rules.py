"""
최근 상승일 조건(최근 N일 중 M일 이상 상승) 테스트.

swing 프로파일은 min_recent_up_days 가 4인데 신호 코드가 최근 3일만 봐서
매수 조건이 절대 성립하지 않았다(검증 백테스트에서 거래 0회).
"""

import pandas as pd
import pytest

from oepnstock.backtest.signal_rules import count_recent_up_days, validate_signal_settings
from oepnstock.config.settings import BacktestSettings
from oepnstock.utils.config import ConfigManager

PROFILES = ["default", "aggressive", "conservative", "scalping", "swing"]


def test_count_recent_up_days_uses_given_window():
    # 종가 변화: +, +, -, +, +, +  → 최근 5개 변화 중 상승 4개
    closes = pd.Series([100, 101, 102, 101, 102, 103, 104])
    assert count_recent_up_days(closes, window=5) == 4
    assert count_recent_up_days(closes, window=3) == 3


def test_count_recent_up_days_ignores_flat_days():
    closes = pd.Series([100, 100, 101, 101, 102])
    assert count_recent_up_days(closes, window=4) == 2


def test_count_recent_up_days_with_short_history():
    # 변화가 window 보다 적으면 있는 변화만 센다
    closes = pd.Series([100, 101])
    assert count_recent_up_days(closes, window=5) == 1


def test_validate_rejects_impossible_up_days():
    cfg = BacktestSettings()
    cfg.recent_up_window = 3
    cfg.min_recent_up_days = 4
    with pytest.raises(ValueError, match="min_recent_up_days"):
        validate_signal_settings(cfg)


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_can_satisfy_up_days_condition(profile):
    config = ConfigManager().apply_backtest_profile(profile)
    validate_signal_settings(config.backtest)
