"""
oepnStock - Korean Stock Market Trading System
4-Stage Checklist Strategy Implementation

Author: Claude Code
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude Code"
__description__ = "Korean Stock Market Trading System with 4-Stage Checklist Strategy"

from .core import *
from .modules import *
from .utils import *

# Core strategy stages
STRATEGY_STAGES = [
    "stage1_market_flow",      # 시장의 흐름 파악
    "stage2_support_detection", # 매수 후보 지점 찾기  
    "stage3_signal_confirmation", # 진짜 반등 신호 확인
    "stage4_risk_management"    # 리스크 관리 계획
]

# Module implementation phases
MODULE_PHASES = {
    "critical": ["fundamental_filter", "portfolio_concentration", "gap_strategy"],
    "performance": ["volatility_adaptive", "order_book", "correlation_risk"], 
    "advanced": ["ml_validator", "intraday_time"]
}