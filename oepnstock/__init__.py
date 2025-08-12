"""
oepnStock - Korean Stock Market Trading System
4-Stage Checklist Strategy Implementation

Author: Claude Code
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude Code"
__description__ = "Korean Stock Market Trading System with 4-Stage Checklist Strategy"

# Core strategy components
from .core import (
    MarketFlowAnalyzer,
    SupportDetector,
    SignalConfirmator,
    RiskManager
)

# Critical modules  
from .modules import (
    FundamentalEventFilter,
    PortfolioConcentrationManager,
    GapTradingStrategy
)

# Utility functions
from .utils import (
    get_logger,
    setup_logging,
    ConfigManager,
    MarketDataManager,
    TechnicalIndicators,
    KoreanMarketUtils
)

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

# Explicit exports
__all__ = [
    # Core strategy components
    "MarketFlowAnalyzer",
    "SupportDetector", 
    "SignalConfirmator",
    "RiskManager",
    
    # Critical modules
    "FundamentalEventFilter",
    "PortfolioConcentrationManager",
    "GapTradingStrategy",
    
    # Utility functions
    "get_logger",
    "setup_logging",
    "ConfigManager", 
    "MarketDataManager",
    "TechnicalIndicators",
    "KoreanMarketUtils",
    
    # Constants
    "STRATEGY_STAGES",
    "MODULE_PHASES",
    
    # Package metadata
    "__version__",
    "__author__",
    "__description__"
]