"""
Main configuration settings for oepnStock trading system
"""
from typing import Dict, Any
import os
from dataclasses import dataclass


@dataclass
class TradingSettings:
    """Trading strategy configuration"""
    # Market scoring thresholds
    market_score_threshold: int = 70
    strong_uptrend_threshold: int = 80
    
    # Position management
    max_positions: int = 5
    max_single_position_ratio: float = 0.2  # 20%
    max_sector_exposure: float = 0.4  # 40%
    min_cash_ratio: float = 0.1  # 10%
    
    # Risk management
    initial_risk_per_trade: float = 0.02  # 2% rule for first 100 trades
    kelly_threshold_trades: int = 100  # Switch to Kelly after 100 trades
    max_correlation_exposure: float = 0.6
    max_daily_loss_ratio: float = 0.05  # 5% maximum daily loss
    
    # Signal strength thresholds
    immediate_buy_threshold: int = 80
    split_entry_threshold: int = 60
    min_signal_threshold: int = 60
    
    # Time constraints
    max_holding_days: int = 3
    lunch_break_start: str = "11:00"
    lunch_break_end: str = "13:00"
    
    # Korean market specifics
    min_market_cap_krw: int = 50000000000  # 500억 KRW
    sector_overheating_return: float = 0.20  # 20%
    sector_overheating_volume_ratio: float = 3.0
    
    # Gap trading thresholds
    major_gap_threshold: float = 0.03  # 3%
    minor_gap_threshold: float = 0.01  # 1%


@dataclass 
class TechnicalSettings:
    """Technical analysis configuration"""
    # Moving averages
    ma_short: int = 5
    ma_medium: int = 20
    ma_long: int = 60
    
    # RSI settings
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    
    # Volume analysis
    volume_ma_period: int = 20
    volume_spike_threshold: float = 1.5  # 150% of average
    
    # Support/resistance detection
    support_touch_threshold: int = 3
    resistance_touch_threshold: int = 3
    support_cluster_tolerance: float = 0.01  # 1%
    
    # ATR for volatility
    atr_period: int = 14
    low_volatility_threshold: float = 0.015
    high_volatility_threshold: float = 0.035


@dataclass
class BacktestSettings:
    """Backtesting configuration"""
    # Basic settings
    initial_capital: int = 10000000  # 1000만원
    rebalance_frequency: int = 5     # 5일마다 리밸런싱
    
    # Walk-forward analysis
    training_window_days: int = 252  # 1 year
    testing_window_days: int = 63   # 3 months
    step_size_days: int = 63
    
    # Market scenarios
    bull_market_period: tuple = ("2020-04-01", "2021-06-30")
    bear_market_period: tuple = ("2022-01-01", "2022-10-31")
    sideways_period: tuple = ("2023-01-01", "2023-06-30") 
    high_volatility_period: tuple = ("2020-03-01", "2020-04-30")
    
    # Scenario weights
    scenario_weights: Dict[str, float] = None
    
    # Test symbols (Korean stocks)
    test_symbols: list = None
    symbol_names: Dict[str, str] = None
    
    # Test periods
    default_start_date: str = "2023-01-01"
    default_end_date: str = "2023-12-31"
    
    # Signal generation parameters
    signal_ma_short: int = 5         # 단기 이평선
    signal_ma_long: int = 20         # 장기 이평선  
    signal_rsi_period: int = 14      # RSI 기간
    signal_rsi_oversold: int = 30    # RSI 과매도
    signal_rsi_overbought: int = 70  # RSI 과매수
    min_recent_up_days: int = 2      # 최근 상승일 최소 기준
    
    # Entry/Exit conditions
    ma_trend_factor: float = 1.0     # 이평선 상승 추세 기준 (MA5 > MA20)
    sell_threshold_ratio: float = 0.95  # 매도 임계값 (MA5 < MA20 * 0.95)
    
    def __post_init__(self):
        if self.scenario_weights is None:
            self.scenario_weights = {
                "bull": 0.2,
                "bear": 0.3,
                "sideways": 0.3,
                "high_volatility": 0.2
            }
        
        if self.test_symbols is None:
            self.test_symbols = ['005930', '000660', '035420', '055550', '005380']
        
        if self.symbol_names is None:
            self.symbol_names = {
                '005930': '삼성전자',
                '000660': 'SK하이닉스', 
                '035420': 'NAVER',
                '055550': '신한지주',
                '005380': '현대차'
            }


@dataclass
class TradingCosts:
    """Korean market trading costs"""
    commission_buy: float = 0.00015   # 0.015%
    commission_sell: float = 0.00015  # 0.015%
    tax: float = 0.0023              # 0.23% (매도시)
    slippage_market: float = 0.002   # 0.2% (시장가)
    slippage_limit: float = 0.001    # 0.1% (지정가)


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.trading = TradingSettings()
        self.technical = TechnicalSettings()
        self.backtest = BacktestSettings()
        self.costs = TradingCosts()
        
        # Database settings
        self.db_url = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/oepnstock")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # API keys (to be set via environment variables)
        self.api_keys = {
            "kiwoom": os.getenv("KIWOOM_API_KEY"),
            "korea_investment": os.getenv("KOREA_INVESTMENT_API_KEY"),
            "data_provider": os.getenv("DATA_PROVIDER_API_KEY")
        }
        
        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "logs/oepnstock.log")
        
        # Safety settings
        self.enable_paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.max_daily_loss_ratio = float(os.getenv("MAX_DAILY_LOSS", "0.05"))  # 5%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "trading": self.trading.__dict__,
            "technical": self.technical.__dict__,
            "backtest": self.backtest.__dict__,
            "costs": self.costs.__dict__,
            "db_url": self.db_url,
            "redis_url": self.redis_url,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_paper_trading": self.enable_paper_trading,
            "max_daily_loss_ratio": self.max_daily_loss_ratio
        }


# Global configuration instance
config = Config()