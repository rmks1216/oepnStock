"""
Logging utilities for oepnStock trading system
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from typing import Optional
import structlog
from pathlib import Path

from ..config import config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up structured logging for the trading system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Ensure logs directory exists
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    if not _logging_initialized:
        initialize_logging()
    
    return structlog.get_logger(name)


class TradingLogger:
    """
    Specialized logger for trading activities with custom formatting
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_market_analysis(self, symbol: str, market_score: float, regime: str, tradable: bool):
        """Log market analysis results"""
        self.logger.info(
            "Market analysis completed",
            symbol=symbol,
            market_score=market_score,
            regime=regime,
            tradable=tradable,
            event_type="market_analysis"
        )
    
    def log_support_detection(self, symbol: str, support_count: int, cluster_count: int, 
                            strongest_support: Optional[float] = None):
        """Log support detection results"""
        self.logger.info(
            "Support detection completed",
            symbol=symbol,
            support_count=support_count,
            cluster_count=cluster_count,
            strongest_support=strongest_support,
            event_type="support_detection"
        )
    
    def log_signal_confirmation(self, symbol: str, signal_strength: float, 
                              signal_type: str, action: str):
        """Log signal confirmation results"""
        self.logger.info(
            "Signal confirmation completed",
            symbol=symbol,
            signal_strength=signal_strength,
            signal_type=signal_type,
            action=action,
            event_type="signal_confirmation"
        )
    
    def log_trade_execution(self, symbol: str, action: str, quantity: int, 
                          price: float, order_id: str):
        """Log trade execution"""
        self.logger.info(
            "Trade executed",
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            order_id=order_id,
            timestamp=datetime.now().isoformat(),
            event_type="trade_execution"
        )
    
    def log_risk_management(self, symbol: str, action: str, reason: str, 
                          position_size: Optional[float] = None):
        """Log risk management decisions"""
        self.logger.warning(
            "Risk management action",
            symbol=symbol,
            action=action,
            reason=reason,
            position_size=position_size,
            event_type="risk_management"
        )
    
    def log_error(self, error_type: str, error_message: str, symbol: Optional[str] = None, 
                  **kwargs):
        """Log errors with context"""
        self.logger.error(
            "System error occurred",
            error_type=error_type,
            error_message=error_message,
            symbol=symbol,
            event_type="error",
            **kwargs
        )
    
    def log_performance(self, metric_name: str, value: float, period: str = "daily"):
        """Log performance metrics"""
        self.logger.info(
            "Performance metric",
            metric_name=metric_name,
            value=value,
            period=period,
            timestamp=datetime.now().isoformat(),
            event_type="performance"
        )
    
    def log_system_health(self, component: str, status: str, details: Optional[dict] = None):
        """Log system health checks"""
        self.logger.info(
            "System health check",
            component=component,
            status=status,
            details=details or {},
            timestamp=datetime.now().isoformat(),
            event_type="health_check"
        )


# Logging initialization state
_logging_initialized = False


def initialize_logging(log_level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Initialize logging system explicitly
    
    Args:
        log_level: Logging level (defaults to config.log_level)
        log_file: Log file path (defaults to config.log_file)
    """
    global _logging_initialized
    
    if _logging_initialized:
        return
    
    setup_logging(
        log_level=log_level or config.log_level,
        log_file=log_file or config.log_file
    )
    
    _logging_initialized = True


def is_logging_initialized() -> bool:
    """Check if logging has been initialized"""
    return _logging_initialized


# Convenience function for getting trading logger
def get_trading_logger(name: str) -> TradingLogger:
    """Get a specialized trading logger instance"""
    if not _logging_initialized:
        initialize_logging()
    
    return TradingLogger(name)