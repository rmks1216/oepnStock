"""
oepnStock 알림 시스템
실시간 텔레그램, 이메일 알림 및 종합 리포트 발송
"""

from .telegram_notifier import TelegramNotifier, TelegramMessage
from .email_notifier import EmailNotifier
from .alert_manager import AlertManager, AlertType, AlertRule, AlertHistory

__all__ = [
    'TelegramNotifier',
    'TelegramMessage', 
    'EmailNotifier',
    'AlertManager',
    'AlertType',
    'AlertRule',
    'AlertHistory'
]

# 버전 정보
__version__ = '1.0.0'