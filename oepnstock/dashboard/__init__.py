"""
oepnStock 대시보드 모듈

웹 기반 실시간 모니터링 및 시각화 시스템
"""

from .web_dashboard import WebDashboard
from .data_manager import DashboardDataManager

__all__ = [
    'WebDashboard',
    'DashboardDataManager'
]