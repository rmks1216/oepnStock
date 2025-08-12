"""
모바일 API 데이터 모델
Pydantic 모델 정의
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

__all__ = [
    # Enums
    "UserRole",
    "AlertLevel",
    
    # Authentication models
    "LoginRequest",
    "LoginResponse", 
    "RefreshTokenRequest",
    "User",
    
    # Dashboard models
    "DashboardOverview",
    "Position",
    "Trade",
    "Alert",
    "PerformanceSummary",
    
    # Trading control models
    "TradingControlRequest",
    "TradingControlResponse",
    
    # Settings models
    "NotificationSettings",
    "RiskSettings",
    
    # Chart models
    "ChartDataRequest",
    "ChartDataPoint",
    "ChartData",
    
    # WebSocket models
    "WebSocketMessage",
    "LiveUpdate",
    
    # API response models
    "APIResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    
    # Filter models
    "TradeFilter",
    "AlertFilter",
    
    # Statistics models
    "TradingStatistics",
    "RiskMetrics",
    
    # System models
    "SystemStatus",
    "HealthCheck"
]


class UserRole(str, Enum):
    """사용자 권한"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class AlertLevel(str, Enum):
    """알림 레벨"""
    EMERGENCY = "EMERGENCY"
    WARNING = "WARNING"
    INFO = "INFO"
    SUCCESS = "SUCCESS"


# === 인증 모델 ===

class LoginRequest(BaseModel):
    """로그인 요청"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class LoginResponse(BaseModel):
    """로그인 응답"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user_info: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    """토큰 갱신 요청"""
    refresh_token: str


class User(BaseModel):
    """사용자 정보"""
    id: int
    username: str
    email: Optional[EmailStr] = None
    role: UserRole = UserRole.USER
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


# === 대시보드 모델 ===

class DashboardOverview(BaseModel):
    """대시보드 개요"""
    total_asset: float = Field(..., description="총 자산")
    daily_return: float = Field(..., description="일일 수익률")
    daily_pnl: float = Field(..., description="일일 손익")
    monthly_return: float = Field(..., description="월간 수익률")
    risk_level: str = Field(..., description="리스크 레벨")
    market_score: int = Field(..., ge=0, le=100, description="시장 점수")
    positions_count: int = Field(..., ge=0, description="포지션 수")
    win_rate: float = Field(..., ge=0, le=1, description="승률")
    consecutive_losses: int = Field(..., ge=0, description="연속 손실")
    is_trading_active: bool = Field(..., description="거래 활성 상태")
    last_updated: datetime = Field(..., description="마지막 업데이트")


class Position(BaseModel):
    """포지션 정보"""
    symbol: str = Field(..., description="종목코드")
    name: Optional[str] = Field(None, description="종목명")
    quantity: int = Field(..., description="보유 수량")
    avg_price: float = Field(..., description="평균 단가")
    current_price: float = Field(..., description="현재가")
    market_value: float = Field(..., description="평가금액")
    unrealized_pnl: float = Field(..., description="평가손익")
    unrealized_pnl_pct: float = Field(..., description="수익률")
    purchase_date: datetime = Field(..., description="매수일")
    days_held: int = Field(..., description="보유일수")


class Trade(BaseModel):
    """거래 내역"""
    id: int
    date: datetime = Field(..., description="거래일시")
    symbol: str = Field(..., description="종목코드")
    name: Optional[str] = Field(None, description="종목명")
    action: str = Field(..., description="거래구분 (buy/sell)")
    quantity: int = Field(..., description="거래수량")
    price: float = Field(..., description="거래가격")
    trade_value: float = Field(..., description="거래금액")
    costs: float = Field(..., description="수수료")
    pnl: Optional[float] = Field(None, description="손익")
    reason: str = Field(..., description="거래사유")


class Alert(BaseModel):
    """알림 정보"""
    id: int
    timestamp: datetime = Field(..., description="발생시간")
    level: AlertLevel = Field(..., description="알림레벨")
    title: str = Field(..., description="제목")
    message: str = Field(..., description="내용")
    is_read: bool = Field(False, description="읽음여부")
    data: Optional[Dict[str, Any]] = Field(None, description="추가데이터")


class PerformanceSummary(BaseModel):
    """성과 요약"""
    period_start: datetime = Field(..., description="시작일")
    period_end: datetime = Field(..., description="종료일")
    total_return: float = Field(..., description="총 수익률")
    annual_return: float = Field(..., description="연간 수익률")
    volatility: float = Field(..., description="변동성")
    sharpe_ratio: float = Field(..., description="샤프 비율")
    max_drawdown: float = Field(..., description="최대 낙폭")
    win_rate: float = Field(..., description="승률")
    total_trades: int = Field(..., description="총 거래수")
    profit_factor: float = Field(..., description="수익팩터")


# === 제어 모델 ===

class TradingControlRequest(BaseModel):
    """거래 제어 요청"""
    action: str = Field(..., regex="^(pause|resume|stop)$")
    duration_hours: Optional[int] = Field(None, ge=1, le=24, description="일시정지 시간")
    reason: Optional[str] = Field(None, max_length=200, description="사유")


class TradingControlResponse(BaseModel):
    """거래 제어 응답"""
    status: str
    message: str
    active_until: Optional[datetime] = None


# === 설정 모델 ===

class NotificationSettings(BaseModel):
    """알림 설정"""
    telegram_enabled: bool = Field(True, description="텔레그램 알림")
    email_enabled: bool = Field(True, description="이메일 알림")
    push_enabled: bool = Field(True, description="푸시 알림")
    daily_report: bool = Field(True, description="일일 리포트")
    risk_alerts: bool = Field(True, description="리스크 알림")
    trade_notifications: bool = Field(True, description="거래 알림")


class RiskSettings(BaseModel):
    """리스크 설정"""
    daily_loss_limit: float = Field(-0.02, ge=-0.1, le=0, description="일일 손실 한도")
    monthly_drawdown_limit: float = Field(-0.1, ge=-0.3, le=0, description="월간 드로다운 한도")
    max_positions: int = Field(5, ge=1, le=10, description="최대 포지션 수")
    position_size_limit: float = Field(0.2, ge=0.05, le=0.5, description="포지션 크기 한도")
    consecutive_loss_limit: int = Field(3, ge=1, le=10, description="연속 손실 한도")


# === 차트 모델 ===

class ChartDataRequest(BaseModel):
    """차트 데이터 요청"""
    chart_type: str = Field(..., regex="^(equity|returns|drawdown|positions)$")
    period: str = Field("1M", regex="^(1D|1W|1M|3M|6M|1Y|ALL)$")
    symbol: Optional[str] = Field(None, description="특정 종목 (옵션)")


class ChartDataPoint(BaseModel):
    """차트 데이터 포인트"""
    timestamp: datetime
    value: float
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChartData(BaseModel):
    """차트 데이터"""
    title: str
    chart_type: str
    data_points: List[ChartDataPoint]
    y_axis_label: str
    period: str
    last_updated: datetime


# === 웹소켓 모델 ===

class WebSocketMessage(BaseModel):
    """웹소켓 메시지"""
    type: str = Field(..., description="메시지 타입")
    data: Dict[str, Any] = Field(..., description="데이터")
    timestamp: datetime = Field(default_factory=datetime.now)


class LiveUpdate(BaseModel):
    """실시간 업데이트"""
    current_capital: float
    daily_return: float
    daily_pnl: float
    positions_count: int
    risk_level: str
    market_score: int
    last_trade: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# === 응답 래퍼 모델 ===

class APIResponse(BaseModel):
    """API 응답 래퍼"""
    success: bool = True
    message: str = "Success"
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """오류 응답"""
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# === 페이징 모델 ===

class PaginationParams(BaseModel):
    """페이징 파라미터"""
    page: int = Field(1, ge=1, description="페이지 번호")
    limit: int = Field(20, ge=1, le=100, description="페이지 크기")


class PaginatedResponse(BaseModel):
    """페이징된 응답"""
    items: List[Any]
    total: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool


# === 필터 모델 ===

class TradeFilter(BaseModel):
    """거래 필터"""
    symbol: Optional[str] = None
    action: Optional[str] = Field(None, regex="^(buy|sell)$")
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_pnl: Optional[float] = None
    max_pnl: Optional[float] = None


class AlertFilter(BaseModel):
    """알림 필터"""
    level: Optional[AlertLevel] = None
    is_read: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


# === 통계 모델 ===

class TradingStatistics(BaseModel):
    """거래 통계"""
    period: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    avg_holding_period: float  # days
    
    
class RiskMetrics(BaseModel):
    """리스크 지표"""
    period: str
    var_95: float = Field(..., description="95% VaR")
    cvar_95: float = Field(..., description="95% CVaR")
    max_drawdown: float = Field(..., description="최대 드로다운")
    volatility: float = Field(..., description="변동성")
    beta: float = Field(..., description="베타")
    alpha: float = Field(..., description="알파")
    sharpe_ratio: float = Field(..., description="샤프 비율")
    sortino_ratio: float = Field(..., description="소르티노 비율")
    calmar_ratio: float = Field(..., description="칼마 비율")


# === 시스템 모델 ===

class SystemStatus(BaseModel):
    """시스템 상태"""
    is_running: bool = Field(..., description="시스템 실행 상태")
    is_trading_active: bool = Field(..., description="거래 활성 상태")
    last_heartbeat: datetime = Field(..., description="마지막 하트비트")
    uptime_seconds: int = Field(..., description="가동 시간")
    cpu_usage: float = Field(..., description="CPU 사용률")
    memory_usage: float = Field(..., description="메모리 사용률")
    disk_usage: float = Field(..., description="디스크 사용률")
    market_connection: bool = Field(..., description="시장 데이터 연결")
    broker_connection: bool = Field(..., description="브로커 연결")


class HealthCheck(BaseModel):
    """헬스 체크"""
    status: str = Field("healthy", regex="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=datetime.now)
    checks: Dict[str, bool] = Field(default_factory=dict)
    message: Optional[str] = None