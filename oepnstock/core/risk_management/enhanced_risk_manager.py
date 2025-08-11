"""
강화된 리스크 관리 시스템
소액 투자자를 위한 다층 안전망
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

from ..config.realistic_targets import RiskLimits, realistic_targets

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """리스크 수준 분류"""
    SAFE = "안전"
    CAUTION = "주의"
    WARNING = "경고"
    DANGER = "위험"
    EMERGENCY = "비상"


@dataclass
class TradeRecord:
    """거래 기록"""
    date: datetime
    symbol: str
    action: str  # 'buy' or 'sell'
    price: float
    quantity: int
    pnl: float = 0.0
    is_win: bool = False


@dataclass
class RiskStatus:
    """현재 리스크 상태"""
    level: RiskLevel
    daily_pnl: float
    consecutive_losses: int
    monthly_drawdown: float
    can_trade: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class EnhancedRiskManager:
    """강화된 리스크 관리자"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limits = RiskLimits()
        self.targets = realistic_targets
        
        # 거래 기록
        self.trade_history: List[TradeRecord] = []
        self.daily_pnl_history: Dict[str, float] = {}  # date -> pnl
        
        # 리스크 추적
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.current_positions: Dict[str, Dict] = {}  # symbol -> position_info
        
        # 월간 추적
        self.monthly_pnl: Dict[str, float] = {}  # YYYY-MM -> pnl
        
        logger.info(f"Enhanced Risk Manager initialized with capital: {initial_capital:,.0f}원")
    
    def check_trade_safety(self, symbol: str, action: str, amount: float,
                          market_score: int, vix: float = 20) -> Tuple[bool, List[str]]:
        """거래 안전성 검사"""
        warnings = []
        
        # 1. 일일 손실 한도 검사
        today = datetime.now().strftime('%Y-%m-%d')
        daily_pnl = self.daily_pnl_history.get(today, 0.0)
        daily_loss_ratio = daily_pnl / self.current_capital
        
        if daily_loss_ratio <= self.risk_limits.daily_max_loss:
            warnings.append(f"일일 손실 한도 도달: {daily_loss_ratio:.2%}")
            return False, warnings
        
        # 2. 연속 손실 검사
        if self.consecutive_losses >= self.risk_limits.consecutive_loss_limit:
            warnings.append(f"연속 손실 {self.consecutive_losses}회 도달")
            return False, warnings
        
        # 3. 시장 상황 검사
        if market_score < self.risk_limits.market_score_minimum:
            warnings.append(f"Market Score 부족: {market_score} < {self.risk_limits.market_score_minimum}")
            return False, warnings
        
        if vix >= self.risk_limits.vix_threshold:
            warnings.append(f"고변동성 시장: VIX {vix} >= {self.risk_limits.vix_threshold}")
            return False, warnings
        
        # 4. 포지션 크기 검사
        if action == 'buy':
            position_ratio = amount / self.current_capital
            if position_ratio > 0.25:  # 최대 25% 제한
                warnings.append(f"포지션 크기 초과: {position_ratio:.1%} > 25%")
                return False, warnings
        
        # 5. 월간 손실 한도 검사
        current_month = datetime.now().strftime('%Y-%m')
        monthly_pnl = self.monthly_pnl.get(current_month, 0.0)
        monthly_loss_ratio = monthly_pnl / self.initial_capital
        
        if monthly_loss_ratio <= self.risk_limits.monthly_drawdown:
            warnings.append(f"월간 손실 한도 도달: {monthly_loss_ratio:.2%}")
            return False, warnings
        
        return True, warnings
    
    def record_trade(self, symbol: str, action: str, price: float, 
                    quantity: int, pnl: float = 0.0):
        """거래 기록"""
        trade = TradeRecord(
            date=datetime.now(),
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            pnl=pnl,
            is_win=(pnl > 0) if pnl != 0 else False
        )
        
        self.trade_history.append(trade)
        
        # 일일 PnL 업데이트
        today = trade.date.strftime('%Y-%m-%d')
        self.daily_pnl_history[today] = self.daily_pnl_history.get(today, 0.0) + pnl
        
        # 월간 PnL 업데이트
        month = trade.date.strftime('%Y-%m')
        self.monthly_pnl[month] = self.monthly_pnl.get(month, 0.0) + pnl
        
        # 자본 업데이트
        self.current_capital += pnl
        
        # 연속 손익 추적
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # 포지션 업데이트
        if action == 'buy':
            self.current_positions[symbol] = {
                'quantity': quantity,
                'avg_price': price,
                'entry_date': trade.date,
                'unrealized_pnl': 0.0
            }
        elif action == 'sell' and symbol in self.current_positions:
            del self.current_positions[symbol]
        
        logger.info(f"Trade recorded: {action} {symbol} {quantity}주 @{price:,.0f}원, PnL: {pnl:,.0f}원")
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """개별 포지션 손절 검사"""
        if symbol not in self.current_positions:
            return False
        
        position = self.current_positions[symbol]
        entry_price = position['avg_price']
        loss_ratio = (current_price - entry_price) / entry_price
        
        if loss_ratio <= self.risk_limits.position_stop_loss:
            logger.warning(f"Stop loss triggered for {symbol}: {loss_ratio:.2%}")
            return True
        
        return False
    
    def get_current_risk_status(self) -> RiskStatus:
        """현재 리스크 상태 평가"""
        today = datetime.now().strftime('%Y-%m-%d')
        daily_pnl = self.daily_pnl_history.get(today, 0.0)
        daily_ratio = daily_pnl / self.current_capital
        
        current_month = datetime.now().strftime('%Y-%m')
        monthly_pnl = self.monthly_pnl.get(current_month, 0.0)
        monthly_ratio = monthly_pnl / self.initial_capital
        
        # 리스크 수준 결정
        risk_level = self._calculate_risk_level(daily_ratio, monthly_ratio)
        
        # 거래 가능 여부
        can_trade = (
            daily_ratio > self.risk_limits.daily_max_loss and
            self.consecutive_losses < self.risk_limits.consecutive_loss_limit and
            monthly_ratio > self.risk_limits.monthly_drawdown
        )
        
        # 경고 및 권장사항
        warnings, recommendations = self._generate_warnings_and_recommendations(
            daily_ratio, monthly_ratio, risk_level
        )
        
        return RiskStatus(
            level=risk_level,
            daily_pnl=daily_pnl,
            consecutive_losses=self.consecutive_losses,
            monthly_drawdown=monthly_ratio,
            can_trade=can_trade,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _calculate_risk_level(self, daily_ratio: float, monthly_ratio: float) -> RiskLevel:
        """리스크 수준 계산"""
        # 비상: 월간 손실 한도 초과
        if monthly_ratio <= self.risk_limits.monthly_drawdown:
            return RiskLevel.EMERGENCY
        
        # 위험: 일일 손실 한도 도달 또는 연속 손실
        if (daily_ratio <= self.risk_limits.daily_max_loss or
            self.consecutive_losses >= self.risk_limits.consecutive_loss_limit):
            return RiskLevel.DANGER
        
        # 경고: 일일 손실 1.5% 이상 또는 연속 손실 2회
        if daily_ratio <= -0.015 or self.consecutive_losses >= 2:
            return RiskLevel.WARNING
        
        # 주의: 일일 손실 1% 이상 또는 월간 손실 3% 이상
        if daily_ratio <= -0.01 or monthly_ratio <= -0.03:
            return RiskLevel.CAUTION
        
        return RiskLevel.SAFE
    
    def _generate_warnings_and_recommendations(self, daily_ratio: float, 
                                             monthly_ratio: float, 
                                             risk_level: RiskLevel) -> Tuple[List[str], List[str]]:
        """경고 및 권장사항 생성"""
        warnings = []
        recommendations = []
        
        if risk_level == RiskLevel.EMERGENCY:
            warnings.append("월간 손실 한도 초과 - 즉시 거래 중단")
            recommendations.append("5일간 거래 중단 후 전략 재검토")
        
        elif risk_level == RiskLevel.DANGER:
            warnings.append("위험 수준 도달 - 신규 거래 금지")
            recommendations.append("기존 포지션 정리 고려")
        
        elif risk_level == RiskLevel.WARNING:
            warnings.append("손실 확대 중 - 포지션 축소 권장")
            recommendations.append("진입 기준 상향 조정")
        
        elif risk_level == RiskLevel.CAUTION:
            warnings.append("주의 필요 - 신중한 거래")
            recommendations.append("포지션 크기 20% 축소")
        
        # 연속 손실 관련
        if self.consecutive_losses >= 2:
            recommendations.append(f"연속 손실 {self.consecutive_losses}회 - 전략 점검")
        
        # 월간 성과 관련
        if monthly_ratio > 0.05:  # 월 5% 초과 수익
            recommendations.append("목표 초과 달성 - 보수적 전환 고려")
        
        return warnings, recommendations
    
    def get_position_sizing_recommendation(self, capital: float, 
                                         current_risk_level: RiskLevel) -> float:
        """리스크 수준에 따른 포지션 크기 권장"""
        base_ratio = 0.25  # 기본 25%
        
        if current_risk_level == RiskLevel.EMERGENCY:
            return 0.0  # 거래 중단
        elif current_risk_level == RiskLevel.DANGER:
            return base_ratio * 0.5  # 50% 축소
        elif current_risk_level == RiskLevel.WARNING:
            return base_ratio * 0.7  # 30% 축소
        elif current_risk_level == RiskLevel.CAUTION:
            return base_ratio * 0.8  # 20% 축소
        else:
            return base_ratio  # 정상
    
    def should_pause_trading(self) -> Tuple[bool, str]:
        """거래 중단 여부 판단"""
        status = self.get_current_risk_status()
        
        if status.level == RiskLevel.EMERGENCY:
            return True, "월간 손실 한도 초과로 인한 거래 중단"
        
        if status.level == RiskLevel.DANGER:
            return True, "위험 수준 도달로 인한 거래 중단"
        
        if self.consecutive_losses >= 3:
            return True, f"연속 손실 {self.consecutive_losses}회로 인한 거래 중단"
        
        return False, ""
    
    def get_daily_summary(self) -> Dict:
        """일일 요약 리포트"""
        today = datetime.now().strftime('%Y-%m-%d')
        daily_pnl = self.daily_pnl_history.get(today, 0.0)
        daily_ratio = daily_pnl / self.current_capital
        
        status = self.get_current_risk_status()
        
        return {
            "date": today,
            "daily_pnl": daily_pnl,
            "daily_return": daily_ratio,
            "current_capital": self.current_capital,
            "risk_level": status.level.value,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "active_positions": len(self.current_positions),
            "can_trade": status.can_trade,
            "target_achieved": daily_ratio >= self.targets.daily_target_min,
            "warnings": status.warnings,
            "recommendations": status.recommendations
        }