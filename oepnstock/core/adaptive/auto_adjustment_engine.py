"""
자동 조정 메커니즘 엔진
성과와 변동성을 기반으로 전략 파라미터를 동적 조정
"""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import numpy as np

from ..config.realistic_targets import RealisticTargets, RiskLimits
from ..risk_management import RiskLevel, EnhancedRiskManager

logger = logging.getLogger(__name__)


class AdjustmentAction(Enum):
    """조정 액션 타입"""
    MAINTAIN = "maintain"           # 현상 유지
    REDUCE_RISK = "reduce_risk"     # 리스크 축소
    INCREASE_CAUTION = "increase_caution"  # 주의 강화
    PAUSE_TRADING = "pause_trading"  # 거래 중단
    RESUME_TRADING = "resume_trading"  # 거래 재개


@dataclass
class AdjustmentParameters:
    """조정 파라미터"""
    position_size_multiplier: float = 1.0
    entry_threshold_adjustment: int = 0
    max_positions_adjustment: int = 0
    rebalance_frequency_adjustment: int = 0
    stop_loss_tightening: float = 0.0
    action: AdjustmentAction = AdjustmentAction.MAINTAIN
    reason: str = ""
    duration_days: int = 0


@dataclass  
class MarketCondition:
    """시장 상황 분석"""
    volatility: float           # VIX 또는 변동성 지수
    trend_strength: float       # 추세 강도
    market_score: int          # Market Score
    volume_activity: float     # 거래량 활성도
    sector_rotation: bool      # 섹터 로테이션 여부
    
    def get_condition_level(self) -> str:
        """시장 상황 레벨"""
        if self.volatility > 30:
            return "고변동성"
        elif self.volatility > 20:
            return "중간변동성"
        else:
            return "저변동성"


class AutoAdjustmentEngine:
    """자동 조정 엔진"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.targets = RealisticTargets()
        self.risk_limits = RiskLimits()
        
        # 성과 추적
        self.performance_history: Dict[str, Dict] = {}  # date -> metrics
        self.adjustment_history: List[Dict] = []
        
        # 현재 조정 상태
        self.current_adjustments = AdjustmentParameters()
        self.last_review_date = datetime.now()
        self.pause_until: Optional[datetime] = None
        
        logger.info("Auto Adjustment Engine initialized")
    
    def analyze_performance(self, monthly_return: float, sharpe_ratio: float,
                          win_rate: float, max_drawdown: float,
                          consecutive_losses: int) -> Dict[str, str]:
        """성과 분석"""
        analysis = {
            "monthly_performance": self._classify_monthly_performance(monthly_return),
            "risk_adjusted_performance": self._classify_sharpe_performance(sharpe_ratio),
            "consistency": self._classify_win_rate(win_rate),
            "risk_management": self._classify_drawdown(max_drawdown),
            "streak_analysis": self._classify_loss_streak(consecutive_losses)
        }
        
        # 종합 평가
        analysis["overall"] = self._get_overall_assessment(analysis)
        
        return analysis
    
    def calculate_adjustments(self, monthly_return: float, volatility: float,
                            consecutive_losses: int, current_positions: int,
                            market_condition: MarketCondition) -> AdjustmentParameters:
        """조정 파라미터 계산"""
        
        # 1. 월간 성과 기반 조정
        performance_adjustment = self._calculate_performance_adjustment(monthly_return)
        
        # 2. 변동성 기반 조정  
        volatility_adjustment = self._calculate_volatility_adjustment(volatility)
        
        # 3. 연속 손실 기반 조정
        streak_adjustment = self._calculate_streak_adjustment(consecutive_losses)
        
        # 4. 시장 상황 기반 조정
        market_adjustment = self._calculate_market_adjustment(market_condition)
        
        # 조정 통합 (가장 보수적인 것 선택)
        final_adjustment = self._integrate_adjustments([
            performance_adjustment,
            volatility_adjustment, 
            streak_adjustment,
            market_adjustment
        ])
        
        self.current_adjustments = final_adjustment
        self._log_adjustment(final_adjustment)
        
        return final_adjustment
    
    def _calculate_performance_adjustment(self, monthly_return: float) -> AdjustmentParameters:
        """월간 성과 기반 조정"""
        
        if monthly_return >= 0.05:  # 월 5% 이상 (우수)
            return AdjustmentParameters(
                position_size_multiplier=0.8,
                entry_threshold_adjustment=5,
                action=AdjustmentAction.REDUCE_RISK,
                reason="월간 목표 초과 달성으로 리스크 축소",
                stop_loss_tightening=0.005  # 손절을 더 타이트하게
            )
        
        elif 0.02 <= monthly_return < 0.05:  # 월 2~5% (양호)
            return AdjustmentParameters(
                action=AdjustmentAction.MAINTAIN,
                reason="월간 성과 양호 - 현상 유지"
            )
        
        elif 0 <= monthly_return < 0.02:  # 월 0~2% (보통)
            return AdjustmentParameters(
                position_size_multiplier=0.95,
                entry_threshold_adjustment=2,
                action=AdjustmentAction.INCREASE_CAUTION,
                reason="월간 성과 부진으로 주의 강화"
            )
        
        elif -0.03 <= monthly_return < 0:  # 월 0~-3% (주의)
            return AdjustmentParameters(
                position_size_multiplier=0.9,
                entry_threshold_adjustment=3,
                action=AdjustmentAction.INCREASE_CAUTION,
                reason="월간 손실로 인한 포지션 축소"
            )
        
        else:  # 월 -3% 이하 (위험)
            return AdjustmentParameters(
                action=AdjustmentAction.PAUSE_TRADING,
                reason="월간 손실 한도 초과로 거래 중단",
                duration_days=5
            )
    
    def _calculate_volatility_adjustment(self, volatility: float) -> AdjustmentParameters:
        """변동성 기반 조정"""
        
        if volatility >= 30:  # 고변동성
            return AdjustmentParameters(
                position_size_multiplier=0.6,
                entry_threshold_adjustment=10,
                rebalance_frequency_adjustment=2,  # 더 긴 주기
                action=AdjustmentAction.REDUCE_RISK,
                reason=f"고변동성 시장 (VIX: {volatility})"
            )
        
        elif 20 <= volatility < 30:  # 중간변동성
            return AdjustmentParameters(
                position_size_multiplier=0.8,
                entry_threshold_adjustment=5,
                action=AdjustmentAction.INCREASE_CAUTION,
                reason=f"중간변동성 시장 (VIX: {volatility})"
            )
        
        else:  # 저변동성 
            return AdjustmentParameters(
                action=AdjustmentAction.MAINTAIN,
                reason=f"저변동성 시장 (VIX: {volatility})"
            )
    
    def _calculate_streak_adjustment(self, consecutive_losses: int) -> AdjustmentParameters:
        """연속 손실 기반 조정"""
        
        if consecutive_losses >= 3:
            return AdjustmentParameters(
                action=AdjustmentAction.PAUSE_TRADING,
                reason=f"연속 손실 {consecutive_losses}회로 거래 중단",
                duration_days=consecutive_losses  # 손실 횟수만큼 일수
            )
        
        elif consecutive_losses == 2:
            return AdjustmentParameters(
                position_size_multiplier=0.7,
                entry_threshold_adjustment=8,
                action=AdjustmentAction.REDUCE_RISK,
                reason="연속 손실 2회로 리스크 축소"
            )
        
        elif consecutive_losses == 1:
            return AdjustmentParameters(
                position_size_multiplier=0.9,
                entry_threshold_adjustment=3,
                action=AdjustmentAction.INCREASE_CAUTION,
                reason="전일 손실로 주의 강화"
            )
        
        else:
            return AdjustmentParameters(
                action=AdjustmentAction.MAINTAIN,
                reason="손실 없음 - 정상 운영"
            )
    
    def _calculate_market_adjustment(self, market_condition: MarketCondition) -> AdjustmentParameters:
        """시장 상황 기반 조정"""
        
        # Market Score 기반
        if market_condition.market_score < 70:
            return AdjustmentParameters(
                action=AdjustmentAction.PAUSE_TRADING,
                reason=f"Market Score 부족 ({market_condition.market_score})",
                duration_days=1
            )
        
        # 추세 강도 기반
        if market_condition.trend_strength < 0.3:  # 약한 추세
            return AdjustmentParameters(
                position_size_multiplier=0.85,
                entry_threshold_adjustment=5,
                action=AdjustmentAction.INCREASE_CAUTION,
                reason="약한 시장 추세"
            )
        
        # 섹터 로테이션 중
        if market_condition.sector_rotation:
            return AdjustmentParameters(
                max_positions_adjustment=-1,  # 포지션 수 1개 감소
                action=AdjustmentAction.INCREASE_CAUTION,
                reason="섹터 로테이션 진행 중"
            )
        
        return AdjustmentParameters(
            action=AdjustmentAction.MAINTAIN,
            reason="시장 상황 양호"
        )
    
    def _integrate_adjustments(self, adjustments: List[AdjustmentParameters]) -> AdjustmentParameters:
        """여러 조정안을 통합 (가장 보수적으로)"""
        
        # 거래 중단이 하나라도 있으면 중단
        for adj in adjustments:
            if adj.action == AdjustmentAction.PAUSE_TRADING:
                return adj
        
        # 가장 보수적인 값들 선택
        integrated = AdjustmentParameters()
        
        # 포지션 크기는 가장 작은 값
        multipliers = [adj.position_size_multiplier for adj in adjustments 
                      if adj.position_size_multiplier != 1.0]
        if multipliers:
            integrated.position_size_multiplier = min(multipliers)
        
        # 진입 기준은 가장 높은 값
        thresholds = [adj.entry_threshold_adjustment for adj in adjustments 
                     if adj.entry_threshold_adjustment != 0]
        if thresholds:
            integrated.entry_threshold_adjustment = max(thresholds)
        
        # 액션은 가장 보수적인 것
        actions = [adj.action for adj in adjustments]
        if AdjustmentAction.REDUCE_RISK in actions:
            integrated.action = AdjustmentAction.REDUCE_RISK
        elif AdjustmentAction.INCREASE_CAUTION in actions:
            integrated.action = AdjustmentAction.INCREASE_CAUTION
        else:
            integrated.action = AdjustmentAction.MAINTAIN
        
        # 사유 통합
        reasons = [adj.reason for adj in adjustments if adj.reason]
        integrated.reason = " | ".join(reasons)
        
        return integrated
    
    def should_review_strategy(self) -> bool:
        """전략 검토 필요 여부"""
        days_since_review = (datetime.now() - self.last_review_date).days
        return days_since_review >= 30  # 30일마다 검토
    
    def is_trading_paused(self) -> Tuple[bool, str]:
        """거래 중단 상태 확인"""
        if self.pause_until is None:
            return False, ""
        
        if datetime.now() < self.pause_until:
            remaining_days = (self.pause_until - datetime.now()).days
            return True, f"거래 중단 중 (남은 기간: {remaining_days}일)"
        else:
            # 중단 기간 만료
            self.pause_until = None
            return False, "거래 중단 기간 만료 - 재개 가능"
    
    def apply_pause(self, days: int, reason: str):
        """거래 중단 적용"""
        self.pause_until = datetime.now() + timedelta(days=days)
        logger.info(f"Trading paused for {days} days: {reason}")
    
    def get_adjusted_config(self, base_config: Dict) -> Dict:
        """조정된 설정 반환"""
        if self.current_adjustments.action == AdjustmentAction.PAUSE_TRADING:
            # 거래 중단시에는 원본 설정 반환 (사용되지 않음)
            return base_config
        
        adjusted = base_config.copy()
        
        # 포지션 크기 조정
        if 'trading' in adjusted and self.current_adjustments.position_size_multiplier != 1.0:
            if 'max_single_position_ratio' in adjusted['trading']:
                original = adjusted['trading']['max_single_position_ratio']
                adjusted['trading']['max_single_position_ratio'] = \
                    original * self.current_adjustments.position_size_multiplier
        
        # 진입 기준 조정
        if 'trading' in adjusted and self.current_adjustments.entry_threshold_adjustment != 0:
            if 'market_score_threshold' in adjusted['trading']:
                adjusted['trading']['market_score_threshold'] += \
                    self.current_adjustments.entry_threshold_adjustment
        
        # 포지션 수 조정
        if 'trading' in adjusted and self.current_adjustments.max_positions_adjustment != 0:
            if 'max_positions' in adjusted['trading']:
                current_max = adjusted['trading']['max_positions']
                new_max = max(1, current_max + self.current_adjustments.max_positions_adjustment)
                adjusted['trading']['max_positions'] = new_max
        
        # 리밸런싱 주기 조정
        if 'backtest' in adjusted and self.current_adjustments.rebalance_frequency_adjustment != 0:
            if 'rebalance_frequency' in adjusted['backtest']:
                adjusted['backtest']['rebalance_frequency'] += \
                    self.current_adjustments.rebalance_frequency_adjustment
        
        return adjusted
    
    def _classify_monthly_performance(self, monthly_return: float) -> str:
        """월간 성과 분류"""
        if monthly_return >= 0.05:
            return "우수"
        elif monthly_return >= 0.02:
            return "양호"
        elif monthly_return >= 0:
            return "보통"
        elif monthly_return >= -0.03:
            return "주의"
        else:
            return "위험"
    
    def _classify_sharpe_performance(self, sharpe: float) -> str:
        """샤프 비율 성과 분류"""
        if sharpe >= 1.0:
            return "A급"
        elif sharpe >= 0.5:
            return "B급"
        elif sharpe >= 0:
            return "C급"
        else:
            return "재검토 필요"
    
    def _classify_win_rate(self, win_rate: float) -> str:
        """승률 분류"""
        if win_rate >= 0.6:
            return "우수"
        elif win_rate >= 0.55:
            return "목표 달성"
        elif win_rate >= 0.5:
            return "평균"
        else:
            return "개선 필요"
    
    def _classify_drawdown(self, max_drawdown: float) -> str:
        """최대 낙폭 분류"""
        if max_drawdown <= 0.05:
            return "안전"
        elif max_drawdown <= 0.10:
            return "양호"
        elif max_drawdown <= 0.15:
            return "주의"
        else:
            return "위험"
    
    def _classify_loss_streak(self, consecutive_losses: int) -> str:
        """연속 손실 분류"""
        if consecutive_losses == 0:
            return "안정"
        elif consecutive_losses == 1:
            return "정상"
        elif consecutive_losses == 2:
            return "주의"
        else:
            return "위험"
    
    def _get_overall_assessment(self, analysis: Dict[str, str]) -> str:
        """종합 평가"""
        risk_items = sum(1 for v in analysis.values() 
                        if v in ["위험", "재검토 필요", "개선 필요"])
        
        if risk_items >= 3:
            return "전략 재검토 필요"
        elif risk_items >= 2:
            return "개선 필요"  
        elif risk_items >= 1:
            return "주의 필요"
        else:
            return "양호"
    
    def _log_adjustment(self, adjustment: AdjustmentParameters):
        """조정 로깅"""
        self.adjustment_history.append({
            "timestamp": datetime.now(),
            "action": adjustment.action.value,
            "reason": adjustment.reason,
            "position_multiplier": adjustment.position_size_multiplier,
            "threshold_adjustment": adjustment.entry_threshold_adjustment,
            "duration": adjustment.duration_days
        })
        
        logger.info(f"Strategy adjustment applied: {adjustment.action.value} - {adjustment.reason}")