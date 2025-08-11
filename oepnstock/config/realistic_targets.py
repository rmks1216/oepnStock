"""
현실적 수익률 목표 및 리스크 관리 설정
소액 투자자를 위한 맞춤형 파라미터
"""
from dataclasses import dataclass
from typing import Dict, Any
import math


@dataclass
class RealisticTargets:
    """현실적 수익률 목표 클래스"""
    
    # 일일 수익률 목표 (보수적)
    daily_target_min: float = 0.0005  # 0.05%
    daily_target_max: float = 0.001   # 0.1%
    daily_target_optimal: float = 0.0008  # 0.08%
    
    # 월간/연간 목표 (복리 계산)
    monthly_target: float = 0.025     # 2.5%
    annual_target: float = 0.20       # 20%
    
    # 리스크 한도
    daily_max_loss: float = -0.02     # -2%
    monthly_max_drawdown: float = -0.05  # -5%
    consecutive_loss_limit: int = 3   # 연속 손실 3회
    
    # 안정성 지표 목표
    target_sharpe_ratio: float = 1.0  # 샤프 비율 1.0 이상
    target_win_rate: float = 0.55     # 승률 55% 이상
    max_drawdown: float = 0.10        # 최대 낙폭 10% 이하
    
    def calculate_compound_return(self, daily_rate: float, days: int = 250) -> float:
        """복리 수익률 계산"""
        return (1 + daily_rate) ** days - 1
    
    def get_realistic_annual_range(self) -> tuple:
        """현실적 연간 수익률 범위"""
        min_annual = self.calculate_compound_return(self.daily_target_min)
        max_annual = self.calculate_compound_return(self.daily_target_max)
        optimal_annual = self.calculate_compound_return(self.daily_target_optimal)
        
        return (min_annual, optimal_annual, max_annual)


@dataclass
class CapitalBasedStrategy:
    """투자금액별 맞춤 전략"""
    
    capital: int
    max_positions: int
    position_ratio: float
    daily_target: float
    risk_level: str
    max_daily_trades: int
    preferred_stocks: list
    sector_limit: float = 0.5
    
    def get_position_size(self) -> int:
        """포지션 크기 계산"""
        return int(self.capital * self.position_ratio)
    
    def get_daily_target_amount(self) -> int:
        """일일 목표 수익 금액"""
        return int(self.capital * self.daily_target)


class InvestmentProfiles:
    """투자금액별 프로파일 관리"""
    
    def __init__(self):
        self.profiles = {
            1_000_000: CapitalBasedStrategy(
                capital=1_000_000,
                max_positions=2,
                position_ratio=0.5,
                daily_target=0.0005,  # 0.05%
                risk_level="초보수적",
                max_daily_trades=2,
                preferred_stocks=["대형주", "우량주"],
                sector_limit=1.0  # 제한 없음 (종목수 적음)
            ),
            
            3_000_000: CapitalBasedStrategy(
                capital=3_000_000,
                max_positions=3,
                position_ratio=0.33,
                daily_target=0.0006,  # 0.06%
                risk_level="보수적",
                max_daily_trades=3,
                preferred_stocks=["대형주", "우량주"],
                sector_limit=0.6
            ),
            
            5_000_000: CapitalBasedStrategy(
                capital=5_000_000,
                max_positions=3,
                position_ratio=0.33,
                daily_target=0.0007,  # 0.07%
                risk_level="보수적",
                max_daily_trades=4,
                preferred_stocks=["대형주", "중형주"],
                sector_limit=0.5
            ),
            
            10_000_000: CapitalBasedStrategy(
                capital=10_000_000,
                max_positions=4,
                position_ratio=0.25,
                daily_target=0.0008,  # 0.08%
                risk_level="균형적",
                max_daily_trades=6,
                preferred_stocks=["모든 유형"],
                sector_limit=0.4
            )
        }
    
    def get_profile(self, capital: int) -> CapitalBasedStrategy:
        """투자금액에 맞는 프로파일 반환"""
        if capital <= 1_000_000:
            return self.profiles[1_000_000]
        elif capital <= 3_000_000:
            return self.profiles[3_000_000]
        elif capital <= 5_000_000:
            return self.profiles[5_000_000]
        else:
            return self.profiles[10_000_000]
    
    def get_all_profiles(self) -> Dict[int, CapitalBasedStrategy]:
        """모든 프로파일 반환"""
        return self.profiles


@dataclass
class RiskLimits:
    """강화된 리스크 관리 한도"""
    
    # 손실 제한
    daily_max_loss: float = -0.02        # 일일 최대 손실 -2%
    position_stop_loss: float = -0.015   # 개별 포지션 -1.5% 손절
    monthly_drawdown: float = -0.05      # 월간 -5% 도달 시 휴식
    consecutive_loss_limit: int = 3      # 연속 손실 3회 시 중단
    
    # 거래량 제한
    max_volume_ratio: float = 3.0        # 평균 거래량 대비 300% 초과 회피
    
    # 시장 상황별 제한
    vix_threshold: int = 30              # VIX 30 이상 시 신규 진입 중단
    market_score_minimum: int = 75       # Market Score 75점 이상에서만 거래
    
    # 기술지표 동의율
    signal_consensus_threshold: float = 0.7  # 11개 지표 중 70% 이상 동의
    
    def is_safe_to_trade(self, current_loss: float, consecutive_losses: int, 
                        vix: float, market_score: int) -> bool:
        """거래 안전성 검사"""
        return (
            current_loss > self.daily_max_loss and
            consecutive_losses < self.consecutive_loss_limit and
            vix < self.vix_threshold and
            market_score >= self.market_score_minimum
        )


class PerformanceMonitor:
    """성과 모니터링 클래스"""
    
    def __init__(self):
        self.targets = RealisticTargets()
        self.risk_limits = RiskLimits()
    
    def evaluate_performance(self, daily_return: float, monthly_return: float,
                           sharpe_ratio: float, win_rate: float) -> Dict[str, str]:
        """성과 평가"""
        evaluation = {
            "daily_return": self._evaluate_daily_return(daily_return),
            "monthly_return": self._evaluate_monthly_return(monthly_return),
            "sharpe_ratio": self._evaluate_sharpe_ratio(sharpe_ratio),
            "win_rate": self._evaluate_win_rate(win_rate)
        }
        return evaluation
    
    def _evaluate_daily_return(self, return_rate: float) -> str:
        """일일 수익률 평가"""
        if return_rate >= self.targets.daily_target_optimal:
            return "우수"
        elif return_rate >= self.targets.daily_target_min:
            return "양호"
        elif return_rate >= 0:
            return "보통"
        elif return_rate >= self.risk_limits.daily_max_loss:
            return "주의"
        else:
            return "위험"
    
    def _evaluate_monthly_return(self, return_rate: float) -> str:
        """월간 수익률 평가"""
        if return_rate >= self.targets.monthly_target:
            return "목표 달성"
        elif return_rate >= 0:
            return "플러스 수익"
        elif return_rate >= self.risk_limits.monthly_drawdown:
            return "손실 관용 범위"
        else:
            return "휴식 필요"
    
    def _evaluate_sharpe_ratio(self, sharpe: float) -> str:
        """샤프 비율 평가"""
        if sharpe >= self.targets.target_sharpe_ratio:
            return "A급 전략"
        elif sharpe >= 0.5:
            return "양호한 전략"
        elif sharpe >= 0:
            return "개선 필요"
        else:
            return "전략 재검토"
    
    def _evaluate_win_rate(self, win_rate: float) -> str:
        """승률 평가"""
        if win_rate >= self.targets.target_win_rate:
            return "목표 달성"
        elif win_rate >= 0.5:
            return "평균 이상"
        elif win_rate >= 0.45:
            return "개선 필요"
        else:
            return "전략 수정 필요"


# 전역 인스턴스
realistic_targets = RealisticTargets()
investment_profiles = InvestmentProfiles()
risk_limits = RiskLimits()
performance_monitor = PerformanceMonitor()