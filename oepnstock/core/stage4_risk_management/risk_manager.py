"""
Risk Management - Stage 4 of 4-Stage Trading Strategy
리스크 관리 계획 세우기: 손절, 목표가, 포지션 관리, 비용 반영
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ...config import config
from ...utils import get_logger

logger = get_logger(__name__)


@dataclass
class StopLoss:
    """Stop loss configuration"""
    price: float
    percentage: float
    type: str  # 'fixed', 'trailing', 'time_based'
    reason: str
    activation_time: Optional[datetime] = None


@dataclass
class TargetPrice:
    """Target price configuration"""
    price: float
    sell_ratio: float  # 매도할 비율 (0.0 ~ 1.0)
    type: str  # 'resistance', 'fibonacci', 'trailing'
    reason: str
    priority: int


@dataclass
class PositionSize:
    """Position size calculation result"""
    shares: int
    investment_amount: float
    position_ratio: float  # 전체 자본 대비 비율
    risk_amount: float     # 위험 금액
    method: str           # '2_percent_rule', 'kelly_formula'
    reasoning: List[str]


@dataclass
class TradingCosts:
    """Trading cost breakdown"""
    commission_buy: float
    commission_sell: float
    tax: float
    slippage: float
    total_cost: float
    net_breakeven_price: float


@dataclass
class RiskManagementPlan:
    """Complete risk management plan"""
    symbol: str
    entry_price: float
    position_size: PositionSize
    stop_loss: StopLoss
    target_prices: List[TargetPrice]
    trading_costs: TradingCosts
    max_holding_period: int  # days
    position_limits: Dict[str, Any]
    risk_reward_ratio: float
    expected_return: float
    max_drawdown_limit: float
    emergency_exit_conditions: List[str]
    monitoring_alerts: List[Dict[str, Any]]


class RiskManager:
    """
    4단계: 리스크 관리 계획 세우기 (안전장치)
    
    주요 기능:
    - 초기 리스크 관리 (2% 룰) / 켈리 공식 기반 자금 관리
    - 손절선 설정 (시장 상황별 차등 적용)
    - 단계별 목표가 설정
    - 거래 비용 반영 및 실제 순수익률 계산
    - 포지션 관리 및 집중도 제한
    """
    
    def __init__(self):
        self.config = config.trading
        self.costs = config.costs
        
        # Trading statistics for Kelly formula (초기값)
        self.trading_stats = {
            'trade_count': 0,
            'win_rate': 0.6,        # 초기 추정치
            'avg_win_percent': 0.08,  # 8% 평균 수익
            'avg_loss_percent': 0.03  # 3% 평균 손실
        }
        
        # Risk management parameters by market condition
        self.risk_params_by_market = {
            'strong_uptrend': {
                'base_stop_loss': 0.03,      # 3%
                'position_multiplier': 1.2,
                'max_holding_days': 5
            },
            'weak_uptrend': {
                'base_stop_loss': 0.025,     # 2.5%
                'position_multiplier': 1.0,
                'max_holding_days': 3
            },
            'sideways': {
                'base_stop_loss': 0.02,      # 2%
                'position_multiplier': 0.8,
                'max_holding_days': 3
            },
            'downtrend': {
                'base_stop_loss': 0.015,     # 1.5%
                'position_multiplier': 0.5,
                'max_holding_days': 2
            }
        }
    
    def create_risk_management_plan(self,
                                  symbol: str,
                                  entry_price: float,
                                  support_levels: List[float],
                                  resistance_levels: List[float],
                                  market_condition: str = 'sideways',
                                  total_capital: float = 10000000,  # 1천만원 기본값
                                  current_portfolio: Optional[Dict] = None) -> RiskManagementPlan:
        """
        종합적인 리스크 관리 계획 수립
        
        Args:
            symbol: 종목 코드
            entry_price: 진입가격
            support_levels: 지지선 가격들
            resistance_levels: 저항선 가격들  
            market_condition: 시장 상황
            total_capital: 총 자본
            current_portfolio: 현재 포트폴리오 정보
            
        Returns:
            RiskManagementPlan: 종합 리스크 관리 계획
        """
        logger.info(f"Creating risk management plan for {symbol} at {entry_price}")
        
        try:
            # 1. 손절선 설정
            stop_loss = self._calculate_stop_loss(
                entry_price, support_levels, market_condition
            )
            
            # 2. 포지션 크기 계산
            position_size = self._calculate_position_size(
                total_capital, entry_price, stop_loss.price, market_condition
            )
            
            # 3. 목표가 설정
            target_prices = self._calculate_target_prices(
                entry_price, resistance_levels, market_condition
            )
            
            # 4. 거래 비용 계산
            trading_costs = self._calculate_trading_costs(
                entry_price, position_size.investment_amount
            )
            
            # 5. 리스크 리워드 비율 계산
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                entry_price, stop_loss.price, target_prices
            )
            
            # 6. 기대 수익률 계산
            expected_return = self._calculate_expected_return(
                entry_price, target_prices, stop_loss, trading_costs
            )
            
            # 7. 포지션 제한 및 검증
            position_limits = self._validate_position_limits(
                position_size, current_portfolio, symbol
            )
            
            # 8. 비상 청산 조건
            emergency_conditions = self._define_emergency_exit_conditions(
                market_condition
            )
            
            # 9. 모니터링 알림 설정
            monitoring_alerts = self._setup_monitoring_alerts(
                symbol, entry_price, stop_loss, target_prices
            )
            
            # 10. 최대 보유 기간
            max_holding_period = self.risk_params_by_market[market_condition]['max_holding_days']
            
            plan = RiskManagementPlan(
                symbol=symbol,
                entry_price=entry_price,
                position_size=position_size,
                stop_loss=stop_loss,
                target_prices=target_prices,
                trading_costs=trading_costs,
                max_holding_period=max_holding_period,
                position_limits=position_limits,
                risk_reward_ratio=risk_reward_ratio,
                expected_return=expected_return,
                max_drawdown_limit=self.config.max_daily_loss_ratio,
                emergency_exit_conditions=emergency_conditions,
                monitoring_alerts=monitoring_alerts
            )
            
            logger.info(f"Risk management plan created - Position: {position_size.shares} shares, "
                       f"Risk/Reward: {risk_reward_ratio:.2f}, Expected Return: {expected_return:.2%}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating risk management plan for {symbol}: {e}")
            raise
    
    def _calculate_stop_loss(self, entry_price: float, support_levels: List[float],
                           market_condition: str) -> StopLoss:
        """손절선 계산"""
        risk_params = self.risk_params_by_market[market_condition]
        base_stop_percentage = risk_params['base_stop_loss']
        
        # 기본 손절선 (진입가 대비 %)
        basic_stop_price = entry_price * (1 - base_stop_percentage)
        
        # 지지선 기반 손절선
        support_based_stop = None
        if support_levels:
            nearest_support = max([s for s in support_levels if s < entry_price], default=0)
            if nearest_support > 0:
                # 지지선 하단 -2% 
                support_based_stop = nearest_support * 0.98
        
        # 두 방식 중 더 보수적인(높은) 가격 선택
        stop_candidates = [basic_stop_price]
        if support_based_stop:
            stop_candidates.append(support_based_stop)
        
        final_stop_price = max(stop_candidates)
        stop_percentage = (entry_price - final_stop_price) / entry_price
        
        # 손절 타입 결정
        stop_type = 'fixed'
        reason = f"기본 손절 {base_stop_percentage:.1%}"
        
        if support_based_stop and final_stop_price == support_based_stop:
            reason = f"지지선 기반 손절 (지지선: {nearest_support:,.0f})"
        
        # 시장 상황별 조정
        if market_condition == 'strong_uptrend':
            stop_type = 'trailing'
            reason += " + 트레일링 스탑"
        
        return StopLoss(
            price=final_stop_price,
            percentage=stop_percentage,
            type=stop_type,
            reason=reason
        )
    
    def _calculate_position_size(self, total_capital: float, entry_price: float,
                               stop_loss_price: float, market_condition: str) -> PositionSize:
        """포지션 크기 계산"""
        risk_params = self.risk_params_by_market[market_condition]
        position_multiplier = risk_params['position_multiplier']
        
        # 거래 횟수에 따른 방법 선택
        if self.trading_stats['trade_count'] < self.config.kelly_threshold_trades:
            # 2% 룰 적용
            size_info = self._calculate_initial_position_size(
                total_capital, entry_price, stop_loss_price
            )
            method = '2_percent_rule'
        else:
            # 켈리 공식 적용
            size_info = self._calculate_kelly_position_size(
                total_capital, entry_price, stop_loss_price
            )
            method = 'kelly_formula'
        
        # 시장 상황별 조정
        adjusted_amount = size_info['investment_amount'] * position_multiplier
        adjusted_shares = int(adjusted_amount / entry_price)
        actual_amount = adjusted_shares * entry_price
        
        # 포지션 제한 적용
        max_position_amount = total_capital * self.config.max_single_position_ratio
        if actual_amount > max_position_amount:
            adjusted_shares = int(max_position_amount / entry_price)
            actual_amount = adjusted_shares * entry_price
        
        position_ratio = actual_amount / total_capital
        risk_amount = adjusted_shares * (entry_price - stop_loss_price)
        
        reasoning = [
            f"방법: {method}",
            f"시장상황 조정: {position_multiplier}x",
            f"위험금액: {risk_amount:,.0f}원 ({risk_amount/total_capital:.2%})"
        ]
        
        if actual_amount < size_info['investment_amount'] * position_multiplier:
            reasoning.append("단일 포지션 한도로 인한 축소")
        
        return PositionSize(
            shares=adjusted_shares,
            investment_amount=actual_amount,
            position_ratio=position_ratio,
            risk_amount=risk_amount,
            method=method,
            reasoning=reasoning
        )
    
    def _calculate_initial_position_size(self, capital: float, entry_price: float,
                                       stop_loss_price: float) -> Dict[str, float]:
        """초기 2% 룰 기반 포지션 크기"""
        max_loss_amount = capital * self.config.initial_risk_per_trade  # 2%
        loss_per_share = entry_price - stop_loss_price
        
        if loss_per_share <= 0:
            raise ValueError("Stop loss price must be below entry price")
        
        shares = max_loss_amount / loss_per_share
        investment_amount = shares * entry_price
        
        # 최대 포지션 제한
        max_position = capital * self.config.max_single_position_ratio
        if investment_amount > max_position:
            shares = max_position / entry_price
            investment_amount = shares * entry_price
        
        return {
            'shares': int(shares),
            'investment_amount': investment_amount
        }
    
    def _calculate_kelly_position_size(self, capital: float, entry_price: float,
                                     stop_loss_price: float) -> Dict[str, float]:
        """켈리 공식 기반 포지션 크기"""
        stats = self.trading_stats
        
        if stats['trade_count'] < self.config.kelly_threshold_trades:
            # 데이터 부족 시 2% 룰로 폴백
            return self._calculate_initial_position_size(capital, entry_price, stop_loss_price)
        
        win_rate = stats['win_rate']
        avg_win = stats['avg_win_percent']
        avg_loss = stats['avg_loss_percent']
        
        # 켈리 비율 계산
        kelly_ratio = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # 안전 계수 적용 (켈리의 25%만 사용)
        safe_kelly = kelly_ratio * 0.25
        
        # 제한 적용
        position_ratio = max(0.01, min(safe_kelly, self.config.max_single_position_ratio))
        investment_amount = capital * position_ratio
        
        shares = investment_amount / entry_price
        
        return {
            'shares': int(shares),
            'investment_amount': investment_amount
        }
    
    def _calculate_target_prices(self, entry_price: float, resistance_levels: List[float],
                               market_condition: str) -> List[TargetPrice]:
        """단계별 목표가 설정"""
        targets = []
        
        # 1차 목표: 가장 가까운 저항선
        if resistance_levels:
            nearest_resistance = min([r for r in resistance_levels if r > entry_price], 
                                   default=entry_price * 1.05)
            targets.append(TargetPrice(
                price=nearest_resistance,
                sell_ratio=0.5,  # 50% 물량
                type='resistance',
                reason='가장 가까운 저항선',
                priority=1
            ))
        
        # 2차 목표: 피보나치 또는 직전 고점
        fib_382_price = entry_price * 1.08  # 8% 상승 (간단한 구현)
        previous_high = entry_price * 1.12   # 12% 상승 (간단한 구현)
        
        second_target_price = min(fib_382_price, previous_high)
        targets.append(TargetPrice(
            price=second_target_price,
            sell_ratio=0.3,  # 30% 물량
            type='fibonacci',
            reason='피보나치 0.382 또는 직전 고점',
            priority=2
        ))
        
        # 3차 목표: 트레일링 스탑 (나머지 20%)
        trailing_trigger_price = entry_price * 1.15  # 15% 상승 시점
        targets.append(TargetPrice(
            price=trailing_trigger_price,
            sell_ratio=0.2,  # 나머지 20%
            type='trailing',
            reason='트레일링 스탑 (고점 대비 5% 하락시)',
            priority=3
        ))
        
        # 시장 상황별 목표가 조정
        if market_condition == 'strong_uptrend':
            # 강세장에서는 목표가를 더 높게
            for target in targets:
                target.price *= 1.1
        elif market_condition == 'downtrend':
            # 약세장에서는 목표가를 더 낮게
            for target in targets:
                target.price *= 0.9
        
        return sorted(targets, key=lambda x: x.priority)
    
    def _calculate_trading_costs(self, entry_price: float, investment_amount: float) -> TradingCosts:
        """거래 비용 계산"""
        commission_buy = investment_amount * self.costs.commission_buy
        commission_sell = investment_amount * self.costs.commission_sell
        tax = investment_amount * self.costs.tax  # 매도시만 적용
        slippage = investment_amount * self.costs.slippage_limit  # 지정가 기준
        
        total_cost = commission_buy + commission_sell + tax + slippage
        
        # 손익분기점 (매수가 + 모든 비용)
        shares = investment_amount / entry_price
        cost_per_share = total_cost / shares if shares > 0 else 0
        breakeven_price = entry_price + cost_per_share
        
        return TradingCosts(
            commission_buy=commission_buy,
            commission_sell=commission_sell,
            tax=tax,
            slippage=slippage,
            total_cost=total_cost,
            net_breakeven_price=breakeven_price
        )
    
    def _calculate_risk_reward_ratio(self, entry_price: float, stop_loss_price: float,
                                   target_prices: List[TargetPrice]) -> float:
        """리스크 리워드 비율 계산"""
        max_loss = entry_price - stop_loss_price
        
        # 가중 평균 수익 계산
        total_expected_gain = 0
        for target in target_prices:
            gain_per_share = target.price - entry_price
            weighted_gain = gain_per_share * target.sell_ratio
            total_expected_gain += weighted_gain
        
        if max_loss <= 0:
            return 0.0
        
        risk_reward_ratio = total_expected_gain / max_loss
        return risk_reward_ratio
    
    def _calculate_expected_return(self, entry_price: float, target_prices: List[TargetPrice],
                                 stop_loss: StopLoss, trading_costs: TradingCosts) -> float:
        """기대 수익률 계산 (비용 포함)"""
        # 승률 기반 기대값 계산
        win_rate = self.trading_stats['win_rate']
        
        # 수익 시나리오 (가중 평균)
        win_return = 0
        for target in target_prices:
            net_gain = target.price - entry_price - (trading_costs.total_cost / target.sell_ratio)
            win_return += (net_gain / entry_price) * target.sell_ratio
        
        # 손실 시나리오
        net_loss = entry_price - stop_loss.price + trading_costs.total_cost
        loss_return = -net_loss / entry_price
        
        # 기대 수익률
        expected_return = win_rate * win_return + (1 - win_rate) * loss_return
        
        return expected_return
    
    def _validate_position_limits(self, position_size: PositionSize, 
                                current_portfolio: Optional[Dict],
                                symbol: str) -> Dict[str, Any]:
        """포지션 제한 검증"""
        limits = {
            'single_position_ok': position_size.position_ratio <= self.config.max_single_position_ratio,
            'total_positions_ok': True,  # 포트폴리오 데이터가 있을 때 계산
            'sector_exposure_ok': True,   # 섹터별 노출 한도
            'correlation_ok': True        # 상관관계 리스크
        }
        
        warnings = []
        
        if not limits['single_position_ok']:
            warnings.append(f"단일 포지션 한도 {self.config.max_single_position_ratio:.0%} 초과")
        
        if current_portfolio:
            # 현재 포지션 수 확인
            current_positions = len(current_portfolio.get('positions', []))
            limits['total_positions_ok'] = current_positions < self.config.max_positions
            
            if not limits['total_positions_ok']:
                warnings.append(f"최대 포지션 수 {self.config.max_positions}개 초과")
        
        limits['warnings'] = warnings
        
        return limits
    
    def _define_emergency_exit_conditions(self, market_condition: str) -> List[str]:
        """비상 청산 조건 정의"""
        conditions = [
            "일일 손실 한도 도달",
            "시스템 장애 발생",
            "API 연결 장애",
            "중대 뉴스/공시 발생"
        ]
        
        if market_condition == 'downtrend':
            conditions.extend([
                "시장 지수 -5% 이상 하락",
                "섹터 지수 -8% 이상 하락"
            ])
        
        return conditions
    
    def _setup_monitoring_alerts(self, symbol: str, entry_price: float,
                               stop_loss: StopLoss, target_prices: List[TargetPrice]) -> List[Dict]:
        """모니터링 알림 설정"""
        alerts = []
        
        # 손절선 근접 알림 (95% 지점)
        alerts.append({
            'type': 'stop_loss_warning',
            'trigger_price': stop_loss.price * 1.05,  # 손절선 +5%
            'message': f'{symbol} 손절선 근접 경고'
        })
        
        # 목표가 도달 알림
        for i, target in enumerate(target_prices, 1):
            alerts.append({
                'type': f'target_{i}_reached',
                'trigger_price': target.price,
                'message': f'{symbol} {i}차 목표가 도달',
                'action': f'{target.sell_ratio:.0%} 매도 검토'
            })
        
        # 시간 기반 알림 (보유기간 경고)
        alerts.append({
            'type': 'time_limit_warning',
            'trigger_time': datetime.now() + timedelta(days=self.config.max_holding_days - 1),
            'message': f'{symbol} 최대 보유기간 임박'
        })
        
        return alerts
    
    def update_trading_statistics(self, trade_result: Dict[str, Any]) -> None:
        """거래 통계 업데이트 (켈리 공식용)"""
        current_stats = self.trading_stats
        
        # 새로운 거래 결과 반영
        is_win = trade_result['return'] > 0
        return_pct = abs(trade_result['return'])
        
        # 이동 평균 방식으로 통계 업데이트
        alpha = 0.1  # 학습률
        
        if current_stats['trade_count'] == 0:
            # 첫 번째 거래
            current_stats['win_rate'] = 1.0 if is_win else 0.0
            current_stats['avg_win_percent'] = return_pct if is_win else 0.08
            current_stats['avg_loss_percent'] = return_pct if not is_win else 0.03
        else:
            # 승률 업데이트
            current_win_rate = current_stats['win_rate']
            new_win_rate = (current_win_rate * (1 - alpha) + 
                           (1.0 if is_win else 0.0) * alpha)
            current_stats['win_rate'] = new_win_rate
            
            # 평균 수익/손실 업데이트
            if is_win:
                current_stats['avg_win_percent'] = (
                    current_stats['avg_win_percent'] * (1 - alpha) + 
                    return_pct * alpha
                )
            else:
                current_stats['avg_loss_percent'] = (
                    current_stats['avg_loss_percent'] * (1 - alpha) + 
                    return_pct * alpha
                )
        
        current_stats['trade_count'] += 1
        
        logger.info(f"Trading statistics updated - Trades: {current_stats['trade_count']}, "
                   f"Win Rate: {current_stats['win_rate']:.2%}, "
                   f"Avg Win: {current_stats['avg_win_percent']:.2%}, "
                   f"Avg Loss: {current_stats['avg_loss_percent']:.2%}")
    
    def calculate_net_return(self, entry_price: float, exit_price: float,
                           shares: int, order_type: str = 'limit') -> Dict[str, float]:
        """실제 순수익률 계산 (모든 비용 차감)"""
        gross_return = (exit_price - entry_price) / entry_price
        investment_amount = shares * entry_price
        
        # 비용 계산
        commission_buy = investment_amount * self.costs.commission_buy
        commission_sell = investment_amount * self.costs.commission_sell
        tax = investment_amount * self.costs.tax if exit_price > entry_price else 0  # 수익시만
        slippage_cost = investment_amount * getattr(self.costs, f'slippage_{order_type}', self.costs.slippage_limit)
        
        total_costs = commission_buy + commission_sell + tax + slippage_cost
        
        # 순수익 계산
        gross_pnl = shares * (exit_price - entry_price)
        net_pnl = gross_pnl - total_costs
        net_return = net_pnl / investment_amount
        
        return {
            'gross_return': gross_return,
            'net_return': net_return,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'total_costs': total_costs,
            'cost_impact': (gross_return - net_return)
        }