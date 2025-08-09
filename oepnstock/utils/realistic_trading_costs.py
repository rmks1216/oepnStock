"""
Realistic Trading Costs Model - 현실적인 거래비용 계산
FEEDBACK.md의 지적사항을 반영하여 보수적이고 현실적인 비용을 적용합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """주문 유형"""
    MARKET_BUY = "market_buy"      # 시장가 매수
    MARKET_SELL = "market_sell"    # 시장가 매도
    LIMIT_BUY = "limit_buy"        # 지정가 매수
    LIMIT_SELL = "limit_sell"      # 지정가 매도


class LiquidityTier(Enum):
    """유동성 등급"""
    TIER1 = "tier1"  # 대형주 (시총 1조 이상)
    TIER2 = "tier2"  # 중형주 (시총 1000억~1조)
    TIER3 = "tier3"  # 중소형주 (시총 500억~1000억)
    TIER4 = "tier4"  # 소형주 (시총 500억 미만)


@dataclass
class TradingCostComponents:
    """거래비용 구성요소"""
    commission_buy: float      # 매수 수수료
    commission_sell: float     # 매도 수수료
    transaction_tax: float     # 거래세 (매도시)
    slippage: float           # 슬리피지
    bid_ask_spread: float     # 호가 스프레드
    market_impact: float      # 시장 충격비용
    total_round_trip: float   # 총 왕복 비용
    
    # 세부 정보
    order_amount: float       # 주문금액
    liquidity_tier: LiquidityTier  # 유동성 등급
    volatility_adjustment: float   # 변동성 조정


class RealisticTradingCosts:
    """
    현실적인 거래비용 계산기
    
    한국 주식시장 기준:
    - 매수 수수료: 증권사별 차등 (0.015% ~ 0.05%)
    - 매도 수수료: 증권사별 차등 + 거래세 0.23%
    - 슬리피지: 유동성/변동성/주문규모 dependent
    - 호가 스프레드: Tier별 차등
    - 시장 충격비용: 대량 거래시
    """
    
    def __init__(self):
        # 기본 수수료율 (보수적 설정)
        self.base_rates = {
            'commission_buy_rate': 0.0003,    # 0.03% (높은 수수료 가정)
            'commission_sell_rate': 0.0003,   # 0.03%
            'transaction_tax_rate': 0.0023,   # 0.23% (거래세)
            'min_commission': 1000,           # 최소 수수료 1,000원
        }
        
        # 유동성 등급별 비용 (더 보수적으로 설정)
        self.liquidity_costs = {
            LiquidityTier.TIER1: {
                'base_slippage': 0.0005,      # 0.05%
                'bid_ask_spread': 0.0003,     # 0.03%
                'volatility_multiplier': 1.0
            },
            LiquidityTier.TIER2: {
                'base_slippage': 0.001,       # 0.1%
                'bid_ask_spread': 0.0005,     # 0.05%
                'volatility_multiplier': 1.2
            },
            LiquidityTier.TIER3: {
                'base_slippage': 0.002,       # 0.2%
                'bid_ask_spread': 0.001,      # 0.1%
                'volatility_multiplier': 1.5
            },
            LiquidityTier.TIER4: {
                'base_slippage': 0.004,       # 0.4%
                'bid_ask_spread': 0.002,      # 0.2%
                'volatility_multiplier': 2.0
            }
        }
        
        # 시장 충격비용 기준 (주문금액 대비)
        self.market_impact_thresholds = {
            'small_order': 10000000,      # 1천만원 이하
            'medium_order': 100000000,    # 1억원 이하
            'large_order': 500000000,     # 5억원 이하
        }
        
        # 변동성별 추가 비용
        self.volatility_adjustments = {
            'low': 1.0,      # ATR < 2%
            'medium': 1.3,   # ATR 2-4%
            'high': 1.8,     # ATR 4-6%
            'extreme': 2.5   # ATR > 6%
        }
    
    def classify_liquidity_tier(self, market_cap: float) -> LiquidityTier:
        """
        시가총액 기준 유동성 등급 분류
        
        Args:
            market_cap: 시가총액 (억원)
            
        Returns:
            LiquidityTier
        """
        if market_cap >= 10000:  # 1조원 이상
            return LiquidityTier.TIER1
        elif market_cap >= 1000:  # 1000억원 이상
            return LiquidityTier.TIER2
        elif market_cap >= 500:   # 500억원 이상
            return LiquidityTier.TIER3
        else:
            return LiquidityTier.TIER4
    
    def calculate_volatility_adjustment(self, atr_ratio: float) -> Tuple[float, str]:
        """
        변동성 기준 비용 조정
        
        Args:
            atr_ratio: ATR / 가격 비율
            
        Returns:
            (조정 배수, 분류)
        """
        if atr_ratio < 0.02:
            return self.volatility_adjustments['low'], 'low'
        elif atr_ratio < 0.04:
            return self.volatility_adjustments['medium'], 'medium'
        elif atr_ratio < 0.06:
            return self.volatility_adjustments['high'], 'high'
        else:
            return self.volatility_adjustments['extreme'], 'extreme'
    
    def calculate_commission(self, 
                           order_amount: float, 
                           is_sell: bool = False) -> float:
        """
        수수료 계산
        
        Args:
            order_amount: 주문금액
            is_sell: 매도 주문 여부
            
        Returns:
            수수료 금액
        """
        if is_sell:
            rate = self.base_rates['commission_sell_rate']
        else:
            rate = self.base_rates['commission_buy_rate']
        
        commission = order_amount * rate
        return max(commission, self.base_rates['min_commission'])
    
    def calculate_slippage(self, 
                          order_amount: float,
                          liquidity_tier: LiquidityTier,
                          volatility_adj: float,
                          order_type: OrderType) -> float:
        """
        슬리피지 계산
        
        Args:
            order_amount: 주문금액
            liquidity_tier: 유동성 등급
            volatility_adj: 변동성 조정배수
            order_type: 주문 유형
            
        Returns:
            슬리피지 비용 (주문금액 대비 비율)
        """
        # 기본 슬리피지
        base_slippage = self.liquidity_costs[liquidity_tier]['base_slippage']
        
        # 시장가 주문시 추가 비용
        if order_type in [OrderType.MARKET_BUY, OrderType.MARKET_SELL]:
            market_penalty = 1.5  # 50% 추가
        else:
            market_penalty = 1.0  # 지정가는 기본
        
        # 주문 규모별 조정
        if order_amount > self.market_impact_thresholds['large_order']:
            size_multiplier = 2.0
        elif order_amount > self.market_impact_thresholds['medium_order']:
            size_multiplier = 1.5
        elif order_amount > self.market_impact_thresholds['small_order']:
            size_multiplier = 1.2
        else:
            size_multiplier = 1.0
        
        # 최종 슬리피지
        total_slippage = (
            base_slippage * 
            volatility_adj * 
            market_penalty * 
            size_multiplier
        )
        
        return min(total_slippage, 0.02)  # 최대 2% 제한
    
    def calculate_market_impact(self, 
                              order_amount: float,
                              daily_volume: float,
                              liquidity_tier: LiquidityTier) -> float:
        """
        시장 충격비용 계산
        
        Args:
            order_amount: 주문금액
            daily_volume: 일평균 거래대금
            liquidity_tier: 유동성 등급
            
        Returns:
            시장 충격비용 (비율)
        """
        if daily_volume <= 0:
            return 0.005  # 기본값 0.5%
        
        # 주문비중 (일 거래량 대비)
        order_ratio = order_amount / daily_volume
        
        # 기본 충격비용 (유동성 등급별)
        base_impact = {
            LiquidityTier.TIER1: 0.0001,
            LiquidityTier.TIER2: 0.0002,
            LiquidityTier.TIER3: 0.0005,
            LiquidityTier.TIER4: 0.001
        }[liquidity_tier]
        
        # 비선형 증가 (제곱근 함수)
        if order_ratio < 0.01:  # 1% 미만
            impact_multiplier = 1.0
        elif order_ratio < 0.05:  # 5% 미만
            impact_multiplier = 2.0 + (order_ratio - 0.01) * 50
        else:  # 5% 이상
            impact_multiplier = 4.0 + (order_ratio - 0.05) * 100
        
        market_impact = base_impact * impact_multiplier
        return min(market_impact, 0.01)  # 최대 1% 제한
    
    def calculate_total_costs(self,
                            order_amount: float,
                            market_cap: float,
                            atr_ratio: float,
                            daily_volume: float,
                            order_type: OrderType = OrderType.MARKET_BUY) -> TradingCostComponents:
        """
        종합 거래비용 계산
        
        Args:
            order_amount: 주문금액
            market_cap: 시가총액 (억원)
            atr_ratio: ATR/가격 비율
            daily_volume: 일평균 거래대금
            order_type: 주문 유형
            
        Returns:
            TradingCostComponents
        """
        # 기본 분류
        liquidity_tier = self.classify_liquidity_tier(market_cap)
        volatility_adj, vol_category = self.calculate_volatility_adjustment(atr_ratio)
        
        # 각 비용 구성요소 계산
        commission_buy = self.calculate_commission(order_amount, is_sell=False)
        commission_sell = self.calculate_commission(order_amount, is_sell=True)
        transaction_tax = order_amount * self.base_rates['transaction_tax_rate']
        
        # 슬리피지 (매수/매도 각각)
        slippage_rate = self.calculate_slippage(
            order_amount, liquidity_tier, volatility_adj, order_type
        )
        slippage_cost = order_amount * slippage_rate
        
        # 호가 스프레드
        spread_rate = self.liquidity_costs[liquidity_tier]['bid_ask_spread']
        bid_ask_cost = order_amount * spread_rate * volatility_adj
        
        # 시장 충격비용
        impact_rate = self.calculate_market_impact(
            order_amount, daily_volume, liquidity_tier
        )
        market_impact_cost = order_amount * impact_rate
        
        # 총 왕복 비용 (매수 + 매도)
        total_round_trip = (
            commission_buy +      # 매수 수수료
            commission_sell +     # 매도 수수료
            transaction_tax +     # 거래세
            slippage_cost * 2 +   # 슬리피지 (매수+매도)
            bid_ask_cost +        # 호가 스프레드
            market_impact_cost    # 시장 충격
        )
        
        # 로깅
        logger.info(f"거래비용 계산 완료:")
        logger.info(f"  주문금액: {order_amount:,.0f}원")
        logger.info(f"  유동성등급: {liquidity_tier.value}")
        logger.info(f"  변동성: {vol_category} (x{volatility_adj:.1f})")
        logger.info(f"  총 비용: {total_round_trip:,.0f}원 ({total_round_trip/order_amount*100:.3f}%)")
        
        return TradingCostComponents(
            commission_buy=commission_buy,
            commission_sell=commission_sell,
            transaction_tax=transaction_tax,
            slippage=slippage_cost,
            bid_ask_spread=bid_ask_cost,
            market_impact=market_impact_cost,
            total_round_trip=total_round_trip,
            order_amount=order_amount,
            liquidity_tier=liquidity_tier,
            volatility_adjustment=volatility_adj
        )
    
    def get_cost_breakdown_by_strategy(self, strategy_type: str) -> Dict[str, float]:
        """
        전략별 예상 거래비용 (연간 기준)
        
        Args:
            strategy_type: 전략 유형 (default, aggressive, conservative, scalping, swing)
            
        Returns:
            예상 연간 비용 딕셔너리
        """
        # 전략별 거래 특성 (연간 기준)
        strategy_profiles = {
            'default': {
                'trades_per_year': 73,        # 5일 리밸런싱 = 연 73회
                'avg_order_size': 2000000,    # 200만원
                'liquidity_preference': 'tier2'
            },
            'aggressive': {
                'trades_per_year': 122,       # 3일 리밸런싱 = 연 122회
                'avg_order_size': 1400000,    # 140만원 (7개 포지션)
                'liquidity_preference': 'tier2'
            },
            'conservative': {
                'trades_per_year': 52,        # 7일 리밸런싱 = 연 52회
                'avg_order_size': 3300000,    # 330만원 (3개 포지션)
                'liquidity_preference': 'tier1'
            },
            'scalping': {
                'trades_per_year': 365,       # 매일 리밸런싱 = 연 365회
                'avg_order_size': 1000000,    # 100만원 (10개 포지션)
                'liquidity_preference': 'tier3'  # 더 다양한 종목
            },
            'swing': {
                'trades_per_year': 36,        # 10일 리밸런싱 = 연 36회
                'avg_order_size': 3300000,    # 330만원 (3개 포지션)
                'liquidity_preference': 'tier1'
            }
        }
        
        if strategy_type not in strategy_profiles:
            logger.warning(f"Unknown strategy type: {strategy_type}")
            return {}
        
        profile = strategy_profiles[strategy_type]
        
        # 대표적인 거래 조건으로 비용 계산
        market_cap_by_tier = {
            'tier1': 15000,  # 1.5조원
            'tier2': 3000,   # 3000억원
            'tier3': 700,    # 700억원
        }
        
        market_cap = market_cap_by_tier[profile['liquidity_preference']]
        atr_ratio = 0.025    # 보통 변동성 2.5%
        daily_volume = profile['avg_order_size'] * 100  # 충분한 유동성
        
        # 단위 거래 비용 계산
        costs = self.calculate_total_costs(
            order_amount=profile['avg_order_size'],
            market_cap=market_cap,
            atr_ratio=atr_ratio,
            daily_volume=daily_volume
        )
        
        # 연간 총 비용
        annual_total_cost = costs.total_round_trip * profile['trades_per_year']
        annual_cost_ratio = annual_total_cost / (10000000)  # 1천만원 기준
        
        return {
            'trades_per_year': profile['trades_per_year'],
            'cost_per_trade': costs.total_round_trip,
            'annual_total_cost': annual_total_cost,
            'annual_cost_ratio': annual_cost_ratio,
            'cost_breakdown': {
                'commission': (costs.commission_buy + costs.commission_sell) * profile['trades_per_year'],
                'tax': costs.transaction_tax * profile['trades_per_year'],
                'slippage': costs.slippage * 2 * profile['trades_per_year'],
                'spread': costs.bid_ask_spread * profile['trades_per_year'],
                'market_impact': costs.market_impact * profile['trades_per_year']
            }
        }


def main():
    """테스트용 메인 함수"""
    cost_calculator = RealisticTradingCosts()
    
    print("=== 거래비용 계산 예제 ===")
    
    # 다양한 시나리오 테스트
    scenarios = [
        {
            'name': '대형주 소액 거래',
            'order_amount': 1000000,   # 100만원
            'market_cap': 20000,       # 2조원
            'atr_ratio': 0.02,         # 2%
            'daily_volume': 10000000000  # 100억원
        },
        {
            'name': '중소형주 중간 거래',
            'order_amount': 5000000,   # 500만원
            'market_cap': 800,         # 800억원
            'atr_ratio': 0.04,         # 4%
            'daily_volume': 1000000000   # 10억원
        },
        {
            'name': '소형주 고변동성',
            'order_amount': 2000000,   # 200만원
            'market_cap': 300,         # 300억원
            'atr_ratio': 0.07,         # 7%
            'daily_volume': 500000000    # 5억원
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        costs = cost_calculator.calculate_total_costs(
            order_amount=scenario['order_amount'],
            market_cap=scenario['market_cap'],
            atr_ratio=scenario['atr_ratio'],
            daily_volume=scenario['daily_volume']
        )
        
        print(f"주문금액: {costs.order_amount:,.0f}원")
        print(f"총 비용: {costs.total_round_trip:,.0f}원 ({costs.total_round_trip/costs.order_amount*100:.3f}%)")
        print(f"  - 매수수수료: {costs.commission_buy:,.0f}원")
        print(f"  - 매도수수료: {costs.commission_sell:,.0f}원")
        print(f"  - 거래세: {costs.transaction_tax:,.0f}원")
        print(f"  - 슬리피지: {costs.slippage:,.0f}원")
        print(f"  - 호가스프레드: {costs.bid_ask_spread:,.0f}원")
        print(f"  - 시장충격: {costs.market_impact:,.0f}원")
    
    print("\n=== 전략별 연간 거래비용 ===")
    strategies = ['default', 'aggressive', 'conservative', 'scalping', 'swing']
    
    for strategy in strategies:
        cost_breakdown = cost_calculator.get_cost_breakdown_by_strategy(strategy)
        print(f"\n{strategy.upper()} 전략:")
        print(f"  연간 거래수: {cost_breakdown['trades_per_year']}회")
        print(f"  거래당 비용: {cost_breakdown['cost_per_trade']:,.0f}원")
        print(f"  연간 총 비용: {cost_breakdown['annual_total_cost']:,.0f}원")
        print(f"  연간 비용률: {cost_breakdown['annual_cost_ratio']*100:.2f}%")


if __name__ == "__main__":
    main()