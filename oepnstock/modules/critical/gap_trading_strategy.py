"""
Gap Trading Strategy - Critical Phase 1 Module
갭(Gap) 대응 전략: 시가 갭 발생 시 적절한 대응 전략 자동 적용
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from enum import Enum
import numpy as np
import pandas as pd

from ...config import config
from ...utils import get_logger
from ...utils.korean_market import KoreanMarketUtils

logger = get_logger(__name__)


class GapType(Enum):
    """Gap types"""
    NO_GAP = "no_gap"
    MINOR_GAP_UP = "minor_gap_up"      # 1-3%
    MAJOR_GAP_UP = "major_gap_up"      # 3%+
    MINOR_GAP_DOWN = "minor_gap_down"  # -1% ~ -3%
    MAJOR_GAP_DOWN = "major_gap_down"  # -3% 이하


class GapStrategy(Enum):
    """Gap trading strategies"""
    WAIT_FOR_PULLBACK = "wait_for_pullback"
    MOMENTUM_PLAY = "momentum_play"
    RECALCULATE_ALL = "recalculate_all"
    NORMAL_ENTRY = "normal_entry"
    NO_TRADE = "no_trade"


@dataclass
class GapAnalysis:
    """Gap analysis result"""
    symbol: str
    yesterday_close: float
    today_open: float
    gap_ratio: float
    gap_type: GapType
    gap_size_won: float
    fill_probability: float
    volume_spike: bool
    market_sentiment: str  # 'positive', 'negative', 'neutral'
    volatility_context: str  # 'low', 'normal', 'high'


@dataclass
class GapStrategy:
    """Gap trading strategy recommendation"""
    strategy_type: str
    entry_points: List[Dict[str, Any]]
    stop_loss: Dict[str, float]
    position_adjustment: float  # Position size multiplier
    max_wait_time: Optional[str]  # Time limit for entry
    risk_level: str
    confidence: float
    reasoning: List[str]
    warnings: List[str]


@dataclass
class GapTradingPlan:
    """Complete gap trading plan"""
    symbol: str
    analysis: GapAnalysis
    strategy: GapStrategy
    original_plan_modification: Dict[str, Any]
    monitoring_alerts: List[Dict[str, Any]]
    exit_conditions: List[str]
    success_probability: float


class GapTradingStrategy:
    """
    갭 대응 전략 시스템
    
    주요 기능:
    - 갭 유형 분석 (상승/하락, 대/소)
    - 갭 채우기 확률 계산
    - 상황별 대응 전략 결정
    - 기존 매매 계획 수정
    - 실시간 모니터링 설정
    """
    
    def __init__(self):
        self.config = config.trading
        
        # Gap thresholds
        self.gap_thresholds = {
            'major_gap_up': 0.03,      # 3% 이상
            'minor_gap_up': 0.01,      # 1% 이상
            'minor_gap_down': -0.01,   # -1% 이하
            'major_gap_down': -0.03    # -3% 이하
        }
        
        # Gap fill probabilities (historical statistics)
        self.gap_fill_probabilities = {
            GapType.MINOR_GAP_UP: 0.75,    # 75%
            GapType.MAJOR_GAP_UP: 0.60,    # 60%
            GapType.MINOR_GAP_DOWN: 0.80,  # 80%
            GapType.MAJOR_GAP_DOWN: 0.45   # 45%
        }
        
        # Strategy configurations
        self.strategy_configs = {
            'wait_for_pullback': {
                'position_multiplier': 0.8,
                'max_wait_hours': 4,
                'entry_levels': ['gap_fill', 'half_gap_fill', 'minor_pullback'],
                'stop_loss_type': 'below_gap'
            },
            'momentum_play': {
                'position_multiplier': 1.0,
                'max_wait_hours': 1,
                'entry_levels': ['breakout_confirmation', 'first_pullback'],
                'stop_loss_type': 'opening_price'
            },
            'recalculate_all': {
                'position_multiplier': 0.5,
                'max_wait_hours': None,
                'entry_levels': ['new_support_levels'],
                'stop_loss_type': 'tight'
            }
        }
        
        # Market context weights for gap analysis
        self.context_weights = {
            'gap_size': 0.4,
            'volume': 0.3,
            'market_sentiment': 0.2,
            'volatility': 0.1
        }
    
    def analyze_gap(self, symbol: str, yesterday_close: float, today_open: float,
                   market_data: Optional[Dict] = None) -> GapAnalysis:
        """
        갭 분석 수행
        
        Args:
            symbol: 종목 코드
            yesterday_close: 전일 종가
            today_open: 당일 시가
            market_data: 추가 시장 데이터 (거래량, 변동성 등)
            
        Returns:
            GapAnalysis: 갭 분석 결과
        """
        logger.info(f"Analyzing gap for {symbol}: {yesterday_close} -> {today_open}")
        
        try:
            # 1. 기본 갭 계산
            gap_ratio = (today_open - yesterday_close) / yesterday_close
            gap_size_won = today_open - yesterday_close
            
            # 2. 갭 유형 분류
            gap_type = self._classify_gap_type(gap_ratio)
            
            # 3. 갭 채우기 확률 계산
            fill_probability = self._calculate_gap_fill_probability(gap_type, gap_ratio, market_data)
            
            # 4. 거래량 급증 여부
            volume_spike = self._detect_volume_spike(symbol, market_data)
            
            # 5. 시장 심리 분석
            market_sentiment = self._analyze_market_sentiment(gap_type, market_data)
            
            # 6. 변동성 맥락 분석
            volatility_context = self._analyze_volatility_context(symbol, gap_ratio, market_data)
            
            analysis = GapAnalysis(
                symbol=symbol,
                yesterday_close=yesterday_close,
                today_open=today_open,
                gap_ratio=gap_ratio,
                gap_type=gap_type,
                gap_size_won=gap_size_won,
                fill_probability=fill_probability,
                volume_spike=volume_spike,
                market_sentiment=market_sentiment,
                volatility_context=volatility_context
            )
            
            logger.info(f"Gap analysis complete for {symbol} - Type: {gap_type.value}, "
                       f"Fill probability: {fill_probability:.1%}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing gap for {symbol}: {e}")
            raise
    
    def determine_gap_strategy(self, gap_analysis: GapAnalysis,
                             current_time: Optional[datetime] = None) -> GapStrategy:
        """
        갭 분석 결과를 바탕으로 거래 전략 결정
        
        Args:
            gap_analysis: 갭 분석 결과
            current_time: 현재 시간
            
        Returns:
            GapStrategy: 권장 거래 전략
        """
        if current_time is None:
            current_time = datetime.now()
        
        gap_type = gap_analysis.gap_type
        fill_probability = gap_analysis.fill_probability
        
        logger.info(f"Determining strategy for {gap_type.value} with fill probability {fill_probability:.1%}")
        
        try:
            # 전략 결정 로직
            if gap_type == GapType.NO_GAP:
                return self._normal_entry_strategy(gap_analysis)
            
            elif gap_type == GapType.MAJOR_GAP_UP:
                if fill_probability > 0.7:
                    return self._wait_for_pullback_strategy(gap_analysis, current_time)
                else:
                    return self._momentum_play_strategy(gap_analysis, current_time)
            
            elif gap_type == GapType.MINOR_GAP_UP:
                if gap_analysis.volume_spike and gap_analysis.market_sentiment == 'positive':
                    return self._momentum_play_strategy(gap_analysis, current_time)
                else:
                    return self._wait_for_pullback_strategy(gap_analysis, current_time)
            
            elif gap_type in [GapType.MAJOR_GAP_DOWN, GapType.MINOR_GAP_DOWN]:
                return self._gap_down_strategy(gap_analysis, current_time)
            
            else:
                return self._no_trade_strategy(gap_analysis, "Unknown gap type")
                
        except Exception as e:
            logger.error(f"Error determining gap strategy: {e}")
            return self._no_trade_strategy(gap_analysis, f"Strategy error: {e}")
    
    def create_gap_trading_plan(self, symbol: str, yesterday_close: float, today_open: float,
                              original_trading_plan: Dict[str, Any],
                              market_data: Optional[Dict] = None) -> GapTradingPlan:
        """
        종합적인 갭 거래 계획 수립
        
        Args:
            symbol: 종목 코드
            yesterday_close: 전일 종가
            today_open: 당일 시가
            original_trading_plan: 기존 거래 계획
            market_data: 시장 데이터
            
        Returns:
            GapTradingPlan: 완전한 갭 거래 계획
        """
        logger.info(f"Creating gap trading plan for {symbol}")
        
        try:
            # 1. 갭 분석
            analysis = self.analyze_gap(symbol, yesterday_close, today_open, market_data)
            
            # 2. 전략 결정
            strategy = self.determine_gap_strategy(analysis)
            
            # 3. 기존 계획 수정
            plan_modifications = self._modify_original_plan(original_trading_plan, analysis, strategy)
            
            # 4. 모니터링 알림 설정
            monitoring_alerts = self._setup_gap_monitoring(analysis, strategy)
            
            # 5. 청산 조건 설정
            exit_conditions = self._define_gap_exit_conditions(analysis, strategy)
            
            # 6. 성공 확률 계산
            success_probability = self._calculate_success_probability(analysis, strategy)
            
            plan = GapTradingPlan(
                symbol=symbol,
                analysis=analysis,
                strategy=strategy,
                original_plan_modification=plan_modifications,
                monitoring_alerts=monitoring_alerts,
                exit_conditions=exit_conditions,
                success_probability=success_probability
            )
            
            logger.info(f"Gap trading plan created - Strategy: {strategy.strategy_type}, "
                       f"Success probability: {success_probability:.1%}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating gap trading plan for {symbol}: {e}")
            raise
    
    # Private strategy methods
    
    def _wait_for_pullback_strategy(self, analysis: GapAnalysis, current_time: datetime) -> GapStrategy:
        """상승 갭 되돌림 대기 전략"""
        gap_fill_price = analysis.yesterday_close
        half_gap_price = analysis.yesterday_close + (analysis.gap_size_won * 0.5)
        minor_pullback_price = analysis.today_open * 0.98  # 2% 되돌림
        
        entry_points = [
            {
                'level': 'gap_fill',
                'price': gap_fill_price,
                'size_ratio': 0.5,
                'priority': 1,
                'description': '완전 갭 채우기'
            },
            {
                'level': 'half_gap_fill', 
                'price': half_gap_price,
                'size_ratio': 0.3,
                'priority': 2,
                'description': '절반 갭 채우기'
            },
            {
                'level': 'minor_pullback',
                'price': minor_pullback_price,
                'size_ratio': 0.2,
                'priority': 3,
                'description': '경미한 되돌림'
            }
        ]
        
        # 시간 제한 설정 (오전 중 진입)
        max_wait_time = "11:00" if current_time.hour < 11 else "14:00"
        
        return GapStrategy(
            strategy_type='wait_for_pullback',
            entry_points=entry_points,
            stop_loss={
                'type': 'below_gap',
                'price': analysis.yesterday_close * 0.98,
                'percentage': 0.02
            },
            position_adjustment=0.8,  # 포지션 20% 축소
            max_wait_time=max_wait_time,
            risk_level='medium',
            confidence=analysis.fill_probability,
            reasoning=[
                f"갭 채우기 확률 {analysis.fill_probability:.1%}",
                "추격 매수 금지, 되돌림 매수만",
                f"갭 크기: {analysis.gap_ratio:.1%}"
            ],
            warnings=[
                "갭 채우기 실패 시 추세 전환 가능성",
                f"{max_wait_time} 이후 전략 재검토 필요"
            ]
        )
    
    def _momentum_play_strategy(self, analysis: GapAnalysis, current_time: datetime) -> GapStrategy:
        """상승 갭 모멘텀 전략"""
        breakout_price = analysis.today_open * 1.02  # 시가 대비 2% 상승
        pullback_price = analysis.today_open * 0.995  # 시가 근처 되돌림
        
        entry_points = [
            {
                'level': 'breakout_confirmation',
                'price': breakout_price,
                'size_ratio': 0.3,
                'priority': 1,
                'description': '돌파 확인 후 진입'
            },
            {
                'level': 'first_pullback',
                'price': pullback_price,
                'size_ratio': 0.7,
                'priority': 2,
                'description': '첫 번째 되돌림'
            }
        ]
        
        return GapStrategy(
            strategy_type='momentum_play',
            entry_points=entry_points,
            stop_loss={
                'type': 'opening_price',
                'price': analysis.today_open,
                'percentage': 0.0
            },
            position_adjustment=1.0,
            max_wait_time="10:00",  # 오전 중 빠른 진입
            risk_level='high',
            confidence=1 - analysis.fill_probability,  # 갭이 안 채워질 확률
            reasoning=[
                "갭 유지 시 추세 추종",
                f"거래량 급증: {analysis.volume_spike}",
                f"시장 심리: {analysis.market_sentiment}"
            ],
            warnings=[
                "갭 채우기 시 손실 위험",
                "빠른 손절 필요"
            ]
        )
    
    def _gap_down_strategy(self, analysis: GapAnalysis, current_time: datetime) -> GapStrategy:
        """하락 갭 대응 전략"""
        return GapStrategy(
            strategy_type='recalculate_all',
            entry_points=[
                {
                    'level': 'new_support_levels',
                    'price': 0,  # 새로운 지지선 계산 필요
                    'size_ratio': 0.5,
                    'priority': 1,
                    'description': '새 지지선에서만 진입'
                }
            ],
            stop_loss={
                'type': 'tight',
                'price': analysis.today_open * 0.985,  # 시가 대비 -1.5%
                'percentage': 0.015
            },
            position_adjustment=0.5,  # 포지션 절반으로 축소
            max_wait_time=None,
            risk_level='high',
            confidence=0.3,  # 낮은 신뢰도
            reasoning=[
                "하락 갭은 추세 전환 신호일 수 있음",
                "기존 지지선 무효화 가능성",
                "포지션 크기 대폭 축소"
            ],
            warnings=[
                "전면적 계획 재수립 필요",
                "추가 하락 시 즉시 손절",
                "시장 상황 악화 가능성"
            ]
        )
    
    def _normal_entry_strategy(self, analysis: GapAnalysis) -> GapStrategy:
        """갭 없는 일반 진입 전략"""
        return GapStrategy(
            strategy_type='normal_entry',
            entry_points=[
                {
                    'level': 'original_plan',
                    'price': analysis.today_open,
                    'size_ratio': 1.0,
                    'priority': 1,
                    'description': '기존 계획대로 진입'
                }
            ],
            stop_loss={
                'type': 'original',
                'price': analysis.today_open * 0.97,
                'percentage': 0.03
            },
            position_adjustment=1.0,
            max_wait_time=None,
            risk_level='normal',
            confidence=0.8,
            reasoning=['갭 없음으로 기존 계획 유지'],
            warnings=[]
        )
    
    def _no_trade_strategy(self, analysis: GapAnalysis, reason: str) -> GapStrategy:
        """거래 중지 전략"""
        return GapStrategy(
            strategy_type='no_trade',
            entry_points=[],
            stop_loss={'type': 'none', 'price': 0, 'percentage': 0},
            position_adjustment=0.0,
            max_wait_time=None,
            risk_level='critical',
            confidence=0.0,
            reasoning=[reason],
            warnings=['전체 거래 중지 권고']
        )
    
    # Helper methods
    
    def _classify_gap_type(self, gap_ratio: float) -> GapType:
        """갭 비율로부터 갭 유형 분류"""
        if gap_ratio >= self.gap_thresholds['major_gap_up']:
            return GapType.MAJOR_GAP_UP
        elif gap_ratio >= self.gap_thresholds['minor_gap_up']:
            return GapType.MINOR_GAP_UP
        elif gap_ratio <= self.gap_thresholds['major_gap_down']:
            return GapType.MAJOR_GAP_DOWN
        elif gap_ratio <= self.gap_thresholds['minor_gap_down']:
            return GapType.MINOR_GAP_DOWN
        else:
            return GapType.NO_GAP
    
    def _calculate_gap_fill_probability(self, gap_type: GapType, gap_ratio: float,
                                      market_data: Optional[Dict] = None) -> float:
        """갭 채우기 확률 계산"""
        base_probability = self.gap_fill_probabilities.get(gap_type, 0.5)
        
        # 갭 크기별 조정
        if abs(gap_ratio) > 0.05:  # 5% 이상
            base_probability *= 0.8
        elif abs(gap_ratio) < 0.015:  # 1.5% 미만
            base_probability *= 1.2
        
        # 시장 데이터 기반 조정
        if market_data:
            # 거래량 영향
            volume_ratio = market_data.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                base_probability *= 0.9  # 큰 거래량은 갭 유지 가능성 증가
            
            # 시장 지수 상황
            market_gap = market_data.get('market_index_gap', 0)
            if abs(market_gap) > 0.01:  # 시장 전체 갭
                base_probability *= 0.85  # 시장 갭은 채우기 어려움
        
        return max(0.1, min(0.9, base_probability))
    
    def _detect_volume_spike(self, symbol: str, market_data: Optional[Dict] = None) -> bool:
        """거래량 급증 감지"""
        if not market_data:
            return False
        
        volume_ratio = market_data.get('volume_ratio', 1.0)
        return volume_ratio > 2.0  # 평균 대비 2배 이상
    
    def _analyze_market_sentiment(self, gap_type: GapType, market_data: Optional[Dict] = None) -> str:
        """시장 심리 분석"""
        if not market_data:
            return 'neutral'
        
        # 시장 지수 움직임
        kospi_change = market_data.get('kospi_change', 0)
        kosdaq_change = market_data.get('kosdaq_change', 0)
        avg_market_change = (kospi_change + kosdaq_change) / 2
        
        if avg_market_change > 0.01:
            return 'positive'
        elif avg_market_change < -0.01:
            return 'negative'
        else:
            return 'neutral'
    
    def _analyze_volatility_context(self, symbol: str, gap_ratio: float,
                                  market_data: Optional[Dict] = None) -> str:
        """변동성 맥락 분석"""
        # VIX나 변동성 지표 활용
        vix_level = market_data.get('vix', 20) if market_data else 20
        
        if vix_level > 30:
            return 'high'
        elif vix_level < 15:
            return 'low'
        else:
            return 'normal'
    
    def _modify_original_plan(self, original_plan: Dict[str, Any], 
                            analysis: GapAnalysis, strategy: GapStrategy) -> Dict[str, Any]:
        """기존 거래 계획 수정"""
        modifications = {
            'original_entry_price': original_plan.get('entry_price'),
            'original_position_size': original_plan.get('position_size'),
            'modified': True
        }
        
        if strategy.strategy_type == 'wait_for_pullback':
            modifications.update({
                'entry_type': 'limit_orders',
                'entry_prices': [ep['price'] for ep in strategy.entry_points],
                'position_size_adjustment': strategy.position_adjustment,
                'time_limit': strategy.max_wait_time
            })
        
        elif strategy.strategy_type == 'momentum_play':
            modifications.update({
                'entry_type': 'conditional_market',
                'breakout_level': strategy.entry_points[0]['price'],
                'position_size_adjustment': strategy.position_adjustment
            })
        
        elif strategy.strategy_type == 'recalculate_all':
            modifications.update({
                'recalculate_support_levels': True,
                'position_size_adjustment': strategy.position_adjustment,
                'tighten_stop_loss': True,
                'reduce_targets': True
            })
        
        return modifications
    
    def _setup_gap_monitoring(self, analysis: GapAnalysis, strategy: GapStrategy) -> List[Dict]:
        """갭 모니터링 알림 설정"""
        alerts = []
        
        # 갭 채우기 알림
        if analysis.gap_type in [GapType.MINOR_GAP_UP, GapType.MAJOR_GAP_UP]:
            alerts.append({
                'type': 'gap_fill_watch',
                'trigger_price': analysis.yesterday_close,
                'message': f'{analysis.symbol} 갭 채우기 도달',
                'action': '진입 기회 검토'
            })
            
            # 절반 갭 채우기
            half_gap_price = analysis.yesterday_close + (analysis.gap_size_won * 0.5)
            alerts.append({
                'type': 'half_gap_fill',
                'trigger_price': half_gap_price,
                'message': f'{analysis.symbol} 절반 갭 채우기',
                'action': '부분 진입 고려'
            })
        
        # 시간 기반 알림
        if strategy.max_wait_time:
            alerts.append({
                'type': 'time_limit_warning',
                'trigger_time': strategy.max_wait_time,
                'message': f'{analysis.symbol} 갭 전략 시간 만료 임박',
                'action': '전략 재검토'
            })
        
        return alerts
    
    def _define_gap_exit_conditions(self, analysis: GapAnalysis, strategy: GapStrategy) -> List[str]:
        """갭 전략 청산 조건 정의"""
        conditions = []
        
        # 공통 조건
        conditions.append("손절선 터치 시 즉시 청산")
        
        # 갭 유형별 특수 조건
        if analysis.gap_type == GapType.MAJOR_GAP_UP:
            if strategy.strategy_type == 'wait_for_pullback':
                conditions.extend([
                    "갭 완전 채우기 시 75% 물량 진입",
                    "시간 만료 시 전략 포기",
                    "추가 상승 시 트레일링 스탑 적용"
                ])
            elif strategy.strategy_type == 'momentum_play':
                conditions.extend([
                    "갭 채우기 시작 시 즉시 청산",
                    "목표가 도달 시 50% 청산"
                ])
        
        elif analysis.gap_type in [GapType.MAJOR_GAP_DOWN, GapType.MINOR_GAP_DOWN]:
            conditions.extend([
                "추가 하락 시 즉시 전량 청산",
                "새로운 지지선 하향 돌파 시 청산",
                "장중 5% 추가 하락 시 무조건 청산"
            ])
        
        return conditions
    
    def _calculate_success_probability(self, analysis: GapAnalysis, strategy: GapStrategy) -> float:
        """전략 성공 확률 계산"""
        base_probability = strategy.confidence
        
        # 시장 상황 조정
        if analysis.market_sentiment == 'positive' and analysis.gap_type in [GapType.MINOR_GAP_UP, GapType.MAJOR_GAP_UP]:
            base_probability *= 1.1
        elif analysis.market_sentiment == 'negative':
            base_probability *= 0.9
        
        # 거래량 조정
        if analysis.volume_spike:
            if strategy.strategy_type == 'momentum_play':
                base_probability *= 1.2
            else:
                base_probability *= 0.9  # 되돌림 전략에는 불리
        
        # 변동성 조정
        if analysis.volatility_context == 'high':
            base_probability *= 0.8  # 높은 변동성은 예측 어려움
        
        return max(0.1, min(0.9, base_probability))