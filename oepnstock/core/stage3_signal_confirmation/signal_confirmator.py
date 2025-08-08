"""
Signal Confirmation - Stage 3 of 4-Stage Trading Strategy
진짜 반등 신호 확인하기: 거래량, 캔들패턴, 보조지표 종합 분석
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, time
import numpy as np
import pandas as pd

from ...config import config
from ...utils import get_logger
from ...utils.korean_market import KoreanMarketUtils

logger = get_logger(__name__)


@dataclass
class VolumeSignal:
    """Volume analysis result"""
    current_volume: int
    avg_volume: int
    volume_ratio: float
    time_strength: float  # 시간대별 거래량 강도
    spike_detected: bool
    volume_score: float


@dataclass
class CandleSignal:
    """Candle pattern analysis result"""
    pattern_type: str  # 'hammer', 'doji', 'engulfing', 'normal'
    pattern_strength: float
    body_ratio: float
    tail_ratio: float
    pattern_score: float
    reversal_probability: float


@dataclass
class IndicatorSignal:
    """Technical indicator signals"""
    rsi: float
    rsi_signal: str  # 'oversold', 'normal', 'overbought'
    macd_histogram: float
    macd_signal: str  # 'bullish', 'bearish', 'neutral'
    stochastic_signal: str
    divergence_detected: bool
    indicator_score: float


@dataclass
class SignalConfirmation:
    """Complete signal confirmation result"""
    symbol: str
    timestamp: datetime
    total_signal_strength: float
    weighted_score: float
    action: str  # 'immediate_buy', 'split_entry', 'wait', 'no_trade'
    confidence_level: str  # 'high', 'medium', 'low'
    volume_signal: VolumeSignal
    candle_signal: CandleSignal
    indicator_signal: IndicatorSignal
    market_condition_adjustment: float
    reasons: List[str]
    warnings: List[str]


class SignalConfirmator:
    """
    3단계: '진짜 반등' 신호 확인하기 (방아쇠 당기기)
    
    주요 기능:
    - 거래량 분석 (시간대별 강도 포함)
    - 캔들 패턴 인식 (망치형, 도지형, 상승장악형)
    - 보조지표 신호 확인 (RSI, MACD, 스토캐스틱)
    - 동적 가중치 시스템 (시장 상황별 조정)
    """
    
    def __init__(self):
        self.config = config.trading
        self.technical_config = config.technical
        
        # Base signal weights (will be adjusted dynamically)
        self.base_weights = {
            'volume': 0.5,      # 50% - 가장 중요
            'candle': 0.3,      # 30%
            'indicators': 0.2   # 20%
        }
        
        # Pattern recognition parameters
        self.candle_patterns = {
            'hammer': {
                'min_tail_ratio': 2.0,    # 아래꼬리가 몸통의 2배 이상
                'max_upper_tail': 0.1,    # 위꼬리는 몸통의 10% 이하
                'strength_multiplier': 1.5
            },
            'doji': {
                'max_body_ratio': 0.005,  # 시가-종가 차이 0.5% 이내
                'tail_ratio_range': (0.7, 1.3),  # 위아래 꼬리 비율 0.7~1.3
                'strength_multiplier': 1.2
            },
            'engulfing': {
                'min_body_coverage': 1.0,  # 전일 음봉을 완전히 감싸야 함
                'volume_confirmation': True,  # 거래량 동반 필수
                'strength_multiplier': 1.8
            }
        }
        
        # Time-of-day volume analysis
        self.time_volume_multipliers = {
            'pre_opening': 0.8,
            'opening_auction': 2.0,     # 오전장 조기 거래량 폭증은 매우 강한 신호
            'morning_volatility': 1.5,
            'morning_trend': 1.2,
            'lunch_lull': 0.6,
            'afternoon_trend': 1.0,
            'closing_volatility': 1.3,
            'closing_auction': 1.1
        }
    
    def confirm_signal(self, 
                      data: pd.DataFrame,
                      symbol: str, 
                      support_price: float,
                      market_condition: str = 'sideways',
                      current_time: Optional[datetime] = None) -> SignalConfirmation:
        """
        종합적인 신호 확인 분석
        
        Args:
            data: OHLCV 데이터
            symbol: 종목 코드
            support_price: 지지선 가격
            market_condition: 시장 상황 ('strong_uptrend', 'weak_uptrend', 'sideways', 'downtrend')
            current_time: 현재 시간 (시간대별 분석용)
            
        Returns:
            SignalConfirmation: 신호 확인 결과
        """
        logger.info(f"Starting signal confirmation for {symbol}")
        
        if len(data) < 20:
            raise ValueError(f"Insufficient data for signal confirmation: {len(data)} < 20")
        
        current_time = current_time or datetime.now()
        
        try:
            # 1. 거래량 분석
            volume_signal = self._analyze_volume(data, current_time)
            
            # 2. 캔들 패턴 분석
            candle_signal = self._analyze_candle_pattern(data, support_price)
            
            # 3. 보조지표 분석
            indicator_signal = self._analyze_indicators(data)
            
            # 4. 동적 가중치 적용
            weights = self._calculate_dynamic_weights(market_condition)
            
            # 5. 종합 점수 계산
            total_strength, weighted_score = self._calculate_total_strength(
                volume_signal, candle_signal, indicator_signal, weights
            )
            
            # 6. 시장 상황별 조정
            market_adjustment = self._apply_market_condition_adjustment(
                total_strength, market_condition
            )
            
            final_score = weighted_score * market_adjustment
            
            # 7. 액션 결정
            action, confidence = self._determine_action(final_score, market_condition)
            
            # 8. 판단 근거 및 경고 생성
            reasons = self._generate_reasons(volume_signal, candle_signal, indicator_signal)
            warnings = self._generate_warnings(volume_signal, candle_signal, indicator_signal, market_condition)
            
            confirmation = SignalConfirmation(
                symbol=symbol,
                timestamp=current_time,
                total_signal_strength=total_strength,
                weighted_score=final_score,
                action=action,
                confidence_level=confidence,
                volume_signal=volume_signal,
                candle_signal=candle_signal, 
                indicator_signal=indicator_signal,
                market_condition_adjustment=market_adjustment,
                reasons=reasons,
                warnings=warnings
            )
            
            logger.info(f"Signal confirmation complete for {symbol} - Score: {final_score:.1f}, Action: {action}")
            return confirmation
            
        except Exception as e:
            logger.error(f"Error in signal confirmation for {symbol}: {e}")
            raise
    
    def _analyze_volume(self, data: pd.DataFrame, current_time: datetime) -> VolumeSignal:
        """거래량 분석 (시간대별 강도 포함)"""
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(self.technical_config.volume_ma_period).mean().iloc[-1]
        prev_volume = data['volume'].iloc[-2] if len(data) >= 2 else current_volume
        
        # 기본 거래량 비율
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        prev_volume_ratio = current_volume / prev_volume if prev_volume > 0 else 1.0
        
        # 시간대별 거래량 강도 분석
        time_strength = self._analyze_intraday_volume_strength(current_time, current_volume, avg_volume)
        
        # 거래량 급증 감지
        spike_detected = (
            volume_ratio >= self.technical_config.volume_spike_threshold and
            prev_volume_ratio >= 2.0  # 전일 대비 200% 이상
        )
        
        # 장중 거래량 분포 분석 (실제 구현시 분봉 데이터 필요)
        intraday_distribution_score = self._analyze_intraday_distribution(data)
        
        # 거래량 점수 계산
        volume_score = self._calculate_volume_score(
            volume_ratio, time_strength, spike_detected, intraday_distribution_score
        )
        
        return VolumeSignal(
            current_volume=int(current_volume),
            avg_volume=int(avg_volume),
            volume_ratio=volume_ratio,
            time_strength=time_strength,
            spike_detected=spike_detected,
            volume_score=volume_score
        )
    
    def _analyze_intraday_volume_strength(self, current_time: datetime, 
                                        current_volume: float, avg_volume: float) -> float:
        """시간대별 거래량 강도 분석"""
        market_session = KoreanMarketUtils.get_current_market_session(current_time)
        
        if not market_session:
            return 1.0  # 장외 시간
        
        session_multiplier = self.time_volume_multipliers.get(market_session.value, 1.0)
        
        # 시간 경과 비율 (장 시작 후 몇 % 시점인지)
        market_open_time = current_time.replace(hour=9, minute=0, second=0)
        market_close_time = current_time.replace(hour=15, minute=30, second=0)
        
        if current_time >= market_open_time and current_time <= market_close_time:
            time_elapsed_ratio = (current_time - market_open_time).total_seconds() / \
                               (market_close_time - market_open_time).total_seconds()
            
            expected_volume_ratio = time_elapsed_ratio  # 기본적으로 시간 비례
            actual_volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 조기 거래량 폭증은 더 강한 신호
            if time_elapsed_ratio <= 0.3:  # 장 시작 후 30% 이내
                if actual_volume_ratio > expected_volume_ratio * 2:
                    session_multiplier *= 1.5
        
        return session_multiplier
    
    def _analyze_intraday_distribution(self, data: pd.DataFrame) -> float:
        """장중 거래량 분포 점수 (현재는 간단한 구현)"""
        # 실제로는 분봉 데이터가 필요하지만, 일봉으로 근사치 계산
        recent_volumes = data['volume'].tail(5)
        current_volume = data['volume'].iloc[-1]
        
        # 최근 5일 대비 현재 거래량의 상대적 위치
        volume_percentile = (recent_volumes <= current_volume).mean()
        
        # 60% 이상 지점에서 발생하면 강한 신호
        if volume_percentile >= 0.6:
            return 1.2
        elif volume_percentile >= 0.4:
            return 1.0
        else:
            return 0.8
    
    def _calculate_volume_score(self, volume_ratio: float, time_strength: float,
                              spike_detected: bool, intraday_score: float) -> float:
        """거래량 종합 점수 계산"""
        base_score = min(volume_ratio / self.technical_config.volume_spike_threshold, 2.0) * 50
        
        # 시간대별 강도 적용
        time_adjusted_score = base_score * time_strength
        
        # 거래량 급증 보너스
        if spike_detected:
            time_adjusted_score *= 1.3
        
        # 장중 분포 보정
        final_score = time_adjusted_score * intraday_score
        
        return min(final_score, 100.0)
    
    def _analyze_candle_pattern(self, data: pd.DataFrame, support_price: float) -> CandleSignal:
        """캔들 패턴 분석"""
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) >= 2 else current
        
        open_price = current['open']
        high_price = current['high']
        low_price = current['low']  
        close_price = current['close']
        
        # 캔들 구성요소 계산
        body = abs(close_price - open_price)
        upper_tail = high_price - max(open_price, close_price)
        lower_tail = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        # 비율 계산
        body_ratio = body / total_range if total_range > 0 else 0
        lower_tail_ratio = lower_tail / body if body > 0 else 0
        upper_tail_ratio = upper_tail / body if body > 0 else 0
        
        # 지지선 관련 정보
        support_touch = abs(low_price - support_price) / support_price < 0.01  # 1% 이내
        recovery_strength = (close_price - low_price) / (high_price - low_price) if high_price > low_price else 0
        
        # 패턴 인식
        pattern_type, pattern_strength = self._identify_candle_pattern(
            body_ratio, lower_tail_ratio, upper_tail_ratio, 
            support_touch, recovery_strength, current, prev
        )
        
        # 반등 확률 계산
        reversal_probability = self._calculate_reversal_probability(
            pattern_type, pattern_strength, support_touch, recovery_strength
        )
        
        # 캔들 점수 계산
        pattern_score = pattern_strength * 100
        
        return CandleSignal(
            pattern_type=pattern_type,
            pattern_strength=pattern_strength,
            body_ratio=body_ratio,
            tail_ratio=lower_tail_ratio,
            pattern_score=pattern_score,
            reversal_probability=reversal_probability
        )
    
    def _identify_candle_pattern(self, body_ratio: float, lower_tail_ratio: float, 
                               upper_tail_ratio: float, support_touch: bool,
                               recovery_strength: float, current: pd.Series, 
                               prev: pd.Series) -> Tuple[str, float]:
        """캔들 패턴 식별"""
        
        # 망치형 (Hammer) 체크
        hammer_params = self.candle_patterns['hammer']
        if (lower_tail_ratio >= hammer_params['min_tail_ratio'] and
            upper_tail_ratio <= hammer_params['max_upper_tail'] and
            support_touch):
            strength = min(lower_tail_ratio / hammer_params['min_tail_ratio'], 2.0) * \
                      hammer_params['strength_multiplier'] * 0.5
            return 'hammer', strength
        
        # 도지형 (Doji) 체크
        doji_params = self.candle_patterns['doji']
        if body_ratio <= doji_params['max_body_ratio']:
            tail_ratio = lower_tail_ratio / upper_tail_ratio if upper_tail_ratio > 0 else float('inf')
            min_ratio, max_ratio = doji_params['tail_ratio_range']
            
            if min_ratio <= tail_ratio <= max_ratio or tail_ratio == float('inf'):
                strength = doji_params['strength_multiplier'] * 0.4
                if support_touch:
                    strength *= 1.2
                return 'doji', strength
        
        # 상승장악형 (Bullish Engulfing) 체크  
        engulfing_params = self.candle_patterns['engulfing']
        if len(current) > 0 and len(prev) > 0:
            prev_body = abs(prev['close'] - prev['open'])
            current_body = abs(current['close'] - current['open'])
            
            # 전일 음봉, 당일 양봉
            prev_bearish = prev['close'] < prev['open']
            current_bullish = current['close'] > current['open']
            
            # 완전 포함 여부
            engulfs_prev = (current['open'] <= prev['close'] and 
                           current['close'] >= prev['open'])
            
            if (prev_bearish and current_bullish and engulfs_prev and 
                current_body >= prev_body * engulfing_params['min_body_coverage']):
                strength = engulfing_params['strength_multiplier'] * 0.6
                return 'engulfing', strength
        
        # 일반적인 양봉/음봉
        if current['close'] > current['open']:
            strength = min(body_ratio * 2, 1.0) * 0.3  # 양봉
            if support_touch:
                strength *= 1.1
            return 'bullish', strength
        else:
            return 'bearish', 0.1  # 음봉은 약한 신호
    
    def _calculate_reversal_probability(self, pattern_type: str, pattern_strength: float,
                                      support_touch: bool, recovery_strength: float) -> float:
        """반등 확률 계산"""
        base_probabilities = {
            'hammer': 0.70,
            'doji': 0.60,
            'engulfing': 0.75,
            'bullish': 0.55,
            'bearish': 0.25
        }
        
        base_prob = base_probabilities.get(pattern_type, 0.50)
        
        # 지지선 터치 보너스
        if support_touch:
            base_prob *= 1.1
        
        # 회복 강도 보정
        base_prob *= (0.8 + recovery_strength * 0.4)  # 0.8 ~ 1.2 범위
        
        # 패턴 강도 보정
        base_prob *= (0.7 + pattern_strength * 0.6)   # 0.7 ~ 1.3 범위
        
        return min(base_prob, 0.95)  # 최대 95%
    
    def _analyze_indicators(self, data: pd.DataFrame) -> IndicatorSignal:
        """보조지표 분석"""
        # RSI 계산
        rsi = self._calculate_rsi(data['close'])
        rsi_signal = self._classify_rsi_signal(rsi)
        
        # MACD 계산
        macd_line, signal_line, histogram = self._calculate_macd(data['close'])
        macd_signal = self._classify_macd_signal(macd_line, signal_line, histogram)
        
        # 스토캐스틱 계산
        stoch_k, stoch_d = self._calculate_stochastic(data)
        stochastic_signal = self._classify_stochastic_signal(stoch_k, stoch_d)
        
        # 다이버전스 감지
        divergence_detected = self._detect_divergence(data, rsi)
        
        # 종합 지표 점수
        indicator_score = self._calculate_indicator_score(
            rsi, rsi_signal, macd_signal, stochastic_signal, divergence_detected
        )
        
        return IndicatorSignal(
            rsi=rsi,
            rsi_signal=rsi_signal,
            macd_histogram=histogram,
            macd_signal=macd_signal,
            stochastic_signal=stochastic_signal,
            divergence_detected=divergence_detected,
            indicator_score=indicator_score
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0  # 중립값
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _classify_rsi_signal(self, rsi: float) -> str:
        """RSI 신호 분류"""
        if rsi <= self.technical_config.rsi_oversold:
            return 'oversold'
        elif rsi >= self.technical_config.rsi_overbought:
            return 'overbought'
        else:
            return 'normal'
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD 계산"""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def _classify_macd_signal(self, macd_line: float, signal_line: float, histogram: float) -> str:
        """MACD 신호 분류"""
        if histogram > 0 and macd_line > signal_line:
            return 'bullish'
        elif histogram < 0 and macd_line < signal_line:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """스토캐스틱 계산"""
        if len(data) < k_period:
            return 50.0, 50.0
        
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent.iloc[-1], d_percent.iloc[-1]
    
    def _classify_stochastic_signal(self, k_percent: float, d_percent: float) -> str:
        """스토캐스틱 신호 분류"""
        if k_percent <= 20 and d_percent <= 20 and k_percent > d_percent:
            return 'oversold_bullish'
        elif k_percent >= 80 and d_percent >= 80 and k_percent < d_percent:
            return 'overbought_bearish'
        elif k_percent > d_percent:
            return 'bullish'
        else:
            return 'bearish'
    
    def _detect_divergence(self, data: pd.DataFrame, rsi: float) -> bool:
        """다이버전스 감지 (간단한 구현)"""
        if len(data) < 10:
            return False
        
        # 최근 10일간 가격과 RSI의 방향성 비교
        recent_prices = data['close'].tail(10)
        recent_rsi = [self._calculate_rsi(data['close'].iloc[:i+1]) for i in range(len(data)-10, len(data))]
        
        if len(recent_rsi) < 5:
            return False
        
        # 가격은 신저점, RSI는 저점 높임 (Bullish Divergence)
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
        
        # 가격 하락, RSI 상승 시 bullish divergence
        return price_trend < -0.01 and rsi_trend > 0.5
    
    def _calculate_indicator_score(self, rsi: float, rsi_signal: str, macd_signal: str,
                                 stochastic_signal: str, divergence_detected: bool) -> float:
        """지표 종합 점수 계산"""
        score = 0.0
        
        # RSI 점수
        if rsi_signal == 'oversold':
            score += 40
        elif rsi_signal == 'normal':
            score += 20
        
        # MACD 점수
        if macd_signal == 'bullish':
            score += 30
        elif macd_signal == 'neutral':
            score += 15
        
        # 스토캐스틱 점수
        if 'bullish' in stochastic_signal:
            score += 20
        elif 'oversold' in stochastic_signal:
            score += 30
        
        # 다이버전스 보너스
        if divergence_detected:
            score += 20
        
        return min(score, 100.0)
    
    def _calculate_dynamic_weights(self, market_condition: str) -> Dict[str, float]:
        """시장 상황에 따른 동적 가중치 계산"""
        if market_condition == 'strong_uptrend':
            return {
                'volume': 0.4,    # 거래량 중요도 낮춤
                'candle': 0.4,    # 캔들 패턴 중요도 높임
                'indicators': 0.2
            }
        elif market_condition == 'sideways':
            return {
                'volume': 0.5,
                'candle': 0.2,
                'indicators': 0.3  # 과매도 지표 중요
            }
        else:  # downtrend
            return {
                'volume': 0.6,    # 거래량 매우 중요
                'candle': 0.2,
                'indicators': 0.2
            }
    
    def _calculate_total_strength(self, volume_signal: VolumeSignal, 
                                candle_signal: CandleSignal,
                                indicator_signal: IndicatorSignal,
                                weights: Dict[str, float]) -> Tuple[float, float]:
        """종합 신호 강도 계산"""
        # 기본 점수들
        volume_score = volume_signal.volume_score
        candle_score = candle_signal.pattern_score
        indicator_score = indicator_signal.indicator_score
        
        # 가중 평균
        weighted_score = (
            volume_score * weights['volume'] +
            candle_score * weights['candle'] +
            indicator_score * weights['indicators']
        )
        
        # 시간대별 거래량 강도 추가 보정
        time_adjusted_score = weighted_score * volume_signal.time_strength
        
        total_strength = (volume_score + candle_score + indicator_score) / 3
        
        return total_strength, time_adjusted_score
    
    def _apply_market_condition_adjustment(self, signal_strength: float, market_condition: str) -> float:
        """시장 상황별 신호 강도 조정"""
        adjustments = {
            'strong_uptrend': 1.1,    # 강세장에서는 신호 강화
            'weak_uptrend': 1.0,
            'sideways': 0.9,          # 횡보장에서는 신호 약화
            'downtrend': 0.7          # 하락장에서는 신호 대폭 약화
        }
        
        return adjustments.get(market_condition, 1.0)
    
    def _determine_action(self, final_score: float, market_condition: str) -> Tuple[str, str]:
        """최종 액션 및 신뢰도 결정"""
        # 시장 상황별 임계값 조정
        if market_condition == 'strong_uptrend':
            immediate_threshold = 70
            split_threshold = 50
        elif market_condition == 'downtrend':
            immediate_threshold = 90  # 매우 까다로움
            split_threshold = 75
        else:
            immediate_threshold = self.config.immediate_buy_threshold
            split_threshold = self.config.split_entry_threshold
        
        # 액션 결정
        if final_score >= immediate_threshold:
            action = 'immediate_buy'
            confidence = 'high'
        elif final_score >= split_threshold:
            action = 'split_entry'
            confidence = 'medium'
        elif final_score >= self.config.min_signal_threshold:
            action = 'wait'
            confidence = 'low'
        else:
            action = 'no_trade'
            confidence = 'low'
        
        return action, confidence
    
    def _generate_reasons(self, volume_signal: VolumeSignal, 
                         candle_signal: CandleSignal,
                         indicator_signal: IndicatorSignal) -> List[str]:
        """판단 근거 생성"""
        reasons = []
        
        # 거래량 근거
        if volume_signal.spike_detected:
            reasons.append(f"거래량 급증 감지 ({volume_signal.volume_ratio:.1f}배)")
        
        if volume_signal.time_strength > 1.2:
            reasons.append("시간대별 거래량 강도 높음")
        
        # 캔들 패턴 근거
        if candle_signal.pattern_type in ['hammer', 'doji', 'engulfing']:
            reasons.append(f"{candle_signal.pattern_type} 패턴 감지 (강도: {candle_signal.pattern_strength:.2f})")
        
        # 지표 근거
        if indicator_signal.rsi_signal == 'oversold':
            reasons.append(f"RSI 과매도 구간 ({indicator_signal.rsi:.1f})")
        
        if indicator_signal.macd_signal == 'bullish':
            reasons.append("MACD 상승 신호")
        
        if indicator_signal.divergence_detected:
            reasons.append("Bullish Divergence 감지")
        
        return reasons
    
    def _generate_warnings(self, volume_signal: VolumeSignal,
                          candle_signal: CandleSignal, 
                          indicator_signal: IndicatorSignal,
                          market_condition: str) -> List[str]:
        """경고 메시지 생성"""
        warnings = []
        
        # 거래량 경고
        if volume_signal.volume_ratio < 1.2:
            warnings.append("거래량 부족 - 신호 신뢰도 낮음")
        
        # 캔들 패턴 경고
        if candle_signal.pattern_type == 'bearish':
            warnings.append("음봉 패턴 - 반등 신호 약함")
        
        # 지표 경고
        if indicator_signal.rsi > 70:
            warnings.append("RSI 과매수 구간 - 추가 상승 제한적")
        
        # 시장 상황 경고
        if market_condition == 'downtrend':
            warnings.append("하락장에서의 신호 - 매우 신중한 접근 필요")
        
        return warnings