"""
Enhanced Signal Engine - 다변화된 기술적 신호 시스템
FEEDBACK.md의 지적사항을 반영하여 이평선+RSI 의존성을 해결합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """신호 강도"""
    VERY_STRONG = "very_strong"    # 90-100점
    STRONG = "strong"              # 80-89점
    MODERATE = "moderate"          # 60-79점
    WEAK = "weak"                  # 40-59점
    VERY_WEAK = "very_weak"        # 0-39점


@dataclass
class TechnicalSignals:
    """기술적 신호 모음"""
    # 전통적 지표
    ma_signal: float           # 이동평균 신호
    rsi_signal: float          # RSI 신호
    macd_signal: float         # MACD 신호
    
    # 추가된 지표들
    bollinger_signal: float    # 볼린저밴드 신호
    stochastic_signal: float   # 스토캐스틱 신호
    williams_r_signal: float   # Williams %R 신호
    cci_signal: float          # CCI 신호
    momentum_signal: float     # 모멘텀 신호
    
    # 패턴 분석
    candlestick_signal: float  # 캔들패턴 신호
    volume_signal: float       # 거래량 신호
    
    # 종합 점수
    total_score: float         # 가중 평균 점수
    signal_strength: SignalStrength
    confidence: float          # 신뢰도 (0-1)


class EnhancedSignalEngine:
    """
    다변화된 기술적 신호 엔진
    
    특징:
    1. 11개 독립적 기술지표 활용
    2. 시장 상황별 동적 가중치
    3. 신호 상관관계 분석
    4. 노이즈 필터링
    """
    
    def __init__(self):
        # 기본 가중치 (중립적 시장 기준)
        self.base_weights = {
            'ma_signal': 0.15,         # 15% - 추세
            'rsi_signal': 0.12,        # 12% - 과매수/과매도
            'macd_signal': 0.13,       # 13% - 모멘텀
            'bollinger_signal': 0.10,  # 10% - 변동성 돌파
            'stochastic_signal': 0.10, # 10% - 오실레이터
            'williams_r_signal': 0.08, # 8% - 오실레이터 2
            'cci_signal': 0.08,        # 8% - 사이클
            'momentum_signal': 0.08,   # 8% - 순수 모멘텀
            'candlestick_signal': 0.08, # 8% - 패턴
            'volume_signal': 0.08      # 8% - 거래량
        }
        
        # 시장 상황별 가중치 조정
        self.market_adjustments = {
            'trending': {
                'ma_signal': 1.3,      # 추세 지표 강화
                'momentum_signal': 1.2,
                'rsi_signal': 0.8,     # 오실레이터 약화
                'stochastic_signal': 0.8
            },
            'ranging': {
                'ma_signal': 0.7,      # 추세 지표 약화
                'rsi_signal': 1.3,     # 오실레이터 강화
                'stochastic_signal': 1.3,
                'bollinger_signal': 1.2, # 변동성 돌파 강화
                'williams_r_signal': 1.2
            },
            'volatile': {
                'candlestick_signal': 1.3, # 패턴 분석 강화
                'volume_signal': 1.5,      # 거래량 신호 강화
                'cci_signal': 1.2,         # CCI 강화
                'bollinger_signal': 0.8    # 볼린저밴드 약화
            }
        }
    
    def calculate_ma_signal(self, 
                          close: pd.Series, 
                          short_period: int = 5, 
                          long_period: int = 20) -> float:
        """이동평균 신호 (기존 방식 개선)"""
        if len(close) < long_period:
            return 50
        
        ma_short = close.rolling(short_period).mean()
        ma_long = close.rolling(long_period).mean()
        
        current_short = ma_short.iloc[-1]
        current_long = ma_long.iloc[-1]
        current_price = close.iloc[-1]
        
        # 1. 이평선 배열 점수 (40점)
        if current_short > current_long:
            arrangement_score = 70 + min(30, (current_short/current_long - 1) * 3000)
        else:
            arrangement_score = 30 - min(30, (current_long/current_short - 1) * 3000)
        
        # 2. 가격 위치 점수 (30점)
        if current_price > current_short:
            position_score = 65 + min(35, (current_price/current_short - 1) * 1000)
        else:
            position_score = 35 - min(35, (current_short/current_price - 1) * 1000)
        
        return max(0, min(100, (arrangement_score + position_score) / 2))
    
    def calculate_bollinger_signal(self, close: pd.Series, period: int = 20, std: float = 2) -> float:
        """볼린저밴드 신호"""
        if len(close) < period:
            return 50
        
        ma = close.rolling(period).mean()
        rolling_std = close.rolling(period).std()
        upper_band = ma + (rolling_std * std)
        lower_band = ma - (rolling_std * std)
        
        current_price = close.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_ma = ma.iloc[-1]
        
        # 밴드 내 위치 계산 (0~100)
        band_position = ((current_price - current_lower) / 
                        (current_upper - current_lower)) * 100
        
        # 돌파 신호 가중치
        if current_price > current_upper:
            signal = min(100, 85 + (current_price / current_upper - 1) * 500)
        elif current_price < current_lower:
            signal = max(0, 15 - (current_lower / current_price - 1) * 500)
        else:
            # 중심선 기준 신호
            if current_price > current_ma:
                signal = 50 + (band_position - 50) * 0.6
            else:
                signal = 50 + (band_position - 50) * 0.6
        
        return max(0, min(100, signal))
    
    def calculate_stochastic_signal(self, 
                                  high: pd.Series, 
                                  low: pd.Series, 
                                  close: pd.Series,
                                  k_period: int = 14, 
                                  d_period: int = 3) -> float:
        """스토캐스틱 신호"""
        if len(close) < k_period + d_period:
            return 50
        
        # %K 계산
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D 계산 (3일 이평)
        d_percent = k_percent.rolling(d_period).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        prev_k = k_percent.iloc[-2] if len(k_percent) > 1 else current_k
        prev_d = d_percent.iloc[-2] if len(d_percent) > 1 else current_d
        
        # 기본 신호 (과매수/과매도)
        if current_k < 20 and current_d < 20:
            base_signal = 80  # 과매도 매수신호
        elif current_k > 80 and current_d > 80:
            base_signal = 20  # 과매수 매도신호
        else:
            base_signal = current_k  # 중간값
        
        # 골든크로스/데드크로스 조정
        if current_k > current_d and prev_k <= prev_d:  # 골든크로스
            base_signal = min(100, base_signal + 15)
        elif current_k < current_d and prev_k >= prev_d:  # 데드크로스
            base_signal = max(0, base_signal - 15)
        
        return base_signal
    
    def calculate_williams_r_signal(self, 
                                  high: pd.Series, 
                                  low: pd.Series, 
                                  close: pd.Series,
                                  period: int = 14) -> float:
        """Williams %R 신호"""
        if len(close) < period:
            return 50
        
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        current_wr = williams_r.iloc[-1]
        
        # Williams %R을 0-100 스케일로 변환
        if current_wr <= -80:  # 과매도
            signal = 80 + (current_wr + 80) * 1.0  # -80~-100 → 80~100
        elif current_wr >= -20:  # 과매수  
            signal = 20 + (current_wr + 20) * 1.0  # 0~-20 → 0~20
        else:  # 중간
            signal = 50 + (current_wr + 50) * 0.5  # -50 중심
        
        return max(0, min(100, signal))
    
    def calculate_cci_signal(self, 
                           high: pd.Series, 
                           low: pd.Series, 
                           close: pd.Series,
                           period: int = 20) -> float:
        """CCI (Commodity Channel Index) 신호"""
        if len(close) < period:
            return 50
        
        # Typical Price
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period).mean()
        
        # Mean Deviation
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        # CCI 계산
        cci = (tp - sma_tp) / (0.015 * mad)
        current_cci = cci.iloc[-1]
        
        # CCI를 0-100 스케일로 변환
        if current_cci > 100:  # 과매수
            signal = max(0, 20 - (current_cci - 100) * 0.1)
        elif current_cci < -100:  # 과매도
            signal = min(100, 80 + (current_cci + 100) * 0.1)
        else:  # 중간 (-100 ~ 100 → 20 ~ 80)
            signal = 50 + (current_cci / 100) * 30
        
        return max(0, min(100, signal))
    
    def calculate_momentum_signal(self, close: pd.Series, period: int = 10) -> float:
        """순수 모멘텀 신호"""
        if len(close) < period + 1:
            return 50
        
        # 가격 모멘텀
        momentum = (close / close.shift(period) - 1) * 100
        current_momentum = momentum.iloc[-1]
        
        # ROC (Rate of Change) 기반 신호
        if current_momentum > 5:  # 5% 이상 상승
            signal = min(100, 70 + current_momentum * 2)
        elif current_momentum < -5:  # 5% 이상 하락
            signal = max(0, 30 + current_momentum * 2)
        else:  # 중간
            signal = 50 + current_momentum * 4
        
        return max(0, min(100, signal))
    
    def calculate_candlestick_signal(self, 
                                   open_prices: pd.Series,
                                   high: pd.Series, 
                                   low: pd.Series, 
                                   close: pd.Series) -> float:
        """캔들패턴 신호 (단순화된 버전)"""
        if len(close) < 3:
            return 50
        
        # 최근 3일 데이터
        recent_open = open_prices.iloc[-3:].values
        recent_high = high.iloc[-3:].values
        recent_low = low.iloc[-3:].values
        recent_close = close.iloc[-3:].values
        
        signal = 50  # 기본값
        
        # 간단한 패턴들
        for i in range(len(recent_close)):
            o, h, l, c = recent_open[i], recent_high[i], recent_low[i], recent_close[i]
            
            # 바디와 그림자 크기
            body_size = abs(c - o)
            total_size = h - l
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            
            if total_size == 0:
                continue
            
            # 망치형 (Hammer) - 매수신호
            if (lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5 and
                i == len(recent_close) - 1):  # 최신 캔들
                signal += 15
            
            # 교수형 (Hanging Man) - 매도신호
            elif (lower_shadow > body_size * 2 and 
                  upper_shadow < body_size * 0.5 and
                  c > o and  # 양봉이어야 함
                  i == len(recent_close) - 1):
                signal -= 10
            
            # 도지 (Doji) - 불확실성
            elif body_size < total_size * 0.1:
                signal += 0  # 중립
            
            # 강한 양봉
            elif c > o and body_size > total_size * 0.7:
                signal += 8
            
            # 강한 음봉  
            elif c < o and body_size > total_size * 0.7:
                signal -= 8
        
        return max(0, min(100, signal))
    
    def calculate_volume_signal(self, 
                              close: pd.Series, 
                              volume: pd.Series,
                              period: int = 20) -> float:
        """거래량 신호"""
        if len(volume) < period:
            return 50
        
        # 거래량 이동평균
        volume_ma = volume.rolling(period).mean()
        current_volume = volume.iloc[-1]
        current_volume_ma = volume_ma.iloc[-1]
        
        # 가격 변화
        price_change = (close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0
        
        # 상대 거래량
        volume_ratio = current_volume / current_volume_ma if current_volume_ma > 0 else 1
        
        # 거래량-가격 관계 분석
        if price_change > 0.01:  # 1% 이상 상승
            if volume_ratio > 1.5:  # 거래량 50% 증가
                signal = 80  # 강한 매수신호
            elif volume_ratio > 1.2:
                signal = 70  # 보통 매수신호
            else:
                signal = 55  # 약한 매수신호
        elif price_change < -0.01:  # 1% 이상 하락
            if volume_ratio > 1.5:  # 거래량 증가하며 하락
                signal = 20  # 강한 매도신호
            elif volume_ratio < 0.8:  # 거래량 감소하며 하락
                signal = 60  # 약한 하락 (매수 기회)
            else:
                signal = 35  # 보통 매도신호
        else:  # 횡보
            if volume_ratio > 2.0:  # 거래량 급증
                signal = 65  # 돌파 준비
            else:
                signal = 50  # 중립
        
        return max(0, min(100, signal))
    
    def adjust_weights_by_market_condition(self, 
                                         market_condition: str) -> Dict[str, float]:
        """시장 상황별 가중치 조정"""
        adjusted_weights = self.base_weights.copy()
        
        if market_condition in self.market_adjustments:
            adjustments = self.market_adjustments[market_condition]
            
            for signal_name, adjustment in adjustments.items():
                if signal_name in adjusted_weights:
                    adjusted_weights[signal_name] *= adjustment
        
        # 정규화 (합계가 1.0이 되도록)
        total_weight = sum(adjusted_weights.values())
        for key in adjusted_weights:
            adjusted_weights[key] /= total_weight
        
        return adjusted_weights
    
    def calculate_signal_correlation(self, signals_dict: Dict[str, float]) -> float:
        """신호간 상관관계 기반 신뢰도 계산"""
        signal_values = list(signals_dict.values())
        
        # 신호들이 모두 같은 방향인지 확인
        bullish_signals = sum(1 for s in signal_values if s > 60)
        bearish_signals = sum(1 for s in signal_values if s < 40)
        neutral_signals = len(signal_values) - bullish_signals - bearish_signals
        
        total_signals = len(signal_values)
        
        # 일관성 점수
        if bullish_signals > total_signals * 0.7:
            consistency = bullish_signals / total_signals
        elif bearish_signals > total_signals * 0.7:
            consistency = bearish_signals / total_signals
        else:
            consistency = neutral_signals / total_signals * 0.5  # 중립적 일관성
        
        return consistency
    
    def generate_enhanced_signals(self, 
                                ohlcv_data: pd.DataFrame,
                                market_condition: str = 'neutral') -> TechnicalSignals:
        """
        종합 기술적 신호 생성
        
        Args:
            ohlcv_data: OHLCV 데이터 (open, high, low, close, volume 컬럼 필요)
            market_condition: 시장 상황 ('trending', 'ranging', 'volatile', 'neutral')
            
        Returns:
            TechnicalSignals 객체
        """
        # 필수 컬럼 확인
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in ohlcv_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 각 신호 계산
        signals = {}
        signals['ma_signal'] = self.calculate_ma_signal(ohlcv_data['close'])
        signals['rsi_signal'] = self.calculate_rsi_signal(ohlcv_data['close'])
        signals['macd_signal'] = self.calculate_macd_signal(ohlcv_data['close'])
        signals['bollinger_signal'] = self.calculate_bollinger_signal(ohlcv_data['close'])
        signals['stochastic_signal'] = self.calculate_stochastic_signal(
            ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
        )
        signals['williams_r_signal'] = self.calculate_williams_r_signal(
            ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
        )
        signals['cci_signal'] = self.calculate_cci_signal(
            ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
        )
        signals['momentum_signal'] = self.calculate_momentum_signal(ohlcv_data['close'])
        signals['candlestick_signal'] = self.calculate_candlestick_signal(
            ohlcv_data['open'], ohlcv_data['high'], 
            ohlcv_data['low'], ohlcv_data['close']
        )
        signals['volume_signal'] = self.calculate_volume_signal(
            ohlcv_data['close'], ohlcv_data['volume']
        )
        
        # 시장 상황별 가중치 조정
        weights = self.adjust_weights_by_market_condition(market_condition)
        
        # 가중 평균 계산
        total_score = sum(signals[key] * weights[key] for key in signals.keys())
        
        # 신호 강도 결정
        if total_score >= 90:
            strength = SignalStrength.VERY_STRONG
        elif total_score >= 80:
            strength = SignalStrength.STRONG
        elif total_score >= 60:
            strength = SignalStrength.MODERATE
        elif total_score >= 40:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.VERY_WEAK
        
        # 신뢰도 계산
        confidence = self.calculate_signal_correlation(signals)
        
        # 로깅
        logger.info(f"Enhanced 신호 생성 완료:")
        logger.info(f"  총점: {total_score:.1f}")
        logger.info(f"  강도: {strength.value}")
        logger.info(f"  신뢰도: {confidence:.2f}")
        logger.info(f"  시장상황: {market_condition}")
        
        return TechnicalSignals(
            ma_signal=signals['ma_signal'],
            rsi_signal=signals['rsi_signal'],
            macd_signal=signals['macd_signal'],
            bollinger_signal=signals['bollinger_signal'],
            stochastic_signal=signals['stochastic_signal'],
            williams_r_signal=signals['williams_r_signal'],
            cci_signal=signals['cci_signal'],
            momentum_signal=signals['momentum_signal'],
            candlestick_signal=signals['candlestick_signal'],
            volume_signal=signals['volume_signal'],
            total_score=total_score,
            signal_strength=strength,
            confidence=confidence
        )
    
    def calculate_rsi_signal(self, close: pd.Series, period: int = 14) -> float:
        """RSI 신호 (기존 방식 유지)"""
        if len(close) < period + 1:
            return 50
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # RSI 기반 신호 (역방향)
        if current_rsi < 30:
            return min(100, 80 + (30 - current_rsi) * 2)
        elif current_rsi > 70:
            return max(0, 20 - (current_rsi - 70) * 2)
        else:
            return 50 + (50 - current_rsi) * 0.5
    
    def calculate_macd_signal(self, close: pd.Series) -> float:
        """MACD 신호"""
        if len(close) < 26:
            return 50
        
        # EMA 계산
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        
        # MACD 라인
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        # MACD 신호 계산
        base_signal = 50
        
        # MACD 라인 위치
        if current_macd > current_signal:
            base_signal += 20  # 매수 신호
        else:
            base_signal -= 20  # 매도 신호
        
        # 히스토그램 변화
        if len(histogram) > 1:
            prev_histogram = histogram.iloc[-2]
            if current_histogram > prev_histogram:
                base_signal += 10  # 상승 모멘텀
            else:
                base_signal -= 10  # 하락 모멘텀
        
        return max(0, min(100, base_signal))


def main():
    """테스트용 메인 함수"""
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # 상승 추세 데이터
    base_price = 100
    trend = np.linspace(0, 20, 100)  # 20% 상승
    noise = np.random.normal(0, 1, 100)
    close = base_price + trend + noise
    
    # OHLCV 데이터 생성
    high = close + np.random.uniform(0.5, 2, 100)
    low = close - np.random.uniform(0.5, 2, 100)
    open_price = np.concatenate([[close[0]], close[:-1]]) + np.random.normal(0, 0.5, 100)
    volume = np.random.uniform(1000000, 3000000, 100)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # Enhanced Signal Engine 테스트
    engine = EnhancedSignalEngine()
    
    # 다양한 시장 상황에서 테스트
    market_conditions = ['neutral', 'trending', 'ranging', 'volatile']
    
    for condition in market_conditions:
        print(f"\n=== {condition.upper()} 시장 조건 ===")
        
        signals = engine.generate_enhanced_signals(df, condition)
        
        print(f"총 점수: {signals.total_score:.1f}")
        print(f"신호 강도: {signals.signal_strength.value}")
        print(f"신뢰도: {signals.confidence:.2f}")
        
        print("개별 신호:")
        print(f"  MA: {signals.ma_signal:.1f}")
        print(f"  RSI: {signals.rsi_signal:.1f}")
        print(f"  MACD: {signals.macd_signal:.1f}")
        print(f"  Bollinger: {signals.bollinger_signal:.1f}")
        print(f"  Stochastic: {signals.stochastic_signal:.1f}")
        print(f"  Williams %R: {signals.williams_r_signal:.1f}")
        print(f"  CCI: {signals.cci_signal:.1f}")
        print(f"  Momentum: {signals.momentum_signal:.1f}")
        print(f"  Candlestick: {signals.candlestick_signal:.1f}")
        print(f"  Volume: {signals.volume_signal:.1f}")


if __name__ == "__main__":
    main()