"""
Whipsaw Filter - 횡보장에서의 거짓신호 방지
FEEDBACK.md의 지적사항을 반영하여 이평선 + RSI 의존성 문제를 해결합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """시장 상황 분류"""
    STRONG_UPTREND = "strong_uptrend"    # 강한 상승
    WEAK_UPTREND = "weak_uptrend"        # 약한 상승  
    SIDEWAYS = "sideways"                # 횡보
    WEAK_DOWNTREND = "weak_downtrend"    # 약한 하락
    STRONG_DOWNTREND = "strong_downtrend" # 강한 하락


@dataclass
class WhipsawAnalysis:
    """Whipsaw 분석 결과"""
    regime: MarketRegime
    whipsaw_risk: float        # 0.0 ~ 1.0, 높을수록 위험
    signal_reliability: float  # 0.0 ~ 1.0, 높을수록 신뢰성
    recommended_action: str    # 권장 행동
    
    # 세부 지표들
    atr_percentile: float      # ATR 백분위수
    price_efficiency: float    # 가격 효율성
    trend_consistency: float   # 추세 일관성
    volume_confirmation: float # 거래량 확인


class WhipsawFilter:
    """
    횡보장 거짓신호 필터링 시스템
    
    방법론:
    1. 시장 상황(Regime) 분류
    2. ATR 기반 변동성 분석
    3. 가격 효율성 측정 (Price Efficiency)
    4. 추세 일관성 검증
    5. 거래량 확인
    """
    
    def __init__(self):
        # 시장 상황 판단 기준
        self.regime_thresholds = {
            'strong_trend': 0.02,      # 강한 추세 기준 (2%)
            'weak_trend': 0.005,       # 약한 추세 기준 (0.5%)
            'high_volatility': 0.8,    # 고변동성 기준 (ATR 80%ile)
            'low_efficiency': 0.6      # 낮은 효율성 기준
        }
        
        # Whipsaw 위험도 가중치
        self.risk_weights = {
            'volatility': 0.3,     # 변동성 30%
            'efficiency': 0.25,    # 효율성 25%
            'consistency': 0.25,   # 일관성 25%
            'volume': 0.2          # 거래량 20%
        }
    
    def calculate_atr_percentile(self, 
                               high: pd.Series, 
                               low: pd.Series, 
                               close: pd.Series, 
                               period: int = 14) -> float:
        """
        ATR 백분위수 계산
        
        Args:
            high, low, close: 가격 데이터
            period: ATR 기간
            
        Returns:
            현재 ATR의 백분위수 (0.0 ~ 1.0)
        """
        # True Range 계산
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        if len(atr) < 60:  # 최소 60일 필요
            return 0.5  # 중간값 반환
        
        current_atr = atr.iloc[-1]
        atr_history = atr.dropna().iloc[-60:]  # 최근 60일
        
        percentile = (atr_history < current_atr).sum() / len(atr_history)
        return percentile
    
    def calculate_price_efficiency(self, 
                                 close: pd.Series, 
                                 period: int = 20) -> float:
        """
        가격 효율성 계산 (Price Efficiency)
        
        직선 거리 / 실제 이동 거리 비율
        1.0에 가까울수록 효율적 (추세), 0에 가까울수록 비효율적 (횡보)
        
        Args:
            close: 종가 데이터
            period: 계산 기간
            
        Returns:
            가격 효율성 (0.0 ~ 1.0)
        """
        if len(close) < period:
            return 0.5
        
        recent_close = close.iloc[-period:]
        
        # 직선 거리 (시작점 → 끝점)
        linear_distance = abs(recent_close.iloc[-1] - recent_close.iloc[0])
        
        # 실제 이동 거리 (누적 절댓값 변화량)
        actual_distance = recent_close.diff().abs().sum()
        
        if actual_distance == 0:
            return 0.0
        
        efficiency = linear_distance / actual_distance
        return min(1.0, efficiency)
    
    def calculate_trend_consistency(self, 
                                  ma_short: pd.Series, 
                                  ma_long: pd.Series,
                                  period: int = 10) -> float:
        """
        추세 일관성 계산
        
        최근 기간 동안 이평선 배열이 얼마나 일관되게 유지되었는지 측정
        
        Args:
            ma_short: 단기 이평선
            ma_long: 장기 이평선  
            period: 검증 기간
            
        Returns:
            추세 일관성 (0.0 ~ 1.0)
        """
        if len(ma_short) < period or len(ma_long) < period:
            return 0.5
        
        recent_short = ma_short.iloc[-period:]
        recent_long = ma_long.iloc[-period:]
        
        # 이평선 배열 상태 (1: 정배열, 0: 역배열)
        arrangement = (recent_short > recent_long).astype(int)
        
        # 일관성 = 같은 상태 유지 비율
        consistency = arrangement.rolling(window=min(5, period)).apply(
            lambda x: (x == x.iloc[-1]).sum() / len(x)
        ).mean()
        
        return consistency if not pd.isna(consistency) else 0.5
    
    def calculate_volume_confirmation(self, 
                                    close: pd.Series, 
                                    volume: pd.Series,
                                    period: int = 10) -> float:
        """
        거래량 확인 점수
        
        가격 상승시 거래량 증가, 하락시 거래량 감소하는지 확인
        
        Args:
            close: 종가 데이터
            volume: 거래량 데이터
            period: 확인 기간
            
        Returns:
            거래량 확인 점수 (0.0 ~ 1.0)
        """
        if len(close) < period + 1 or len(volume) < period + 1:
            return 0.5
        
        price_change = close.diff().iloc[-period:]
        volume_recent = volume.iloc[-period:]
        volume_avg = volume.rolling(window=20).mean().iloc[-period:]
        
        # 상대 거래량 (평균 대비)
        relative_volume = volume_recent / volume_avg
        
        # 가격 상승일과 거래량 증가 일치도
        price_up = (price_change > 0)
        volume_up = (relative_volume > 1.2)  # 20% 이상 증가
        
        if len(price_up[price_up]) == 0:  # 상승일이 없으면
            return 0.5
        
        # 상승일에 거래량도 증가한 비율
        confirmation_ratio = (price_up & volume_up).sum() / price_up.sum()
        
        return confirmation_ratio
    
    def classify_market_regime(self, 
                             ma_short: pd.Series, 
                             ma_long: pd.Series,
                             atr_percentile: float,
                             price_efficiency: float) -> MarketRegime:
        """
        시장 상황 분류
        
        Args:
            ma_short, ma_long: 단기/장기 이평선
            atr_percentile: ATR 백분위수
            price_efficiency: 가격 효율성
            
        Returns:
            MarketRegime
        """
        if len(ma_short) < 5 or len(ma_long) < 5:
            return MarketRegime.SIDEWAYS
        
        # 최근 이평선 기울기
        short_slope = (ma_short.iloc[-1] - ma_short.iloc[-5]) / ma_short.iloc[-5]
        long_slope = (ma_long.iloc[-1] - ma_long.iloc[-5]) / ma_long.iloc[-5]
        
        # 배열 상태
        is_bull_arrangement = ma_short.iloc[-1] > ma_long.iloc[-1]
        
        # 강한 추세 조건: 기울기 + 효율성 + 낮은 변동성
        strong_trend_condition = (
            abs(short_slope) > self.regime_thresholds['strong_trend'] and
            price_efficiency > 0.7 and
            atr_percentile < 0.7  # 너무 높은 변동성은 제외
        )
        
        # 약한 추세 조건  
        weak_trend_condition = (
            abs(short_slope) > self.regime_thresholds['weak_trend'] and
            price_efficiency > 0.4
        )
        
        # 분류 로직
        if strong_trend_condition:
            if is_bull_arrangement and short_slope > 0:
                return MarketRegime.STRONG_UPTREND
            elif not is_bull_arrangement and short_slope < 0:
                return MarketRegime.STRONG_DOWNTREND
        
        if weak_trend_condition:
            if is_bull_arrangement:
                return MarketRegime.WEAK_UPTREND
            else:
                return MarketRegime.WEAK_DOWNTREND
        
        return MarketRegime.SIDEWAYS
    
    def calculate_whipsaw_risk(self, 
                             atr_percentile: float,
                             price_efficiency: float,
                             trend_consistency: float,
                             volume_confirmation: float) -> float:
        """
        Whipsaw 위험도 계산
        
        Args:
            atr_percentile: ATR 백분위수
            price_efficiency: 가격 효율성  
            trend_consistency: 추세 일관성
            volume_confirmation: 거래량 확인
            
        Returns:
            위험도 (0.0 ~ 1.0, 높을수록 위험)
        """
        # 각 요소별 위험도 계산
        volatility_risk = atr_percentile  # 높은 변동성 = 높은 위험
        efficiency_risk = 1.0 - price_efficiency  # 낮은 효율성 = 높은 위험
        consistency_risk = 1.0 - trend_consistency  # 낮은 일관성 = 높은 위험
        volume_risk = 1.0 - volume_confirmation  # 낮은 확인 = 높은 위험
        
        # 가중 평균
        total_risk = (
            volatility_risk * self.risk_weights['volatility'] +
            efficiency_risk * self.risk_weights['efficiency'] +
            consistency_risk * self.risk_weights['consistency'] +
            volume_risk * self.risk_weights['volume']
        )
        
        return min(1.0, total_risk)
    
    def analyze_whipsaw_risk(self, 
                           high: pd.Series,
                           low: pd.Series, 
                           close: pd.Series,
                           volume: pd.Series,
                           ma_short: pd.Series,
                           ma_long: pd.Series) -> WhipsawAnalysis:
        """
        종합적인 Whipsaw 위험 분석
        
        Returns:
            WhipsawAnalysis 객체
        """
        # 각 지표 계산
        atr_percentile = self.calculate_atr_percentile(high, low, close)
        price_efficiency = self.calculate_price_efficiency(close)
        trend_consistency = self.calculate_trend_consistency(ma_short, ma_long)
        volume_confirmation = self.calculate_volume_confirmation(close, volume)
        
        # 시장 상황 분류
        regime = self.classify_market_regime(
            ma_short, ma_long, atr_percentile, price_efficiency
        )
        
        # Whipsaw 위험도 계산
        whipsaw_risk = self.calculate_whipsaw_risk(
            atr_percentile, price_efficiency, 
            trend_consistency, volume_confirmation
        )
        
        # 신호 신뢰성 (위험도의 역수)
        signal_reliability = 1.0 - whipsaw_risk
        
        # 권장 행동 결정
        if whipsaw_risk < 0.3:
            if regime in [MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND]:
                recommended_action = "적극적 매매"
            else:
                recommended_action = "보통 매매"
        elif whipsaw_risk < 0.6:
            recommended_action = "신중한 매매"
        else:
            recommended_action = "매매 자제"
        
        # 로깅
        logger.info(f"Whipsaw 분석 완료:")
        logger.info(f"  시장상황: {regime.value}")
        logger.info(f"  위험도: {whipsaw_risk:.2f}")
        logger.info(f"  신뢰성: {signal_reliability:.2f}")
        logger.info(f"  권장행동: {recommended_action}")
        
        return WhipsawAnalysis(
            regime=regime,
            whipsaw_risk=whipsaw_risk,
            signal_reliability=signal_reliability,
            recommended_action=recommended_action,
            atr_percentile=atr_percentile,
            price_efficiency=price_efficiency,
            trend_consistency=trend_consistency,
            volume_confirmation=volume_confirmation
        )
    
    def should_filter_signal(self, 
                           analysis: WhipsawAnalysis,
                           risk_tolerance: float = 0.5) -> Tuple[bool, str]:
        """
        신호 필터링 여부 결정
        
        Args:
            analysis: WhipsawAnalysis 객체
            risk_tolerance: 위험 허용 수준 (0.0 ~ 1.0)
            
        Returns:
            (필터링할지 여부, 이유)
        """
        if analysis.whipsaw_risk > risk_tolerance:
            reason = f"Whipsaw 위험도 {analysis.whipsaw_risk:.2f} > 허용수준 {risk_tolerance:.2f}"
            return True, reason
        
        if analysis.regime == MarketRegime.SIDEWAYS and analysis.price_efficiency < 0.4:
            reason = "횡보장에서 낮은 가격 효율성"
            return True, reason
        
        if analysis.volume_confirmation < 0.3:
            reason = "거래량 확인 불충분"
            return True, reason
        
        return False, "신호 유효"


def main():
    """테스트용 메인 함수"""
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # 횡보 패턴 생성
    base_price = 100
    noise = np.random.normal(0, 2, 100)
    trend = np.sin(np.linspace(0, 4*np.pi, 100)) * 5  # 사인파 패턴
    close = base_price + trend + noise
    
    high = close + np.random.uniform(0.5, 2, 100)
    low = close - np.random.uniform(0.5, 2, 100)
    volume = np.random.uniform(1000000, 5000000, 100)
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        'date': dates,
        'high': high,
        'low': low, 
        'close': close,
        'volume': volume
    })
    
    # 이평선 계산
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    
    # Whipsaw 필터 테스트
    filter_system = WhipsawFilter()
    
    analysis = filter_system.analyze_whipsaw_risk(
        df['high'], df['low'], df['close'], df['volume'],
        df['ma5'], df['ma20']
    )
    
    # 결과 출력
    print("=== Whipsaw 분석 결과 ===")
    print(f"시장 상황: {analysis.regime.value}")
    print(f"Whipsaw 위험도: {analysis.whipsaw_risk:.2f}")
    print(f"신호 신뢰성: {analysis.signal_reliability:.2f}")
    print(f"권장 행동: {analysis.recommended_action}")
    
    print("\n=== 세부 지표 ===")
    print(f"ATR 백분위수: {analysis.atr_percentile:.2f}")
    print(f"가격 효율성: {analysis.price_efficiency:.2f}")
    print(f"추세 일관성: {analysis.trend_consistency:.2f}")
    print(f"거래량 확인: {analysis.volume_confirmation:.2f}")
    
    # 필터링 테스트
    should_filter, reason = filter_system.should_filter_signal(analysis)
    print(f"\n=== 필터링 결과 ===")
    print(f"신호 필터링: {'YES' if should_filter else 'NO'}")
    print(f"사유: {reason}")


if __name__ == "__main__":
    main()