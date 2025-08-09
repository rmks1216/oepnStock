"""
Market Score Calculator - 투명한 시장 흐름 점수 산출
FEEDBACK.md의 지적사항을 반영하여 투명성과 신뢰성을 확보합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketScoreComponents:
    """Market Score 구성 요소"""
    index_position_score: float  # 지수 위치 점수 (0-40점)
    ma_slope_score: float       # 이평선 기울기 점수 (0-30점)
    volatility_score: float     # 변동성 점수 (0-30점)
    total_score: float         # 총 점수 (0-100점)
    
    # 세부 구성요소 (투명성 확보)
    index_vs_ma5: float        # 지수 vs MA5 비교
    index_vs_ma20: float       # 지수 vs MA20 비교
    ma20_slope: float          # MA20 기울기
    daily_change: float        # 일간 변동률
    volatility_rank: float     # 변동성 순위


class MarketScoreCalculator:
    """
    투명한 Market Score 계산기
    
    산출 방식:
    1. 지수 위치 (40점): KOSPI vs MA5(20점) + vs MA20(20점)
    2. MA 기울기 (30점): 20일 이평선의 20일간 기울기
    3. 변동성 (30점): 일간 변동률 기준, -2% 이상시 만점
    """
    
    def __init__(self):
        self.score_weights = {
            'index_position': 40,  # 지수 위치
            'ma_slope': 30,       # 이평선 기울기  
            'volatility': 30      # 변동성
        }
        
        # 점수 산출 기준값들
        self.thresholds = {
            'ma5_bonus': 0.02,     # MA5 대비 2% 이상시 보너스
            'ma20_bonus': 0.01,    # MA20 대비 1% 이상시 보너스
            'slope_strong': 0.005, # 강한 상승 기울기 기준
            'volatility_min': -0.02 # 최소 변동성 기준 -2%
        }
    
    def calculate_index_position_score(self, 
                                     current_index: float,
                                     ma5: float, 
                                     ma20: float) -> Tuple[float, Dict[str, float]]:
        """
        지수 위치 점수 계산 (40점 만점)
        
        Args:
            current_index: 현재 지수값
            ma5: 5일 이평선
            ma20: 20일 이평선
            
        Returns:
            점수, 세부정보
        """
        details = {}
        
        # MA5 대비 위치 (20점)
        ma5_ratio = (current_index - ma5) / ma5
        details['ma5_ratio'] = ma5_ratio
        
        if ma5_ratio > self.thresholds['ma5_bonus']:
            ma5_score = 20  # 2% 이상 초과시 만점
        elif ma5_ratio > 0:
            ma5_score = 15 + (ma5_ratio / self.thresholds['ma5_bonus']) * 5
        else:
            ma5_score = max(0, 15 + ma5_ratio * 150)  # 하락시 페널티
        
        # MA20 대비 위치 (20점)  
        ma20_ratio = (current_index - ma20) / ma20
        details['ma20_ratio'] = ma20_ratio
        
        if ma20_ratio > self.thresholds['ma20_bonus']:
            ma20_score = 20  # 1% 이상 초과시 만점
        elif ma20_ratio > 0:
            ma20_score = 15 + (ma20_ratio / self.thresholds['ma20_bonus']) * 5
        else:
            ma20_score = max(0, 15 + ma20_ratio * 200)  # 하락시 페널티
            
        total_score = ma5_score + ma20_score
        details['ma5_score'] = ma5_score
        details['ma20_score'] = ma20_score
        
        return min(40, total_score), details
    
    def calculate_ma_slope_score(self, ma20_series: pd.Series) -> Tuple[float, Dict[str, float]]:
        """
        이평선 기울기 점수 계산 (30점 만점)
        
        Args:
            ma20_series: 20일 이평선 시계열 (최근 20일)
            
        Returns:
            점수, 세부정보
        """
        details = {}
        
        if len(ma20_series) < 20:
            logger.warning("MA20 데이터 부족, 기본 점수 적용")
            return 15, {'slope': 0, 'trend': 'insufficient_data'}
        
        # 20일간 기울기 계산 (선형회귀 기울기)
        x = np.arange(len(ma20_series))
        y = ma20_series.values
        
        # 선형회귀로 기울기 계산
        slope = np.polyfit(x, y, 1)[0]
        normalized_slope = slope / ma20_series.iloc[-1]  # 정규화
        
        details['raw_slope'] = slope
        details['normalized_slope'] = normalized_slope
        
        # 점수 계산
        if normalized_slope > self.thresholds['slope_strong']:
            score = 30  # 강한 상승 추세
            trend = 'strong_uptrend'
        elif normalized_slope > 0:
            score = 15 + (normalized_slope / self.thresholds['slope_strong']) * 15
            trend = 'weak_uptrend'
        elif normalized_slope > -self.thresholds['slope_strong']:
            score = 15 + normalized_slope * 3000  # 약한 하락
            trend = 'sideways'
        else:
            score = max(0, 15 + normalized_slope * 1500)  # 강한 하락
            trend = 'downtrend'
        
        details['trend'] = trend
        return min(30, max(0, score)), details
    
    def calculate_volatility_score(self, daily_change: float) -> Tuple[float, Dict[str, float]]:
        """
        변동성 점수 계산 (30점 만점)
        
        Args:
            daily_change: 일간 변동률 (-1.0 ~ 1.0)
            
        Returns:
            점수, 세부정보
        """
        details = {'daily_change': daily_change}
        
        if daily_change >= self.thresholds['volatility_min']:
            # -2% 이상이면 만점 (거래 가능한 시장)
            score = 30
            status = 'good_volatility'
        elif daily_change >= -0.04:
            # -4%까지는 부분 점수
            score = 30 * (1 + (daily_change + 0.04) / 0.02)
            status = 'acceptable_volatility'
        else:
            # -4% 미만은 위험한 시장
            score = max(0, 10 * (1 + (daily_change + 0.06) / 0.02))
            status = 'high_risk_volatility'
        
        details['volatility_status'] = status
        return min(30, max(0, score)), details
    
    def calculate_market_score(self, 
                             current_index: float,
                             ma5: float,
                             ma20: float, 
                             ma20_series: pd.Series,
                             daily_change: float) -> MarketScoreComponents:
        """
        종합 Market Score 계산
        
        Args:
            current_index: 현재 지수값
            ma5: 5일 이평선
            ma20: 20일 이평선
            ma20_series: 20일 이평선 시계열
            daily_change: 일간 변동률
            
        Returns:
            MarketScoreComponents 객체
        """
        # 각 구성요소 계산
        index_score, index_details = self.calculate_index_position_score(
            current_index, ma5, ma20
        )
        
        slope_score, slope_details = self.calculate_ma_slope_score(ma20_series)
        
        vol_score, vol_details = self.calculate_volatility_score(daily_change)
        
        # 총 점수
        total_score = index_score + slope_score + vol_score
        
        # 로깅
        logger.info(f"Market Score 계산 완료:")
        logger.info(f"  지수위치: {index_score:.1f}/40")
        logger.info(f"  기울기: {slope_score:.1f}/30") 
        logger.info(f"  변동성: {vol_score:.1f}/30")
        logger.info(f"  총점: {total_score:.1f}/100")
        
        return MarketScoreComponents(
            index_position_score=index_score,
            ma_slope_score=slope_score,
            volatility_score=vol_score,
            total_score=total_score,
            index_vs_ma5=index_details['ma5_ratio'],
            index_vs_ma20=index_details['ma20_ratio'],
            ma20_slope=slope_details['normalized_slope'],
            daily_change=daily_change,
            volatility_rank=vol_score / 30
        )
    
    def get_score_explanation(self, components: MarketScoreComponents) -> Dict[str, str]:
        """
        점수에 대한 상세 설명 제공 (투명성 확보)
        
        Args:
            components: MarketScoreComponents 객체
            
        Returns:
            설명 딕셔너리
        """
        explanations = {}
        
        # 지수 위치 설명
        if components.index_position_score >= 35:
            explanations['index_position'] = "지수가 이평선들 위에 강하게 위치"
        elif components.index_position_score >= 25:
            explanations['index_position'] = "지수가 이평선 근처에 위치"
        else:
            explanations['index_position'] = "지수가 이평선 아래 위치, 약세"
        
        # 기울기 설명
        if components.ma_slope_score >= 25:
            explanations['ma_slope'] = "이평선이 강하게 상승 중"
        elif components.ma_slope_score >= 15:
            explanations['ma_slope'] = "이평선이 완만하게 상승 또는 횡보"
        else:
            explanations['ma_slope'] = "이평선이 하락 추세"
        
        # 변동성 설명
        if components.volatility_score >= 25:
            explanations['volatility'] = "적절한 변동성, 거래 기회 존재"
        elif components.volatility_score >= 15:
            explanations['volatility'] = "보통 수준의 변동성"
        else:
            explanations['volatility'] = "과도한 하락, 위험한 시장 상황"
        
        # 총합 평가
        if components.total_score >= 80:
            explanations['overall'] = "🟢 매우 좋은 시장 환경 - 적극적 매매 추천"
        elif components.total_score >= 70:
            explanations['overall'] = "🟡 좋은 시장 환경 - 매매 가능"
        elif components.total_score >= 60:
            explanations['overall'] = "🟠 보통 시장 환경 - 신중한 매매"
        else:
            explanations['overall'] = "🔴 좋지 않은 시장 환경 - 매매 자제"
            
        return explanations


def main():
    """테스트용 메인 함수"""
    # 샘플 데이터로 테스트
    calculator = MarketScoreCalculator()
    
    # 샘플 값들
    current_index = 2500
    ma5 = 2450  # 2% 위
    ma20 = 2400  # 4% 위
    daily_change = 0.01  # 1% 상승
    
    # MA20 시리즈 생성 (상승 추세)
    ma20_series = pd.Series([2350 + i * 2.5 for i in range(20)])
    
    # Market Score 계산
    components = calculator.calculate_market_score(
        current_index, ma5, ma20, ma20_series, daily_change
    )
    
    # 결과 출력
    print("=== Market Score 분석 결과 ===")
    print(f"지수 위치 점수: {components.index_position_score:.1f}/40")
    print(f"기울기 점수: {components.ma_slope_score:.1f}/30")
    print(f"변동성 점수: {components.volatility_score:.1f}/30")
    print(f"총 점수: {components.total_score:.1f}/100")
    
    # 설명
    explanations = calculator.get_score_explanation(components)
    print("\n=== 상세 분석 ===")
    for key, explanation in explanations.items():
        print(f"{key}: {explanation}")


if __name__ == "__main__":
    main()