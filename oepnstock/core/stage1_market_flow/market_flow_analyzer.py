"""
Market Flow Analyzer - Stage 1 of 4-Stage Trading Strategy
시장의 흐름 파악: 종합지수, 주도업종, 시장분위기 체크
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
class MarketCondition:
    """Market condition assessment result"""
    score: float
    regime: str  # 'strong_uptrend', 'weak_uptrend', 'sideways', 'downtrend'
    index_position: Dict[str, float]
    ma_slope: Dict[str, float]
    daily_change: Dict[str, float]
    leading_sectors: List[Dict[str, Any]]
    market_sentiment: Dict[str, Any]
    tradable: bool
    warnings: List[str]


@dataclass
class SectorAnalysis:
    """Sector rotation and overheating analysis"""
    name: str
    five_day_return: float
    volume_ratio: float
    status: str  # 'normal', 'hot', 'overheated'
    rank: int


class MarketFlowAnalyzer:
    """
    1단계: 시장의 흐름 파악하기 (숲 보기)
    
    주요 기능:
    - 종합지수 위치 및 추세 분석
    - 주도 업종(테마) 파악 및 과열도 체크
    - 외부/내부 지표를 통한 시장 분위기 측정
    - 거래 가능 조건 점수화 (70점 이상 필요)
    """
    
    def __init__(self):
        self.config = config.trading
        self.technical_config = config.technical
        
        # Market condition scoring weights
        self.scoring_weights = {
            'index_position': 0.4,  # 40점
            'ma_slope': 0.3,        # 30점  
            'volatility': 0.3       # 30점
        }
        
        # Market regimes and their parameters
        self.market_regimes = {
            'strong_uptrend': {
                'min_score': 80,
                'position_multiplier': 1.2,
                'stop_loss_ratio': 0.03
            },
            'weak_uptrend': {
                'min_score': 70,
                'position_multiplier': 1.0,
                'stop_loss_ratio': 0.025
            },
            'sideways': {
                'min_score': 60,
                'position_multiplier': 0.8,
                'stop_loss_ratio': 0.02
            },
            'downtrend': {
                'min_score': 90,  # Very strict
                'position_multiplier': 0.5,
                'stop_loss_ratio': 0.015
            }
        }
    
    def analyze_market_flow(self, 
                          kospi_data: pd.DataFrame,
                          kosdaq_data: pd.DataFrame, 
                          sector_data: Dict[str, pd.DataFrame],
                          external_data: Optional[Dict] = None) -> MarketCondition:
        """
        종합적인 시장 흐름 분석
        
        Args:
            kospi_data: KOSPI 지수 데이터 (OHLCV)
            kosdaq_data: KOSDAQ 지수 데이터 (OHLCV) 
            sector_data: 업종별 데이터
            external_data: 외부 지표 데이터 (VIX, 환율 등)
            
        Returns:
            MarketCondition: 시장 상황 분석 결과
        """
        logger.info("Starting market flow analysis")
        
        try:
            # 1. 종합지수 분석
            kospi_analysis = self._analyze_index(kospi_data, "KOSPI")
            kosdaq_analysis = self._analyze_index(kosdaq_data, "KOSDAQ")
            
            # 2. 주도 업종 분석
            sector_analysis = self._analyze_sectors(sector_data)
            
            # 3. 시장 분위기 체크
            sentiment = self._analyze_market_sentiment(
                kospi_data, kosdaq_data, external_data or {}
            )
            
            # 4. 종합 점수 계산
            market_score = self._calculate_market_score(
                kospi_analysis, kosdaq_analysis, sector_analysis, sentiment
            )
            
            # 5. 시장 상황 분류
            regime = self._classify_market_regime(market_score, kospi_analysis, kosdaq_analysis)
            
            # 6. 거래 가능 여부 판단
            tradable = market_score >= self.config.market_score_threshold
            
            # 7. 경고 사항 생성
            warnings = self._generate_warnings(
                market_score, sector_analysis, sentiment
            )
            
            condition = MarketCondition(
                score=market_score,
                regime=regime,
                index_position={
                    'kospi': kospi_analysis,
                    'kosdaq': kosdaq_analysis
                },
                ma_slope={
                    'kospi': kospi_analysis.get('ma_slope', 0),
                    'kosdaq': kosdaq_analysis.get('ma_slope', 0)
                },
                daily_change={
                    'kospi': kospi_analysis.get('daily_change', 0),
                    'kosdaq': kosdaq_analysis.get('daily_change', 0)  
                },
                leading_sectors=sector_analysis[:3],  # Top 3 sectors
                market_sentiment=sentiment,
                tradable=tradable,
                warnings=warnings
            )
            
            logger.info(f"Market analysis complete - Score: {market_score:.1f}, Regime: {regime}, Tradable: {tradable}")
            return condition
            
        except Exception as e:
            logger.error(f"Error in market flow analysis: {e}")
            raise
    
    def _analyze_index(self, data: pd.DataFrame, index_name: str) -> Dict[str, float]:
        """개별 지수 분석"""
        if len(data) < self.technical_config.ma_long:
            raise ValueError(f"Insufficient data for {index_name} analysis")
        
        current_price = data['close'].iloc[-1]
        
        # Moving averages
        ma5 = data['close'].rolling(5).mean().iloc[-1]
        ma20 = data['close'].rolling(20).mean().iloc[-1]
        
        # MA slope calculation (최근 5일 기울기)
        ma20_series = data['close'].rolling(20).mean()
        ma_slope = self._calculate_ma_slope(ma20_series, lookback=5)
        
        # Daily change
        daily_change = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2]
        
        # Position scores
        above_ma5 = 1.0 if current_price > ma5 else 0.0
        above_ma20 = 1.0 if current_price > ma20 else 0.0
        
        return {
            'current_price': current_price,
            'ma5': ma5,
            'ma20': ma20,
            'ma_slope': ma_slope,
            'daily_change': daily_change,
            'above_ma5': above_ma5,
            'above_ma20': above_ma20,
            'position_score': (above_ma5 + above_ma20) * 20  # 최대 40점
        }
    
    def _calculate_ma_slope(self, ma_series: pd.Series, lookback: int = 5) -> float:
        """
        MA 기울기 계산 (-100 ~ +100)
        +3% 이상: 강한 상승
        -1% ~ +3%: 완만한 상승/횡보  
        -3% ~ -1%: 약한 하락
        -3% 이하: 강한 하락
        """
        if len(ma_series) < lookback:
            return 0.0
            
        start_value = ma_series.iloc[-lookback-1]
        end_value = ma_series.iloc[-1]
        
        if start_value == 0:
            return 0.0
            
        slope = ((end_value - start_value) / start_value) * 100
        return slope / lookback  # 일평균 변화율
    
    def _analyze_sectors(self, sector_data: Dict[str, pd.DataFrame]) -> List[SectorAnalysis]:
        """
        업종별 등락률 및 과열도 분석
        상위 3개 섹터 추출 및 과열 경고 시스템
        """
        sector_analyses = []
        
        for sector_name, data in sector_data.items():
            if len(data) < 5:
                continue
                
            # 5일간 누적 수익률
            five_day_return = (data['close'].iloc[-1] / data['close'].iloc[-6]) - 1
            
            # 거래량 비율 (평균 대비)
            current_volume = data['volume'].iloc[-1] 
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 과열도 체크
            status = self._check_sector_overheating(five_day_return, volume_ratio)
            
            sector_analyses.append(SectorAnalysis(
                name=sector_name,
                five_day_return=five_day_return,
                volume_ratio=volume_ratio,
                status=status,
                rank=0  # Will be set after sorting
            ))
        
        # 수익률 기준 정렬 및 순위 부여
        sector_analyses.sort(key=lambda x: x.five_day_return, reverse=True)
        for i, analysis in enumerate(sector_analyses):
            analysis.rank = i + 1
            
        return sector_analyses
    
    def _check_sector_overheating(self, five_day_return: float, volume_ratio: float) -> str:
        """
        섹터 과열도 체크
        Returns: 'normal', 'hot', 'overheated'
        """
        if five_day_return > self.config.sector_overheating_return and \
           volume_ratio > self.config.sector_overheating_volume_ratio:
            return 'overheated'  # 진입 보류
        elif five_day_return > 0.15:  # 15%
            return 'hot'  # 신중한 진입
        else:
            return 'normal'  # 정상 진입
    
    def _analyze_market_sentiment(self, 
                                kospi_data: pd.DataFrame,
                                kosdaq_data: pd.DataFrame, 
                                external_data: Dict) -> Dict[str, Any]:
        """시장 분위기 종합 분석"""
        sentiment = {}
        
        # 외부 변수 체크
        if 'us_markets' in external_data:
            us_data = external_data['us_markets']
            sentiment['us_markets'] = {
                'sp500_change': us_data.get('sp500_change', 0),
                'nasdaq_change': us_data.get('nasdaq_change', 0),
                'warning': abs(us_data.get('sp500_change', 0)) > 0.02
            }
        
        if 'usd_krw' in external_data:
            sentiment['usd_krw'] = {
                'rate': external_data['usd_krw'],
                'warning': False  # 급등 로직 추가 필요
            }
        
        if 'vix' in external_data:
            vix_level = external_data['vix']
            sentiment['vix'] = {
                'level': vix_level,
                'warning': vix_level > 30
            }
        
        # 국내 지표 (프로그램 매매, 외국인 동향 등)
        # 실제 구현 시 관련 API 데이터 연동 필요
        sentiment['program_trading'] = {'net_buy': 0}  # Placeholder
        sentiment['foreign_flow'] = {'net_buy': 0}     # Placeholder
        
        return sentiment
    
    def _calculate_market_score(self,
                              kospi_analysis: Dict,
                              kosdaq_analysis: Dict,
                              sector_analysis: List[SectorAnalysis],
                              sentiment: Dict) -> float:
        """거래 가능 조건 점수화 (최대 100점)"""
        
        score = 0.0
        
        # 1. 지수 위치 점수 (40점)
        kospi_position_score = kospi_analysis.get('position_score', 0)
        kosdaq_position_score = kosdaq_analysis.get('position_score', 0)
        index_score = (kospi_position_score + kosdaq_position_score) / 2
        score += index_score * self.scoring_weights['index_position']
        
        # 2. MA 기울기 점수 (30점)
        kospi_slope = kospi_analysis.get('ma_slope', 0)
        kosdaq_slope = kosdaq_analysis.get('ma_slope', 0)
        avg_slope = (kospi_slope + kosdaq_slope) / 2
        
        slope_score = 0
        if avg_slope > 0.5:
            slope_score = 30
        elif avg_slope > 0:
            slope_score = 20
        elif avg_slope > -0.3:
            slope_score = 10
        
        score += slope_score * self.scoring_weights['ma_slope']
        
        # 3. 변동성 점수 (30점)
        kospi_change = kospi_analysis.get('daily_change', 0)
        kosdaq_change = kosdaq_analysis.get('daily_change', 0)
        avg_change = (kospi_change + kosdaq_change) / 2
        
        volatility_score = 30 if avg_change > -0.02 else 0
        score += volatility_score * self.scoring_weights['volatility']
        
        # 4. 추가 조정 (섹터 과열, 외부 리스크 등)
        overheated_sectors = sum(1 for s in sector_analysis if s.status == 'overheated')
        if overheated_sectors > 2:
            score *= 0.8  # 20% 감점
        
        # VIX 경고 시 감점
        if sentiment.get('vix', {}).get('warning', False):
            score *= 0.9  # 10% 감점
        
        return min(score, 100.0)  # 최대 100점
    
    def _classify_market_regime(self, 
                              score: float,
                              kospi_analysis: Dict, 
                              kosdaq_analysis: Dict) -> str:
        """시장 상황 분류"""
        
        avg_slope = (kospi_analysis.get('ma_slope', 0) + kosdaq_analysis.get('ma_slope', 0)) / 2
        avg_change = (kospi_analysis.get('daily_change', 0) + kosdaq_analysis.get('daily_change', 0)) / 2
        
        if score >= 80 and avg_slope > 0.5:
            return 'strong_uptrend'
        elif score >= 70 and avg_slope > 0:
            return 'weak_uptrend' 
        elif avg_slope > -0.3 and abs(avg_change) < 0.01:
            return 'sideways'
        else:
            return 'downtrend'
    
    def _generate_warnings(self, 
                         market_score: float,
                         sector_analysis: List[SectorAnalysis],
                         sentiment: Dict) -> List[str]:
        """경고 메시지 생성"""
        warnings = []
        
        if market_score < self.config.market_score_threshold:
            warnings.append(f"Market score {market_score:.1f} below threshold {self.config.market_score_threshold}")
        
        overheated_sectors = [s.name for s in sector_analysis if s.status == 'overheated']
        if overheated_sectors:
            warnings.append(f"Overheated sectors: {', '.join(overheated_sectors)}")
        
        if sentiment.get('vix', {}).get('warning', False):
            warnings.append("High VIX level detected - increased market volatility")
        
        if sentiment.get('us_markets', {}).get('warning', False):
            warnings.append("Significant US market movement detected")
            
        return warnings
    
    def get_regime_parameters(self, regime: str) -> Dict[str, float]:
        """시장 상황별 매매 파라미터 조정값 반환"""
        return self.market_regimes.get(regime, self.market_regimes['sideways'])