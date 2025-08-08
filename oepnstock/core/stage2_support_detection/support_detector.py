"""
Support Level Detection - Stage 2 of 4-Stage Trading Strategy
매수 후보 지점 찾기: 수평지지선, 이동평균선, 의미있는 가격대
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
class SupportLevel:
    """Individual support level information"""
    price: float
    strength: float  # 0.0 ~ 1.0
    touch_count: int
    support_type: str  # 'horizontal', 'ma', 'round_figure', 'volume_profile'
    last_touch_date: datetime
    volume_at_level: float
    in_cluster: bool = False
    cluster_id: Optional[int] = None


@dataclass
class SupportCluster:
    """Clustered support zone"""
    price_center: float
    price_range: Tuple[float, float]  # (min, max)
    strength: float
    support_count: int
    support_types: List[str]
    volume_weight: float


@dataclass
class SupportAnalysis:
    """Complete support analysis result"""
    symbol: str
    current_price: float
    individual_supports: List[SupportLevel]
    support_clusters: List[SupportCluster] 
    strongest_support: Optional[SupportLevel]
    nearest_support: Optional[SupportLevel]
    recommendation: Dict[str, Any]


class SupportDetector:
    """
    2단계: 매수 후보 지점 찾기 (나무 찾기)
    
    주요 기능:
    - 수평 지지선 감지 (직전 저점, 전고점, 박스권)
    - 이동평균선 지지 (5일, 10일, 20일선)
    - 의미 있는 가격대 (라운드 피겨, 거래량 밀집 구간)
    - 지지선 클러스터링 분석 (겹치는 지지선들)
    """
    
    def __init__(self):
        self.config = config.technical
        self.trading_config = config.trading
        
        # Support detection parameters
        self.lookback_period = 60  # 60일 분석
        self.min_touch_count = 2
        self.cluster_threshold = 0.01  # 1% range for clustering
        
        # Support type weights
        self.support_weights = {
            'horizontal': 1.0,
            'ma20': 2.0,        # 세력선 - 가장 중요
            'ma10': 1.5,
            'ma5': 1.0,
            'round_figure': 1.2,
            'volume_profile': 1.3
        }
    
    def detect_support_levels(self, data: pd.DataFrame, symbol: str) -> SupportAnalysis:
        """
        종합적인 지지선 분석
        
        Args:
            data: OHLCV 데이터
            symbol: 종목 코드
            
        Returns:
            SupportAnalysis: 지지선 분석 결과
        """
        logger.info(f"Starting support detection for {symbol}")
        
        if len(data) < self.lookback_period:
            raise ValueError(f"Insufficient data for support analysis: {len(data)} < {self.lookback_period}")
        
        current_price = data['close'].iloc[-1]
        
        try:
            # 1. 각 유형별 지지선 감지
            horizontal_supports = self._detect_horizontal_supports(data)
            ma_supports = self._detect_ma_supports(data)
            round_figure_supports = self._detect_round_figures(current_price)
            volume_profile_supports = self._detect_volume_profile_supports(data)
            
            # 2. 모든 지지선 통합
            all_supports = (horizontal_supports + ma_supports + 
                          round_figure_supports + volume_profile_supports)
            
            # 3. 지지선 강도 계산
            for support in all_supports:
                support.strength = self._calculate_support_strength(support, data)
            
            # 4. 지지선 클러스터링
            clusters = self._analyze_support_clustering(all_supports, current_price)
            
            # 5. 클러스터 정보를 개별 지지선에 반영
            self._update_support_clusters(all_supports, clusters)
            
            # 6. 최강/최근 지지선 식별
            strongest_support = max(all_supports, key=lambda x: x.strength) if all_supports else None
            nearest_support = self._find_nearest_support(all_supports, current_price)
            
            # 7. 매수 추천 생성
            recommendation = self._generate_recommendation(
                current_price, all_supports, clusters, strongest_support, nearest_support
            )
            
            analysis = SupportAnalysis(
                symbol=symbol,
                current_price=current_price,
                individual_supports=sorted(all_supports, key=lambda x: x.strength, reverse=True),
                support_clusters=sorted(clusters, key=lambda x: x.strength, reverse=True),
                strongest_support=strongest_support,
                nearest_support=nearest_support,
                recommendation=recommendation
            )
            
            logger.info(f"Support analysis complete for {symbol} - Found {len(all_supports)} supports, {len(clusters)} clusters")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in support detection for {symbol}: {e}")
            raise
    
    def _detect_horizontal_supports(self, data: pd.DataFrame) -> List[SupportLevel]:
        """수평 지지선 감지"""
        supports = []
        
        # 1. 직전 저점들 (최근 20일 내 최저가)
        recent_lows = self._find_recent_lows(data, window=20)
        for low_info in recent_lows:
            supports.append(SupportLevel(
                price=low_info['price'],
                strength=0.0,  # Will be calculated later
                touch_count=low_info['touch_count'],
                support_type='horizontal',
                last_touch_date=low_info['date'],
                volume_at_level=low_info['volume']
            ))
        
        # 2. 전고점 (이전 상승파동의 고점)
        previous_highs = self._find_previous_highs(data)
        for high_info in previous_highs:
            # 현재가 대비 -5% ~ -15% 구간의 고점들
            price_diff_ratio = (high_info['price'] - data['close'].iloc[-1]) / data['close'].iloc[-1]
            if -0.15 <= price_diff_ratio <= -0.05:
                supports.append(SupportLevel(
                    price=high_info['price'],
                    strength=0.0,
                    touch_count=high_info['touch_count'],
                    support_type='horizontal',
                    last_touch_date=high_info['date'],
                    volume_at_level=high_info['volume']
                ))
        
        # 3. 박스권 하단 (10일 이상 횡보 구간)
        box_supports = self._find_box_supports(data)
        supports.extend(box_supports)
        
        return supports
    
    def _detect_ma_supports(self, data: pd.DataFrame) -> List[SupportLevel]:
        """이동평균선 지지 감지"""
        supports = []
        current_date = data.index[-1]
        
        # MA 계산
        ma5 = data['close'].rolling(5).mean().iloc[-1]
        ma10 = data['close'].rolling(10).mean().iloc[-1] 
        ma20 = data['close'].rolling(20).mean().iloc[-1]
        
        # 각 MA에서의 지지 정보 생성
        mas = [
            ('ma5', ma5, 5),
            ('ma10', ma10, 10), 
            ('ma20', ma20, 20)
        ]
        
        for ma_name, ma_value, period in mas:
            # 해당 MA 터치 횟수 계산
            ma_series = data['close'].rolling(period).mean()
            touch_count = self._count_ma_touches(data['close'], ma_series)
            
            supports.append(SupportLevel(
                price=ma_value,
                strength=0.0,
                touch_count=touch_count,
                support_type=ma_name,
                last_touch_date=current_date,
                volume_at_level=data['volume'].rolling(5).mean().iloc[-1]
            ))
        
        return supports
    
    def _detect_round_figures(self, current_price: float) -> List[SupportLevel]:
        """라운드 피겨 감지"""
        supports = []
        
        # 가격대별 라운드 피겨 단위 결정
        if current_price < 10000:
            round_unit = 100      # 100원 단위
        elif current_price < 100000:
            round_unit = 1000     # 1,000원 단위
        else:
            round_unit = 5000     # 5,000원 단위
        
        # 현재가 근처의 라운드 피겨들 찾기
        base_round = int(current_price / round_unit) * round_unit
        
        for i in range(-2, 1):  # 아래 2개, 현재, 위 0개
            round_price = base_round + (i * round_unit)
            if round_price > 0 and round_price < current_price:
                supports.append(SupportLevel(
                    price=float(round_price),
                    strength=0.0,
                    touch_count=1,  # 라운드 피겨는 기본 1회
                    support_type='round_figure',
                    last_touch_date=datetime.now(),
                    volume_at_level=0.0  # 추후 계산 필요
                ))
        
        return supports
    
    def _detect_volume_profile_supports(self, data: pd.DataFrame, lookback_days: int = 5) -> List[SupportLevel]:
        """거래량 밀집 구간 감지"""
        supports = []
        
        if len(data) < lookback_days:
            return supports
        
        # VWAP 계산 (최근 N일)
        recent_data = data.iloc[-lookback_days:]
        vwap_levels = []
        
        for i in range(len(recent_data)):
            day_data = recent_data.iloc[i:i+1]
            if len(day_data) > 0 and day_data['volume'].sum() > 0:
                daily_vwap = (day_data['close'] * day_data['volume']).sum() / day_data['volume'].sum()
                vwap_levels.append(daily_vwap)
        
        if vwap_levels:
            main_level = np.mean(vwap_levels)
            
            supports.append(SupportLevel(
                price=main_level,
                strength=0.0,
                touch_count=len(vwap_levels),
                support_type='volume_profile',
                last_touch_date=data.index[-1],
                volume_at_level=recent_data['volume'].mean()
            ))
        
        return supports
    
    def _calculate_support_strength(self, support: SupportLevel, data: pd.DataFrame) -> float:
        """
        지지선 강도 계산
        강도 = 기본점수 + 클러스터보너스 + MA일치보너스
        """
        strength = 0.0
        
        # 1. 기본 점수 (터치 횟수)
        touch_score = min(support.touch_count * 0.1, 0.3)
        strength += touch_score
        
        # 2. 거래량 비중 (volume_profile 타입일 경우)
        if support.support_type == 'volume_profile' and support.volume_at_level > 0:
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = support.volume_at_level / avg_volume if avg_volume > 0 else 1.0
            strength += volume_ratio * 0.3
        
        # 3. 지지선 타입별 가중치 적용
        type_weight = self.support_weights.get(support.support_type, 1.0)
        strength *= type_weight
        
        # 4. 20일선 근처 보너스
        if len(data) >= 20:
            ma20 = data['close'].rolling(20).mean().iloc[-1]
            if abs(support.price - ma20) / ma20 < 0.01:  # 1% 이내
                strength += 0.4
        
        return min(strength, 1.0)  # 최대 1.0
    
    def _analyze_support_clustering(self, supports: List[SupportLevel], current_price: float) -> List[SupportCluster]:
        """여러 지지선이 겹치는 구간 찾기"""
        clusters = []
        processed_supports = set()
        
        for i, support in enumerate(supports):
            if i in processed_supports:
                continue
            
            cluster_supports = [support]
            cluster_indices = {i}
            
            # 해당 지지선 ±1% 범위 내의 다른 지지선들 찾기
            for j, other_support in enumerate(supports):
                if j == i or j in processed_supports:
                    continue
                
                price_diff_ratio = abs(support.price - other_support.price) / support.price
                if price_diff_ratio < self.cluster_threshold:
                    cluster_supports.append(other_support)
                    cluster_indices.add(j)
            
            # 3개 이상 겹치는 경우만 클러스터로 인정
            if len(cluster_supports) >= 3:
                cluster_price = np.mean([s.price for s in cluster_supports])
                cluster_strength = sum(s.strength for s in cluster_supports) * 1.5  # 클러스터 보너스
                
                min_price = min(s.price for s in cluster_supports)
                max_price = max(s.price for s in cluster_supports)
                
                cluster = SupportCluster(
                    price_center=cluster_price,
                    price_range=(min_price, max_price),
                    strength=cluster_strength,
                    support_count=len(cluster_supports),
                    support_types=[s.support_type for s in cluster_supports],
                    volume_weight=sum(s.volume_at_level for s in cluster_supports)
                )
                
                clusters.append(cluster)
                processed_supports.update(cluster_indices)
        
        return clusters
    
    def _update_support_clusters(self, supports: List[SupportLevel], clusters: List[SupportCluster]):
        """개별 지지선에 클러스터 정보 업데이트"""
        for cluster_id, cluster in enumerate(clusters):
            for support in supports:
                if cluster.price_range[0] <= support.price <= cluster.price_range[1]:
                    support.in_cluster = True
                    support.cluster_id = cluster_id
                    # 클러스터 보너스 적용
                    support.strength += 0.2
    
    def _find_nearest_support(self, supports: List[SupportLevel], current_price: float) -> Optional[SupportLevel]:
        """현재가에서 가장 가까운 아래쪽 지지선 찾기"""
        below_supports = [s for s in supports if s.price < current_price]
        if not below_supports:
            return None
        
        return max(below_supports, key=lambda x: x.price)
    
    def _generate_recommendation(self, 
                               current_price: float,
                               supports: List[SupportLevel], 
                               clusters: List[SupportCluster],
                               strongest_support: Optional[SupportLevel],
                               nearest_support: Optional[SupportLevel]) -> Dict[str, Any]:
        """매수 추천 생성"""
        
        recommendation = {
            'action': 'wait',
            'target_prices': [],
            'reasoning': [],
            'risk_level': 'medium'
        }
        
        # 1. 클러스터 근처 여부 체크
        near_cluster = False
        best_cluster = None
        
        for cluster in clusters[:3]:  # 상위 3개 클러스터만 체크
            distance_ratio = abs(current_price - cluster.price_center) / current_price
            if distance_ratio <= 0.02:  # 2% 이내
                near_cluster = True
                best_cluster = cluster
                break
        
        if near_cluster and best_cluster:
            recommendation['action'] = 'strong_buy_candidate'
            recommendation['target_prices'].append({
                'price': best_cluster.price_center,
                'type': 'cluster_support',
                'strength': best_cluster.strength,
                'reason': f"{best_cluster.support_count}개 지지선 클러스터"
            })
            recommendation['reasoning'].append(f"강력한 지지선 클러스터({best_cluster.support_count}개) 근처")
            recommendation['risk_level'] = 'low'
        
        # 2. 강력한 단일 지지선 근처
        elif strongest_support and strongest_support.strength > 0.7:
            distance_ratio = abs(current_price - strongest_support.price) / current_price
            if distance_ratio <= 0.03:  # 3% 이내
                recommendation['action'] = 'buy_candidate' 
                recommendation['target_prices'].append({
                    'price': strongest_support.price,
                    'type': strongest_support.support_type,
                    'strength': strongest_support.strength,
                    'reason': f"강력한 {strongest_support.support_type} 지지"
                })
                recommendation['reasoning'].append(f"강력한 지지선(강도: {strongest_support.strength:.2f}) 근처")
        
        # 3. 일반적인 지지선 근처
        elif nearest_support:
            distance_ratio = abs(current_price - nearest_support.price) / current_price
            if distance_ratio <= 0.05:  # 5% 이내
                recommendation['action'] = 'watch'
                recommendation['target_prices'].append({
                    'price': nearest_support.price, 
                    'type': nearest_support.support_type,
                    'strength': nearest_support.strength,
                    'reason': f"{nearest_support.support_type} 지지선"
                })
                recommendation['reasoning'].append("지지선 근처 - 추가 신호 필요")
        
        return recommendation
    
    # Helper methods
    def _find_recent_lows(self, data: pd.DataFrame, window: int = 20) -> List[Dict]:
        """최근 저점들 찾기"""
        lows = []
        recent_data = data.iloc[-window:]
        
        for i in range(1, len(recent_data) - 1):
            current = recent_data.iloc[i]
            prev_low = recent_data.iloc[i-1]['low']  
            next_low = recent_data.iloc[i+1]['low']
            
            # 지역 최저점인지 체크
            if current['low'] < prev_low and current['low'] < next_low:
                lows.append({
                    'price': current['low'],
                    'date': current.name,
                    'volume': current['volume'],
                    'touch_count': 1  # 기본값, 실제로는 더 정교한 계산 필요
                })
        
        return lows
    
    def _find_previous_highs(self, data: pd.DataFrame) -> List[Dict]:
        """이전 고점들 찾기"""
        highs = []
        
        # 간단한 구현 - 실제로는 더 정교한 파동 분석 필요
        for i in range(20, len(data) - 1):
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i] > data['high'].iloc[i+1]):
                highs.append({
                    'price': data['high'].iloc[i],
                    'date': data.index[i],
                    'volume': data['volume'].iloc[i],
                    'touch_count': 1
                })
        
        return highs[-10:]  # 최근 10개만
    
    def _find_box_supports(self, data: pd.DataFrame) -> List[SupportLevel]:
        """박스권 하단 찾기"""
        supports = []
        
        # 간단한 박스권 감지 로직
        # 실제 구현 시 더 정교한 횡보 구간 감지 알고리즘 필요
        window = 10
        for i in range(window, len(data) - window):
            section = data.iloc[i-window:i+window]
            price_range = section['high'].max() - section['low'].min()
            avg_price = section['close'].mean()
            
            # 가격 변동폭이 평균가의 5% 이내면 횡보로 판단
            if price_range / avg_price < 0.05:
                supports.append(SupportLevel(
                    price=section['low'].min(),
                    strength=0.0,
                    touch_count=2,
                    support_type='horizontal',
                    last_touch_date=section.index[-1],
                    volume_at_level=section['volume'].mean()
                ))
        
        return supports[-5:]  # 최근 5개만
    
    def _count_ma_touches(self, price_series: pd.Series, ma_series: pd.Series, tolerance: float = 0.01) -> int:
        """이동평균선 터치 횟수 계산"""
        touches = 0
        
        for i in range(1, len(price_series)):
            if len(ma_series) > i:
                # 가격이 MA 근처(±1%)에서 반등했는지 체크
                price_to_ma_ratio = abs(price_series.iloc[i] - ma_series.iloc[i]) / ma_series.iloc[i]
                if price_to_ma_ratio < tolerance:
                    touches += 1
        
        return touches