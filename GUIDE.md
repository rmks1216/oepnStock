# 단기 매매 4단계 체크리스트 전략 구현 가이드 v2.0

## 전략 개요
이 문서는 단기 매매를 위한 4단계 체크리스트를 프로그램으로 구현하기 위한 개선된 가이드입니다. 실전 피드백을 반영하여 더욱 정교하고 현실적인 매매 시스템을 구축합니다.

---

## 1단계: 시장의 흐름 파악하기 (숲 보기 🌲)

### 개념 설명
내 종목이 아무리 좋아도 시장 전체가 무너지면 함께 하락할 확률이 높습니다.

### 체크 항목

#### ① 종합 주가 지수 확인 (개선됨)
- **목적**: 코스피, 코스닥 지수의 추세와 강도 파악
- **구현 기준**:
  - 20일 이동평균선 vs 현재가: 위치 확인
  - **🆕 20일선 기울기**: 추세의 강도 측정
    ```python
    def calculate_ma_slope(prices, period=20, lookback=5):
        """
        MA 기울기 계산 (-100 ~ +100)
        +3% 이상: 강한 상승
        -1% ~ +3%: 완만한 상승/횡보
        -3% ~ -1%: 약한 하락
        -3% 이하: 강한 하락
        """
        ma = prices.rolling(period).mean()
        slope = ((ma.iloc[-1] - ma.iloc[-lookback]) / ma.iloc[-lookback]) * 100
        return slope / lookback  # 일평균 변화율
    ```
  - 5일 이동평균선 vs 20일 이동평균선: 골든크로스/데드크로스
  - 전일 대비 변화율: -2% 이상 하락 시 위험 신호

- **거래 가능 조건 점수화**:
  ```python
  def calculate_market_score():
      score = 0
      # 지수 위치 (40점)
      if current_price > ma20: score += 20
      if current_price > ma5: score += 20
      
      # MA 기울기 (30점)
      slope = calculate_ma_slope(prices)
      if slope > 0.5: score += 30
      elif slope > 0: score += 20
      elif slope > -0.3: score += 10
      
      # 변동성 (30점)
      if daily_change > -2: score += 30
      
      return score  # 70점 이상이면 거래 가능
  ```

#### ② 주도 업종(테마) 파악 (개선됨)
- **목적**: 시장 자금이 몰리는 섹터 확인 및 과열 여부 판단
- **구현 방법**:
  - 업종별 등락률 상위 3개 섹터 추출
  - 최근 5일간 누적 수익률 계산
  - **🆕 과열 경고 시스템**:
    ```python
    def check_sector_overheating(sector_data):
        """
        섹터 과열도 체크
        Returns: 'normal', 'hot', 'overheated'
        """
        five_day_return = sector_data['5d_return']
        volume_ratio = sector_data['volume'] / sector_data['avg_volume']
        
        if five_day_return > 20 and volume_ratio > 3:
            return 'overheated'  # 진입 보류
        elif five_day_return > 15:
            return 'hot'  # 신중한 진입
        else:
            return 'normal'  # 정상 진입
    ```

#### ③ 시장 분위기 체크
- **외부 변수**:
  - 미국 증시: S&P500, 나스닥 전일 종가 (2% 이상 변동 시 주의)
  - 달러-원 환율: 급등 시 외국인 매도 압력
  - VIX 지수: 30 이상 시 고변동성 주의
- **국내 지표**:
  - 프로그램 매매 동향
  - 외국인/기관 순매수 금액
  - 신용잔고 증감

### 시장 상황별 전략 조정
```python
MARKET_CONDITIONS = {
    'strong_uptrend': {
        'min_score': 80,
        'position_size_multiplier': 1.2,
        'stop_loss_ratio': 0.03  # 3%
    },
    'weak_uptrend': {
        'min_score': 70,
        'position_size_multiplier': 1.0,
        'stop_loss_ratio': 0.025
    },
    'sideways': {
        'min_score': 60,
        'position_size_multiplier': 0.8,
        'stop_loss_ratio': 0.02
    },
    'downtrend': {
        'min_score': 90,  # 매우 엄격
        'position_size_multiplier': 0.5,
        'stop_loss_ratio': 0.015
    }
}
```

---

## 2단계: 매수 후보 지점 찾기 (나무 찾기 🌳)

### 개념 설명
개별 종목 차트에서 의미 있는 '지지'가 발생할 만한 자리를 찾습니다.

### 지지선 종류별 구현

#### ① 수평 지지선
- **직전 저점**: 최근 20일 내 최저가
- **전고점**: 이전 상승파동의 고점 (현재가 대비 -5% ~ -15% 구간)
- **박스권**: 최소 10일 이상 횡보한 구간의 하단
- **신뢰도 가중치**: 터치 횟수가 많을수록 높은 점수
  - 3회 이상 지지: 가중치 1.5
  - 5회 이상 지지: 가중치 2.0

#### ② 이동평균선
- **단기 매매용 주요 이평선**:
  - 5일선: 초단기 지지/저항
  - 10일선: 단기 추세선
  - 20일선(세력선): 가장 중요, 가중치 2.0
- **이격도 활용**: 
  - 20일선 대비 -5% 이상 하락 시 반등 가능성
  - -10% 이상은 과매도 (단, 하락 추세에서는 예외)

#### ③ 의미 있는 가격대
- **라운드 피겨**: 
  - 1만원 미만: 100원 단위
  - 1만원~10만원: 1,000원 단위  
  - 10만원 이상: 5,000원 단위
- **🆕 거래량 밀집 구간 (개선됨)**:
  ```python
  def calculate_volume_profile(data, lookback_days=5):
      """
      최근 N일간 가격대별 거래량 프로파일 계산
      Returns: Dict of price_level: volume
      """
      vwap_levels = []
      for i in range(lookback_days):
          daily_vwap = (data['close'] * data['volume']).sum() / data['volume'].sum()
          vwap_levels.append(daily_vwap)
      
      # 주요 거래 밀집 구간 = VWAP ± 2%
      return {
          'main_level': np.mean(vwap_levels),
          'upper_bound': np.mean(vwap_levels) * 1.02,
          'lower_bound': np.mean(vwap_levels) * 0.98
      }
  ```

### 🆕 지지선 클러스터링 분석 (신규)
```python
def analyze_support_clustering(support_levels, current_price, threshold=0.01):
    """
    여러 지지선이 겹치는 구간 찾기
    Returns: List of clustered zones with strength scores
    """
    clusters = []
    
    for price_level in support_levels:
        # 해당 가격 ±1% 범위 내의 다른 지지선들 찾기
        nearby_supports = [
            s for s in support_levels 
            if abs(s['price'] - price_level['price']) / price_level['price'] < threshold
        ]
        
        if len(nearby_supports) >= 3:
            cluster_strength = sum(s['strength'] for s in nearby_supports)
            clusters.append({
                'price': np.mean([s['price'] for s in nearby_supports]),
                'strength': cluster_strength * 1.5,  # 클러스터 보너스
                'support_count': len(nearby_supports),
                'types': [s['type'] for s in nearby_supports]
            })
    
    return sorted(clusters, key=lambda x: x['strength'], reverse=True)
```

### 지지선 강도 계산 (개선됨)
```python
def calculate_support_strength(support_info):
    """
    지지선 강도 = 기본점수 + 클러스터보너스 + MA일치보너스
    """
    strength = 0
    
    # 기본 점수 (터치 횟수)
    strength += min(support_info['touch_count'] * 0.1, 0.3)
    
    # 거래량 비중
    strength += support_info['volume_ratio'] * 0.3
    
    # 20일선 근처 보너스
    if abs(support_info['price'] - support_info['ma20']) / support_info['ma20'] < 0.01:
        strength += 0.4
    
    # 🆕 클러스터 보너스
    if support_info.get('in_cluster', False):
        strength += 0.2
    
    return min(strength, 1.0)  # 최대 1.0
```

---

## 3단계: '진짜 반등' 신호 확인하기 (방아쇠 당기기 ✅)

### 개념 설명
후보 지점에 도달했다고 바로 매수하는 것이 아니라, 실제 반등 신호를 확인합니다.

### 핵심 신호

#### ① 거래량 (가장 중요! 기본 가중치 50%)
- **기준**: 
  - 20일 평균 거래량 대비 150% 이상
  - 전일 거래량 대비 200% 이상
  - 장중 최저점 대비 종가까지 거래량의 60% 이상 발생
- **🆕 시간대별 거래량 분석**:
  ```python
  def analyze_intraday_volume(current_time, accumulated_volume, daily_avg_volume):
      """
      시간대별 거래량 강도 분석
      Returns: volume_strength (0~2)
      """
      time_ratio = get_market_time_ratio(current_time)  # 0~1
      expected_volume = daily_avg_volume * time_ratio
      
      if current_time <= "10:00":
          # 오전장 조기 거래량 폭증은 매우 강한 신호
          if accumulated_volume > daily_avg_volume * 0.5:
              return 2.0  # 매우 강함
          elif accumulated_volume > expected_volume * 1.5:
              return 1.5
      
      # 일반 시간대
      if accumulated_volume > expected_volume * 1.5:
          return 1.2
      elif accumulated_volume > expected_volume:
          return 1.0
      else:
          return 0.8
  ```

#### ② 캔들 패턴 (기본 가중치 30%)
- **망치형 (Hammer)**:
  - 아래꼬리가 몸통의 2배 이상
  - 위꼬리는 몸통의 10% 이하
  - 당일 저가가 지지선 하단 돌파 후 회복
- **도지형 (Doji)**:
  - 시가와 종가 차이 0.5% 이내
  - 위아래 꼬리 길이 비슷 (비율 0.7~1.3)
- **상승장악형 (Bullish Engulfing)**:
  - 전일 음봉을 완전히 감싸는 양봉
  - 거래량 동반 필수

#### ③ 보조 지표 (기본 가중치 20%)
- **RSI (14일 기준)**:
  - 30 이하: 과매도 신호
  - 다이버전스: 주가는 신저점, RSI는 저점 높임
- **스토캐스틱**:
  - %K가 20 이하에서 %D 상향 돌파
- **MACD**:
  - 0선 아래에서 시그널선 상향 돌파

### 🆕 동적 가중치 시스템
```python
def calculate_dynamic_weights(market_condition):
    """
    시장 상황에 따른 신호 가중치 동적 조정
    """
    if market_condition == 'strong_uptrend':
        return {
            'volume': 0.4,    # 거래량 중요도 낮춤
            'candle': 0.4,    # 캔들 패턴 중요도 높임
            'indicator': 0.2
        }
    elif market_condition == 'sideways':
        return {
            'volume': 0.5,
            'candle': 0.2,
            'indicator': 0.3  # 과매도 지표 중요
        }
    else:  # downtrend
        return {
            'volume': 0.6,    # 거래량 매우 중요
            'candle': 0.2,
            'indicator': 0.2
        }
```

### 신호 강도 종합 평가 (개선됨)
```python
def calculate_signal_strength(signals, market_condition):
    """
    종합 신호 강도 = 동적 가중치 적용
    80점 이상: 즉시 매수
    60-80점: 분할 매수
    60점 미만: 관망
    """
    weights = calculate_dynamic_weights(market_condition)
    
    # 시간대별 거래량 보정 적용
    volume_score = signals['volume_spike'] * signals['time_strength']
    
    total_score = (
        volume_score * weights['volume'] +
        signals['candle_pattern'] * weights['candle'] +
        signals['indicators'] * weights['indicator']
    ) * 100
    
    return total_score
```

---

## 4단계: 리스크 관리 계획 세우기 (안전장치 🛡️)

### 개념 설명
아무리 좋은 자리라도 100%는 없습니다. 진입 전 손실 대비 계획 필수.

### 손절 기준

#### ① 손절선 설정 (시장 상황별 차등 적용)
- **기본 원칙**: 
  - 지지선 하단 -2% (변동성 낮은 대형주, 상승장)
  - 지지선 하단 -3% (변동성 높은 중소형주, 횡보장)
  - 지지선 하단 -1.5% (하락장에서의 방어적 매매)
- **시간 손절**: 
  - 3일 내 목표가 방향으로 움직이지 않으면 청산
- **추가 하락 신호 시 즉시 손절**:
  - 거래량 동반 하락 돌파
  - 20일선 붕괴

#### ② 목표가 설정 (구체화됨)
```python
def calculate_target_prices(entry_price, resistance_levels):
    """
    단계별 목표가 설정
    """
    targets = []
    
    # 1차 목표: 가장 가까운 저항선
    nearest_resistance = find_nearest_resistance(entry_price, resistance_levels)
    targets.append({
        'price': nearest_resistance,
        'sell_ratio': 0.5,  # 50% 물량
        'reason': 'nearest_resistance'
    })
    
    # 2차 목표: 직전 고점 또는 피보나치 0.382
    fib_382 = calculate_fibonacci_level(entry_price, 0.382)
    previous_high = find_previous_high(entry_price)
    targets.append({
        'price': min(fib_382, previous_high),
        'sell_ratio': 0.3,  # 30% 물량
        'reason': 'fib_or_high'
    })
    
    # 3차 목표: 트레일링 스탑
    targets.append({
        'type': 'trailing_stop',
        'trigger': 0.05,  # 고점 대비 5% 하락 시
        'sell_ratio': 0.2,  # 나머지 20%
    })
    
    return targets
```

### 포지션 관리

#### 🆕 초기 리스크 관리 (2% 룰)
```python
def calculate_initial_position_size(capital, entry_price, stop_loss_price):
    """
    초기에는 고정 리스크(2% 룰) 적용
    """
    max_loss_amount = capital * 0.02  # 총자본의 2%
    loss_per_share = entry_price - stop_loss_price
    
    position_size = max_loss_amount / loss_per_share
    max_position = capital * 0.1  # 종목당 최대 10%
    
    return min(position_size, max_position)
```

#### 켈리 공식 기반 자금 관리 (100회 거래 후)
```python
def calculate_kelly_position_size(stats, capital):
    """
    충분한 데이터 축적 후 켈리 공식 적용
    """
    if stats['trade_count'] < 100:
        return None  # 아직 데이터 부족
    
    win_rate = stats['win_rate']
    avg_win = stats['avg_win_percent']
    avg_loss = stats['avg_loss_percent']
    
    # 켈리 비율 계산
    kelly = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
    
    # 안전 계수 적용
    safe_kelly = kelly * 0.25
    
    # 최대/최소 제한
    position_ratio = max(0.01, min(safe_kelly, 0.1))
    
    return capital * position_ratio
```

### 🆕 거래 비용 반영
```python
TRADING_COSTS = {
    'commission_buy': 0.00015,   # 0.015% (매수)
    'commission_sell': 0.00015,  # 0.015% (매도)
    'tax': 0.0023,              # 0.23% (매도시)
    'slippage_market': 0.002,   # 0.2% (시장가)
    'slippage_limit': 0.001     # 0.1% (지정가)
}

def calculate_net_return(entry_price, exit_price, order_type='limit'):
    """
    실제 순수익률 계산 (모든 비용 차감)
    """
    gross_return = (exit_price - entry_price) / entry_price
    
    # 비용 차감
    costs = (
        TRADING_COSTS['commission_buy'] +
        TRADING_COSTS['commission_sell'] +
        TRADING_COSTS['tax'] +
        TRADING_COSTS[f'slippage_{order_type}'] * 2
    )
    
    net_return = gross_return - costs
    return net_return
```

---

## 실전 적용 시나리오 (개선됨)

### 시나리오 1: 이상적인 매수 타이밍
1. **시장**: 코스피 20일선 위, 기울기 +0.8% (상승 추세)
2. **섹터**: 반도체 섹터 10% 상승 (정상 범위)
3. **종목**: 삼성전자 52,000원 도달
   - 52,000원: 직전 저점 + 20일선 + 라운드피겨 (3중 클러스터!)
4. **신호**: 
   - 오전 10시, 거래량 일평균 대비 180%
   - 망치형 캔들, RSI 28
5. **리스크 관리**:
   - 손절: 51,000원 (-1.9%)
   - 1차 목표: 53,500원 (+2.9%)
   - 2차 목표: 55,000원 (+5.8%)
   - 손익비: 1:3.0
6. **포지션**: 총자본의 8% 투입

### 시나리오 2: 부분 진입 상황
1. **시장**: 코스피 횡보, 20일선 기울기 +0.1%
2. **신호 강도**: 65점 (거래량 보통, 캔들 양호)
3. **대응**: 계획 물량의 50%만 초기 진입
4. **추가 매수 조건**: 종가 52,500원 돌파 시

---

## 🆕 백테스팅 및 검증 전략

### 과최적화 방지
```python
def walk_forward_analysis(data, strategy, window_size=252, step_size=63):
    """
    Walk-forward 분석으로 과최적화 방지
    - In-sample: 1년 (252일)
    - Out-of-sample: 3개월 (63일)
    """
    results = []
    
    for i in range(0, len(data) - window_size - step_size, step_size):
        # In-sample 최적화
        train_data = data[i:i+window_size]
        optimal_params = optimize_strategy(strategy, train_data)
        
        # Out-of-sample 검증
        test_data = data[i+window_size:i+window_size+step_size]
        performance = backtest(strategy, test_data, optimal_params)
        
        results.append(performance)
    
    return analyze_consistency(results)
```

### 다양한 시장 상황 테스트
```python
MARKET_SCENARIOS = {
    'bull_market': {'period': '2020-04~2021-06', 'weight': 0.2},
    'bear_market': {'period': '2022-01~2022-10', 'weight': 0.3},
    'sideways': {'period': '2023-01~2023-06', 'weight': 0.3},
    'high_volatility': {'period': '2020-03~2020-04', 'weight': 0.2}
}
```

---

## 🆕 API 안정성 및 예외 처리

### 장애 대응 시스템
```python
class TradingSystemFailsafe:
    def __init__(self):
        self.max_retries = 3
        self.emergency_contacts = ['email@example.com', '010-xxxx-xxxx']
        
    def execute_with_failsafe(self, func, *args, **kwargs):
        """
        모든 중요 함수를 failsafe로 감싸서 실행
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except APIConnectionError:
                retries += 1
                time.sleep(2 ** retries)  # 지수 백오프
            except CriticalError as e:
                self.emergency_liquidation()
                self.send_alert(e)
                raise
                
    def emergency_liquidation(self):
        """긴급 전량 청산"""
        positions = self.get_all_positions()
        for position in positions:
            self.market_sell(position, reason='emergency')
            
    def health_check(self):
        """시스템 상태 점검 (1분마다)"""
        checks = {
            'api_connection': self.check_api(),
            'data_feed': self.check_data_feed(),
            'position_sync': self.verify_positions(),
            'order_status': self.check_pending_orders()
        }
        
        if not all(checks.values()):
            self.send_alert("System health check failed", checks)
```

---

## 프로그램 구현 시 핵심 체크포인트

1. **실시간 감시**: 
   - 지지선 클러스터 근처(±2%) 도달 시 알림
   - 시간대별 거래량 추적
   
2. **자동 계산**: 
   - 신호 강도 점수 실시간 계산 (동적 가중치 적용)
   - 지지선 클러스터링 자동 감지
   
3. **리스크 관리**: 
   - 시장 상황별 손절가 자동 조정
   - 단계별 목표가 설정 및 추적
   
4. **로깅**: 
   - 모든 판단 근거와 시장 상황 저장
   - 거래 비용 포함 실제 수익률 기록
   
5. **백테스팅**: 
   - Walk-forward 분석 필수
   - 다양한 시장 시나리오 검증

---

## 주의사항

- 이 전략은 상승장/횡보장에 최적화됨
- 하락장에서는 손절 기준을 -1.5%로 강화
- 뉴스/공시 등 펀더멘털 이벤트 확인 필수
- 시가총액 500억 이상 종목 권장 (유동성)
- 초기 100회 거래까지는 고정 리스크(2% 룰) 적용
- 섹터 과열 시 진입 보류 또는 물량 축소