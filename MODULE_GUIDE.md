# 4단계 체크리스트 전략 보완 모듈 가이드

## 개요
이 문서는 기본 4단계 체크리스트 전략을 보완하는 추가 모듈들의 구현 가이드입니다. 실전 운영 중 발견된 취약점을 보완하고 성능을 향상시키기 위한 기능들을 포함합니다.

---

## 🔴 필수 보완 모듈 (Critical)

### 1. 뉴스/공시 필터링 시스템

#### 목적
펀더멘털 이벤트로 인한 급격한 가격 변동 리스크를 사전에 차단합니다.

#### 구현 상세
```python
class FundamentalEventFilter:
    """
    매수 전 필수 체크 항목
    """
    def __init__(self):
        self.event_blackout_periods = {
            'earnings': {'before': 3, 'after': 1},  # 실적발표 D-3 ~ D+1
            'dividend': {'before': 1, 'after': 1},   # 배당락 D-1 ~ D+1
            'major_disclosure': {'before': 0, 'after': 2}  # 중요공시 D ~ D+2
        }
        
    def check_fundamental_events(self, symbol, date):
        """
        Returns: (is_safe, risk_events)
        """
        risk_events = []
        
        # 1. 실적 발표 체크
        earnings_date = self.get_earnings_date(symbol)
        if earnings_date:
            days_until = (earnings_date - date).days
            blackout = self.event_blackout_periods['earnings']
            if -blackout['after'] <= days_until <= blackout['before']:
                risk_events.append({
                    'type': 'earnings',
                    'date': earnings_date,
                    'days_until': days_until,
                    'risk_level': 'high'
                })
        
        # 2. 최근 공시 분석
        recent_disclosures = self.get_recent_disclosures(symbol, days=3)
        for disclosure in recent_disclosures:
            if self.is_major_disclosure(disclosure):
                risk_events.append({
                    'type': 'disclosure',
                    'title': disclosure['title'],
                    'date': disclosure['date'],
                    'risk_level': self.assess_disclosure_risk(disclosure)
                })
        
        # 3. 뉴스 감성 분석
        news_sentiment = self.analyze_news_sentiment(symbol, days=2)
        if news_sentiment['score'] < -0.3:  # 부정적
            risk_events.append({
                'type': 'negative_news',
                'score': news_sentiment['score'],
                'headlines': news_sentiment['negative_headlines'],
                'risk_level': 'medium'
            })
        
        # 4. 배당락일 체크
        ex_dividend_date = self.get_ex_dividend_date(symbol)
        if ex_dividend_date:
            days_until = (ex_dividend_date - date).days
            if 0 <= days_until <= 3:
                risk_events.append({
                    'type': 'dividend',
                    'date': ex_dividend_date,
                    'yield': self.get_dividend_yield(symbol),
                    'risk_level': 'medium'
                })
        
        is_safe = len([e for e in risk_events if e['risk_level'] == 'high']) == 0
        return is_safe, risk_events
    
    def is_major_disclosure(self, disclosure):
        """
        주요 공시 판별
        """
        major_keywords = [
            '유상증자', '무상증자', '합병', '분할', '상장폐지',
            '감자', '영업정지', '특별손실', '상한가', '하한가',
            '최대주주변경', '적자전환', '흑자전환'
        ]
        
        title = disclosure['title']
        return any(keyword in title for keyword in major_keywords)
    
    def get_filter_decision(self, symbol):
        """
        최종 매수 가능 여부 판단
        """
        is_safe, events = self.check_fundamental_events(symbol, datetime.now())
        
        if not is_safe:
            high_risk_events = [e for e in events if e['risk_level'] == 'high']
            return {
                'can_buy': False,
                'reason': f"고위험 이벤트 감지: {high_risk_events[0]['type']}",
                'retry_after': self.calculate_safe_date(high_risk_events[0])
            }
        
        medium_risk_count = len([e for e in events if e['risk_level'] == 'medium'])
        if medium_risk_count >= 2:
            return {
                'can_buy': False,
                'reason': "복수의 중간 위험 이벤트 감지",
                'retry_after': datetime.now() + timedelta(days=2)
            }
        
        return {'can_buy': True, 'warnings': events}
```

#### 사용 예시
```python
# 매수 신호 발생 시
if signal_strength > 70:
    # 펀더멘털 체크
    fund_filter = FundamentalEventFilter()
    decision = fund_filter.get_filter_decision(symbol)
    
    if not decision['can_buy']:
        logger.warning(f"매수 차단: {decision['reason']}")
        schedule_retry(symbol, decision['retry_after'])
    else:
        if decision.get('warnings'):
            position_size *= 0.7  # 경고가 있으면 포지션 축소
        execute_buy_order(symbol, position_size)
```

---

### 2. 포트폴리오 집중도 관리 시스템

#### 목적
과도한 집중 투자를 방지하고 리스크를 분산합니다.

#### 구현 상세
```python
class PortfolioConcentrationManager:
    """
    포트폴리오 집중도 관리
    """
    def __init__(self, config=None):
        self.config = config or {
            'max_positions': 5,
            'max_single_position': 0.2,  # 20%
            'max_sector_exposure': 0.4,   # 40%
            'max_correlation_exposure': 0.6,  # 상관계수 0.7 이상 종목들의 합
            'min_cash_ratio': 0.1  # 최소 현금 10%
        }
        
    def can_add_position(self, symbol, planned_size, current_portfolio):
        """
        새 포지션 추가 가능 여부 종합 판단
        """
        checks = {
            'position_count': self._check_position_count(current_portfolio),
            'single_position': self._check_single_position_limit(planned_size),
            'sector_concentration': self._check_sector_concentration(symbol, planned_size, current_portfolio),
            'correlation_risk': self._check_correlation_risk(symbol, current_portfolio),
            'cash_buffer': self._check_cash_buffer(planned_size, current_portfolio)
        }
        
        failed_checks = [k for k, v in checks.items() if not v['passed']]
        
        if failed_checks:
            return {
                'can_add': False,
                'failed_checks': failed_checks,
                'details': {k: v['reason'] for k, v in checks.items() if not v['passed']},
                'suggested_size': self._calculate_allowed_size(symbol, current_portfolio)
            }
        
        return {'can_add': True, 'warnings': self._generate_warnings(checks)}
    
    def _check_sector_concentration(self, symbol, planned_size, portfolio):
        """
        섹터 집중도 체크
        """
        sector = self.get_sector(symbol)
        current_sector_weight = sum(
            pos['weight'] for pos in portfolio 
            if self.get_sector(pos['symbol']) == sector
        )
        
        new_sector_weight = current_sector_weight + planned_size
        
        if new_sector_weight > self.config['max_sector_exposure']:
            return {
                'passed': False,
                'reason': f"{sector} 섹터 비중 {new_sector_weight:.1%} (한도 {self.config['max_sector_exposure']:.0%})"
            }
        
        return {'passed': True, 'current': current_sector_weight, 'new': new_sector_weight}
    
    def _check_correlation_risk(self, symbol, portfolio):
        """
        상관관계 리스크 체크
        """
        high_corr_weight = 0
        correlations = []
        
        for position in portfolio:
            corr = self.calculate_correlation(symbol, position['symbol'])
            correlations.append({
                'symbol': position['symbol'],
                'correlation': corr,
                'weight': position['weight']
            })
            
            if corr > 0.7:
                high_corr_weight += position['weight']
        
        if high_corr_weight > self.config['max_correlation_exposure']:
            return {
                'passed': False,
                'reason': f"높은 상관관계 종목 비중 {high_corr_weight:.1%} 초과"
            }
        
        return {'passed': True, 'correlations': correlations}
    
    def rebalance_suggestions(self, current_portfolio):
        """
        포트폴리오 재조정 제안
        """
        suggestions = []
        
        # 1. 과대 포지션 축소
        for position in current_portfolio:
            if position['weight'] > self.config['max_single_position']:
                suggestions.append({
                    'action': 'reduce',
                    'symbol': position['symbol'],
                    'current_weight': position['weight'],
                    'target_weight': self.config['max_single_position'],
                    'reason': '단일 종목 한도 초과'
                })
        
        # 2. 섹터 집중도 조정
        sector_weights = self._calculate_sector_weights(current_portfolio)
        for sector, weight in sector_weights.items():
            if weight > self.config['max_sector_exposure']:
                sector_positions = [p for p in current_portfolio if self.get_sector(p['symbol']) == sector]
                suggestions.extend(self._generate_sector_reduction_plan(sector_positions, weight))
        
        return suggestions
```

---

### 3. 갭(Gap) 대응 전략

#### 목적
시가 갭 발생 시 적절한 대응 전략을 자동으로 적용합니다.

#### 구현 상세
```python
class GapTradingStrategy:
    """
    갭 상황별 대응 전략
    """
    def __init__(self):
        self.gap_thresholds = {
            'major_gap_up': 0.03,    # 3% 이상
            'minor_gap_up': 0.01,    # 1% 이상
            'minor_gap_down': -0.01, # -1% 이하
            'major_gap_down': -0.03  # -3% 이하
        }
        
    def analyze_gap(self, yesterday_close, today_open):
        """
        갭 분석 및 대응 전략 결정
        """
        gap_ratio = (today_open - yesterday_close) / yesterday_close
        
        # 갭 유형 판별
        gap_type = self._classify_gap(gap_ratio)
        
        # 갭 채우기 확률 계산
        fill_probability = self._calculate_gap_fill_probability(gap_type, gap_ratio)
        
        # 대응 전략 결정
        strategy = self._determine_gap_strategy(gap_type, fill_probability)
        
        return {
            'gap_ratio': gap_ratio,
            'gap_type': gap_type,
            'fill_probability': fill_probability,
            'strategy': strategy
        }
    
    def _classify_gap(self, gap_ratio):
        """
        갭 분류
        """
        if gap_ratio >= self.gap_thresholds['major_gap_up']:
            return 'major_gap_up'
        elif gap_ratio >= self.gap_thresholds['minor_gap_up']:
            return 'minor_gap_up'
        elif gap_ratio <= self.gap_thresholds['major_gap_down']:
            return 'major_gap_down'
        elif gap_ratio <= self.gap_thresholds['minor_gap_down']:
            return 'minor_gap_down'
        else:
            return 'no_gap'
    
    def _determine_gap_strategy(self, gap_type, fill_probability):
        """
        갭 유형별 전략 결정
        """
        strategies = {
            'major_gap_up': self._major_gap_up_strategy,
            'minor_gap_up': self._minor_gap_up_strategy,
            'major_gap_down': self._major_gap_down_strategy,
            'minor_gap_down': self._minor_gap_down_strategy,
            'no_gap': self._no_gap_strategy
        }
        
        return strategies[gap_type](fill_probability)
    
    def _major_gap_up_strategy(self, fill_probability):
        """
        큰 상승 갭 전략
        """
        if fill_probability > 0.7:
            return {
                'action': 'wait_for_pullback',
                'entry_points': [
                    {'level': 'gap_fill', 'size': 0.5},
                    {'level': 'half_gap_fill', 'size': 0.3},
                    {'level': 'minor_pullback', 'size': 0.2}
                ],
                'stop_loss': 'below_gap',
                'time_limit': '11:00',  # 오전 중 진입
                'notes': '추격 매수 금지, 되돌림 매수만'
            }
        else:
            return {
                'action': 'momentum_play',
                'entry_points': [
                    {'level': 'breakout_confirmation', 'size': 0.3},
                    {'level': 'first_pullback', 'size': 0.7}
                ],
                'stop_loss': 'opening_price',
                'notes': '갭 유지 시 추세 추종'
            }
    
    def _major_gap_down_strategy(self, fill_probability):
        """
        큰 하락 갭 전략
        """
        return {
            'action': 'recalculate_all',
            'tasks': [
                'invalidate_previous_support',  # 기존 지지선 무효화
                'find_new_support_levels',       # 새 지지선 찾기
                'reduce_position_size',          # 포지션 축소
                'tighten_stop_loss'             # 손절선 강화
            ],
            'position_adjustment': 0.5,  # 기존 계획의 50%만
            'emergency_exit': 'below_major_support',
            'notes': '하락 갭은 추세 전환 신호일 수 있음'
        }
    
    def apply_gap_strategy(self, symbol, gap_analysis, original_plan):
        """
        갭 전략을 원래 매매 계획에 적용
        """
        strategy = gap_analysis['strategy']
        modified_plan = original_plan.copy()
        
        if strategy['action'] == 'wait_for_pullback':
            modified_plan['entry_type'] = 'limit_orders'
            modified_plan['entry_levels'] = self._calculate_pullback_levels(symbol, gap_analysis)
            modified_plan['max_wait_time'] = strategy['time_limit']
            
        elif strategy['action'] == 'recalculate_all':
            # 전면 재계산
            modified_plan = self._recalculate_trading_plan(symbol, strategy)
            
        return modified_plan
```

#### 사용 예시
```python
# 장 시작 시 갭 체크
def market_open_routine(watchlist):
    gap_analyzer = GapTradingStrategy()
    
    for symbol in watchlist:
        yesterday_close = get_yesterday_close(symbol)
        today_open = get_today_open(symbol)
        
        gap_analysis = gap_analyzer.analyze_gap(yesterday_close, today_open)
        
        if gap_analysis['gap_type'] != 'no_gap':
            # 기존 매매 계획 수정
            original_plan = trading_plans[symbol]
            modified_plan = gap_analyzer.apply_gap_strategy(symbol, gap_analysis, original_plan)
            
            trading_plans[symbol] = modified_plan
            logger.info(f"{symbol}: {gap_analysis['gap_type']} 갭 감지, 전략 수정됨")
```

---

## 🟡 성능 향상 모듈 (Important)

### 4. 변동성 기반 동적 조정 시스템

#### 목적
시장 변동성에 따라 매매 파라미터를 자동으로 조정합니다.

#### 구현 상세
```python
class VolatilityAdaptiveSystem:
    """
    ATR 기반 동적 파라미터 조정
    """
    def __init__(self):
        self.volatility_bands = {
            'low': {'max': 0.015, 'stop_mult': 1.0, 'target_mult': 1.0},
            'normal': {'max': 0.025, 'stop_mult': 1.2, 'target_mult': 1.1},
            'high': {'max': 0.035, 'stop_mult': 1.5, 'target_mult': 1.3},
            'extreme': {'max': float('inf'), 'stop_mult': 2.0, 'target_mult': 1.5}
        }
        
    def calculate_dynamic_parameters(self, symbol, base_params):
        """
        변동성에 따른 매매 파라미터 조정
        """
        # ATR 계산
        atr_data = self.calculate_atr_metrics(symbol)
        volatility_regime = self._classify_volatility(atr_data['atr_ratio'])
        
        # 파라미터 조정
        adjusted_params = {
            'stop_loss': base_params['stop_loss'] * self.volatility_bands[volatility_regime]['stop_mult'],
            'target_profit': base_params['target_profit'] * self.volatility_bands[volatility_regime]['target_mult'],
            'position_size': base_params['position_size'] * self._get_size_adjustment(volatility_regime),
            'holding_period': self._adjust_holding_period(base_params['holding_period'], volatility_regime),
            'entry_timing': self._get_entry_timing(volatility_regime)
        }
        
        # 추가 안전장치
        if volatility_regime == 'extreme':
            adjusted_params['max_loss_per_day'] = base_params['max_loss_per_day'] * 0.5
            adjusted_params['require_confirmation'] = True
            
        return {
            'volatility_metrics': atr_data,
            'regime': volatility_regime,
            'adjusted_params': adjusted_params,
            'warnings': self._generate_volatility_warnings(volatility_regime)
        }
    
    def calculate_atr_metrics(self, symbol, period=14):
        """
        ATR 관련 지표 계산
        """
        df = get_price_data(symbol, period * 3)
        
        # True Range 계산
        df['tr'] = df[['high', 'low', 'close']].apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['close']),
                abs(x['low'] - x['close'])
            ), axis=1
        )
        
        # ATR
        atr = df['tr'].rolling(period).mean().iloc[-1]
        avg_price = df['close'].rolling(period).mean().iloc[-1]
        
        # ATR 비율 및 추세
        atr_ratio = atr / avg_price
        atr_trend = (atr - df['tr'].rolling(period).mean().iloc[-period]) / atr
        
        return {
            'atr': atr,
            'atr_ratio': atr_ratio,
            'atr_trend': atr_trend,
            'current_tr': df['tr'].iloc[-1],
            'tr_percentile': self._calculate_tr_percentile(df['tr'])
        }
    
    def _get_size_adjustment(self, regime):
        """
        변동성별 포지션 크기 조정
        """
        adjustments = {
            'low': 1.2,      # 저변동성 시 포지션 확대
            'normal': 1.0,
            'high': 0.7,     # 고변동성 시 포지션 축소
            'extreme': 0.4   # 극단적 변동성 시 대폭 축소
        }
        return adjustments[regime]
```

---

### 5. 호가창 분석 시스템

#### 목적
호가창 불균형을 분석하여 단기 수급을 파악합니다.

#### 구현 상세
```python
class OrderBookAnalyzer:
    """
    실시간 호가창 분석
    """
    def __init__(self):
        self.imbalance_thresholds = {
            'strong_buy': 0.3,
            'weak_buy': 0.1,
            'neutral': 0.1,
            'weak_sell': -0.1,
            'strong_sell': -0.3
        }
        
    def analyze_order_book(self, symbol, depth=5):
        """
        호가창 종합 분석
        """
        order_book = self.get_order_book(symbol, depth)
        
        # 기본 불균형 계산
        basic_imbalance = self._calculate_basic_imbalance(order_book)
        
        # 가중 불균형 (가격 근접도 반영)
        weighted_imbalance = self._calculate_weighted_imbalance(order_book)
        
        # 대량 호가 감지
        large_orders = self._detect_large_orders(order_book)
        
        # 호가 두께 분석
        book_thickness = self._analyze_book_thickness(order_book)
        
        # 마이크로 구조 패턴
        micro_patterns = self._detect_micro_patterns(order_book)
        
        return {
            'basic_imbalance': basic_imbalance,
            'weighted_imbalance': weighted_imbalance,
            'signal': self._classify_signal(weighted_imbalance),
            'large_orders': large_orders,
            'book_thickness': book_thickness,
            'micro_patterns': micro_patterns,
            'confidence': self._calculate_confidence(order_book)
        }
    
    def _calculate_weighted_imbalance(self, order_book):
        """
        가격 가중 호가 불균형
        """
        mid_price = (order_book['asks'][0]['price'] + order_book['bids'][0]['price']) / 2
        
        weighted_bid_volume = 0
        weighted_ask_volume = 0
        
        for i, bid in enumerate(order_book['bids']):
            # 가격이 가까울수록 높은 가중치
            weight = 1 / (i + 1)
            weighted_bid_volume += bid['volume'] * weight
            
        for i, ask in enumerate(order_book['asks']):
            weight = 1 / (i + 1)
            weighted_ask_volume += ask['volume'] * weight
        
        imbalance = (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume)
        
        return imbalance
    
    def _detect_large_orders(self, order_book):
        """
        대량 호가 감지 (세력 매집/배포 신호)
        """
        all_volumes = [bid['volume'] for bid in order_book['bids']] + \
                     [ask['volume'] for ask in order_book['asks']]
        
        avg_volume = np.mean(all_volumes)
        std_volume = np.std(all_volumes)
        threshold = avg_volume + 2 * std_volume  # 2 표준편차
        
        large_orders = {
            'bids': [],
            'asks': []
        }
        
        for i, bid in enumerate(order_book['bids']):
            if bid['volume'] > threshold:
                large_orders['bids'].append({
                    'level': i + 1,
                    'price': bid['price'],
                    'volume': bid['volume'],
                    'ratio': bid['volume'] / avg_volume
                })
                
        for i, ask in enumerate(order_book['asks']):
            if ask['volume'] > threshold:
                large_orders['asks'].append({
                    'level': i + 1,
                    'price': ask['price'],
                    'volume': ask['volume'],
                    'ratio': ask['volume'] / avg_volume
                })
        
        return large_orders
    
    def _detect_micro_patterns(self, order_book):
        """
        마이크로 구조 패턴 감지
        """
        patterns = []
        
        # 1. 아이스버그 주문 패턴
        if self._detect_iceberg_pattern(order_book):
            patterns.append('iceberg_order')
            
        # 2. 레이어링 패턴 (가짜 호가)
        if self._detect_layering_pattern(order_book):
            patterns.append('potential_layering')
            
        # 3. 모멘텀 이그니션 패턴
        if self._detect_momentum_ignition(order_book):
            patterns.append('momentum_ignition')
            
        return patterns
    
    def generate_trading_signal(self, order_book_analysis, price_action):
        """
        호가창 분석 + 가격 움직임 종합 신호
        """
        signal_strength = 0
        
        # 호가 불균형 신호
        if order_book_analysis['signal'] == 'strong_buy':
            signal_strength += 30
        elif order_book_analysis['signal'] == 'weak_buy':
            signal_strength += 10
            
        # 대량 매수 호가
        if order_book_analysis['large_orders']['bids']:
            signal_strength += 20
            
        # 가격이 매수호가 쪽으로 움직임
        if price_action['direction'] == 'up' and order_book_analysis['weighted_imbalance'] > 0:
            signal_strength += 20
            
        # 마이크로 패턴
        if 'momentum_ignition' in order_book_analysis['micro_patterns']:
            signal_strength += 15
            
        return {
            'signal_strength': signal_strength,
            'confidence': order_book_analysis['confidence'],
            'action': self._determine_action(signal_strength)
        }
```

---

### 6. 상관관계 리스크 관리

#### 목적
포트폴리오 내 종목 간 상관관계를 분석하여 집중 리스크를 방지합니다.

#### 구현 상세
```python
class CorrelationRiskManager:
    """
    종목 간 상관관계 리스크 관리
    """
    def __init__(self):
        self.correlation_thresholds = {
            'high': 0.7,
            'moderate': 0.5,
            'low': 0.3
        }
        self.lookback_period = 60  # 60일 상관관계
        
    def analyze_portfolio_correlation(self, portfolio):
        """
        포트폴리오 전체 상관관계 분석
        """
        # 상관관계 매트릭스 계산
        corr_matrix = self._calculate_correlation_matrix(portfolio)
        
        # 주요 지표 계산
        metrics = {
            'average_correlation': self._calculate_average_correlation(corr_matrix),
            'max_correlation': self._find_max_correlation(corr_matrix),
            'correlation_clusters': self._find_correlation_clusters(corr_matrix),
            'diversification_ratio': self._calculate_diversification_ratio(portfolio, corr_matrix),
            'concentration_risk': self._assess_concentration_risk(corr_matrix)
        }
        
        # 리스크 평가
        risk_assessment = self._evaluate_correlation_risk(metrics)
        
        return {
            'correlation_matrix': corr_matrix,
            'metrics': metrics,
            'risk_assessment': risk_assessment,
            'recommendations': self._generate_recommendations(metrics, portfolio)
        }
    
    def check_new_position_correlation(self, new_symbol, current_portfolio):
        """
        신규 종목의 기존 포트폴리오 대비 상관관계 체크
        """
        correlations = []
        high_corr_exposure = 0
        
        for position in current_portfolio:
            corr = self._calculate_pairwise_correlation(new_symbol, position['symbol'])
            correlations.append({
                'symbol': position['symbol'],
                'correlation': corr,
                'weight': position['weight'],
                'risk_contribution': corr * position['weight']
            })
            
            if corr > self.correlation_thresholds['high']:
                high_corr_exposure += position['weight']
        
        # 상관관계 리스크 점수 계산
        risk_score = self._calculate_correlation_risk_score(correlations, high_corr_exposure)
        
        return {
            'correlations': sorted(correlations, key=lambda x: x['correlation'], reverse=True),
            'high_corr_exposure': high_corr_exposure,
            'risk_score': risk_score,
            'can_add': risk_score < 70,  # 70점 미만이면 추가 가능
            'suggested_weight': self._suggest_position_weight(risk_score)
        }
    
    def _calculate_correlation_matrix(self, portfolio):
        """
        포트폴리오 상관관계 매트릭스 계산
        """
        symbols = [p['symbol'] for p in portfolio]
        n = len(symbols)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self._calculate_pairwise_correlation(symbols[i], symbols[j])
        
        return pd.DataFrame(matrix, index=symbols, columns=symbols)
    
    def _find_correlation_clusters(self, corr_matrix):
        """
        높은 상관관계를 가진 종목 클러스터 찾기
        """
        clusters = []
        processed = set()
        
        for symbol1 in corr_matrix.index:
            if symbol1 in processed:
                continue
                
            cluster = [symbol1]
            for symbol2 in corr_matrix.columns:
                if symbol1 != symbol2 and corr_matrix.loc[symbol1, symbol2] > self.correlation_thresholds['high']:
                    cluster.append(symbol2)
                    processed.add(symbol2)
            
            if len(cluster) > 1:
                clusters.append({
                    'symbols': cluster,
                    'size': len(cluster),
                    'avg_correlation': self._calculate_cluster_correlation(cluster, corr_matrix)
                })
        
        return clusters
    
    def _generate_recommendations(self, metrics, portfolio):
        """
        상관관계 기반 포트폴리오 개선 제안
        """
        recommendations = []
        
        # 1. 과도한 상관관계 클러스터 해소
        for cluster in metrics['correlation_clusters']:
            if cluster['size'] >= 3:
                recommendations.append({
                    'type': 'reduce_cluster',
                    'priority': 'high',
                    'message': f"클러스터 종목 {cluster['symbols']} 중 일부 정리 필요",
                    'action': 'keep_best_performer',
                    'symbols': cluster['symbols']
                })
        
        # 2. 분산 효과 개선
        if metrics['diversification_ratio'] < 0.7:
            recommendations.append({
                'type': 'improve_diversification',
                'priority': 'medium',
                'message': "포트폴리오 분산 효과 부족",
                'action': 'add_low_correlation_assets'
            })
        
        # 3. 개별 종목 조정
        if metrics['max_correlation']['value'] > 0.85:
            recommendations.append({
                'type': 'high_correlation_pair',
                'priority': 'high',
                'message': f"{metrics['max_correlation']['pair']} 간 상관관계 {metrics['max_correlation']['value']:.2f} 너무 높음",
                'action': 'choose_one'
            })
        
        return recommendations

---

## 🟢 고도화 모듈 (Nice to Have)

### 7. 기계학습 신호 검증 시스템

#### 목적
전통적인 기술적 신호를 머신러닝으로 검증하여 정확도를 향상시킵니다.

#### 구현 상세
```python
class MLSignalValidator:
    """
    ML 기반 신호 검증 시스템
    """
    def __init__(self, model_path='models/signal_validator.pkl'):
        self.model = self._load_model(model_path)
        self.feature_extractors = self._initialize_feature_extractors()
        self.min_confidence = 0.7
        
    def validate_trading_signal(self, technical_signals, market_data, order_book_data):
        """
        종합적인 신호 검증
        """
        # 특징 추출
        features = self._extract_features(technical_signals, market_data, order_book_data)
        
        # 예측
        prediction = self.model.predict_proba(features.reshape(1, -1))
        success_probability = prediction[0][1]
        
        # 특징 중요도 분석
        feature_importance = self._analyze_feature_importance(features)
        
        # 신뢰구간 계산
        confidence_interval = self._calculate_confidence_interval(features)
        
        # 최종 판단
        validation_result = {
            'is_valid': success_probability > self.min_confidence,
            'probability': success_probability,
            'confidence_interval': confidence_interval,
            'key_factors': self._identify_key_factors(feature_importance),
            'risk_factors': self._identify_risk_factors(features),
            'suggested_adjustments': self._suggest_adjustments(success_probability, features)
        }
        
        return validation_result
    
    def _extract_features(self, technical_signals, market_data, order_book_data):
        """
        ML 모델용 특징 추출
        """
        features = []
        
        # 1. 기술적 신호 특징
        features.extend([
            technical_signals['rsi'],
            technical_signals['macd_histogram'],
            technical_signals['volume_ratio'],
            technical_signals['price_position'],  # (현재가 - 최저가) / (최고가 - 최저가)
            technical_signals['support_distance'],  # 지지선까지 거리
            technical_signals['resistance_distance']  # 저항선까지 거리
        ])
        
        # 2. 시장 상황 특징
        features.extend([
            market_data['market_trend_score'],
            market_data['sector_momentum'],
            market_data['vix_level'],
            market_data['foreign_net_buy_ratio'],
            market_data['program_buy_ratio']
        ])
        
        # 3. 호가창 특징
        features.extend([
            order_book_data['bid_ask_imbalance'],
            order_book_data['spread_ratio'],
            order_book_data['depth_imbalance'],
            order_book_data['large_order_ratio']
        ])
        
        # 4. 시간 특징
        features.extend([
            self._encode_time_of_day(),
            self._encode_day_of_week(),
            self._encode_days_to_expiry()  # 선물/옵션 만기일까지
        ])
        
        return np.array(features)
    
    def _suggest_adjustments(self, probability, features):
        """
        신호 개선을 위한 조정 제안
        """
        suggestions = []
        
        if 0.5 <= probability < self.min_confidence:
            # 경계선상의 신호 - 개선 가능
            weak_features = self._identify_weak_features(features)
            
            for feature in weak_features:
                if feature == 'volume_ratio' and features[2] < 1.2:
                    suggestions.append({
                        'type': 'wait_for_volume',
                        'message': '거래량 증가 대기',
                        'target': '평균 대비 150% 이상'
                    })
                elif feature == 'rsi' and 30 < features[0] < 40:
                    suggestions.append({
                        'type': 'wait_for_oversold',
                        'message': 'RSI 30 이하 진입 대기',
                        'expected_improvement': '+15% 성공률'
                    })
        
        return suggestions
    
    def update_model(self, new_trades_data):
        """
        새로운 거래 데이터로 모델 업데이트
        """
        if len(new_trades_data) < 100:
            return {'status': 'insufficient_data', 'required': 100, 'current': len(new_trades_data)}
        
        # 특징 및 레이블 준비
        X_new, y_new = self._prepare_training_data(new_trades_data)
        
        # 증분 학습
        self.model.partial_fit(X_new, y_new)
        
        # 성능 평가
        performance = self._evaluate_model_performance()
        
        # 모델 저장
        if performance['accuracy'] > 0.65:
            self._save_model()
            
        return {
            'status': 'updated',
            'new_accuracy': performance['accuracy'],
            'samples_used': len(new_trades_data)
        }
```

---

### 8. 시간대별 전략 세분화

#### 목적
한국 주식시장의 시간대별 특성을 반영한 맞춤 전략을 적용합니다.

#### 구현 상세
```python
class IntradayTimeStrategy:
    """
    시간대별 세분화 전략
    """
    def __init__(self):
        self.time_zones = {
            'pre_opening': {'start': '08:30', 'end': '09:00'},
            'opening_auction': {'start': '09:00', 'end': '09:10'},
            'morning_volatility': {'start': '09:10', 'end': '09:30'},
            'morning_trend': {'start': '09:30', 'end': '11:00'},
            'lunch_lull': {'start': '11:00', 'end': '13:00'},
            'afternoon_trend': {'start': '13:00', 'end': '14:30'},
            'closing_volatility': {'start': '14:30', 'end': '15:20'},
            'closing_auction': {'start': '15:20', 'end': '15:30'}
        }
        
    def get_time_zone_strategy(self, current_time):
        """
        현재 시간대에 맞는 전략 반환
        """
        zone = self._identify_time_zone(current_time)
        
        strategies = {
            'opening_auction': self._opening_auction_strategy(),
            'morning_volatility': self._morning_volatility_strategy(),
            'morning_trend': self._morning_trend_strategy(),
            'lunch_lull': self._lunch_lull_strategy(),
            'afternoon_trend': self._afternoon_trend_strategy(),
            'closing_volatility': self._closing_volatility_strategy()
        }
        
        return strategies.get(zone, self._default_strategy())
    
    def _opening_auction_strategy(self):
        """
        개장 단일가 시간대 전략
        """
        return {
            'name': 'opening_auction',
            'characteristics': '극심한 변동성, 갭 발생 가능',
            'entry_rules': {
                'avoid_market_order': True,
                'wait_for_stability': True,
                'min_time_after_open': 600,  # 10분 대기
                'gap_strategy': 'apply_gap_rules'
            },
            'signal_adjustments': {
                'volume_weight': 0.3,  # 거래량 신호 신뢰도 낮춤
                'technical_weight': 0.4,
                'order_book_weight': 0.3
            },
            'risk_management': {
                'position_size_multiplier': 0.5,
                'wider_stop_loss': 1.5,
                'avoid_full_position': True
            }
        }
    
    def _morning_trend_strategy(self):
        """
        오전 추세 시간대 전략
        """
        return {
            'name': 'morning_trend',
            'characteristics': '가장 활발한 거래, 추세 형성',
            'entry_rules': {
                'prefer_trend_following': True,
                'volume_confirmation': 'required',
                'breakout_trading': True
            },
            'signal_adjustments': {
                'volume_weight': 0.5,
                'technical_weight': 0.3,
                'order_book_weight': 0.2
            },
            'risk_management': {
                'position_size_multiplier': 1.0,
                'normal_stop_loss': 1.0,
                'trailing_stop': True
            },
            'special_patterns': {
                'first_hour_high_break': {
                    'description': '첫 1시간 고점 돌파',
                    'success_rate': 0.68,
                    'additional_filter': 'sector_momentum'
                }
            }
        }
    
    def _lunch_lull_strategy(self):
        """
        점심시간대 전략
        """
        return {
            'name': 'lunch_lull',
            'characteristics': '거래량 감소, 방향성 약함',
            'entry_rules': {
                'avoid_new_positions': True,
                'manage_existing_only': True,
                'tighten_stops': True
            },
            'signal_adjustments': {
                'min_signal_strength': 80,  # 매우 강한 신호만
                'require_multiple_confirmations': True
            },
            'risk_management': {
                'no_new_trades': True,
                'reduce_if_profitable': True
            }
        }
    
    def _closing_volatility_strategy(self):
        """
        장 마감 변동성 시간대 전략
        """
        return {
            'name': 'closing_volatility',
            'characteristics': '기관/외인 정리매매, 높은 변동성',
            'entry_rules': {
                'day_trading_only': True,  # 당일 청산 목표
                'quick_profit_target': True,
                'avoid_overnight': True
            },
            'signal_adjustments': {
                'volume_weight': 0.4,
                'order_book_weight': 0.4,  # 수급 중요
                'technical_weight': 0.2
            },
            'risk_management': {
                'position_size_multiplier': 0.6,
                'tight_stop_loss': 0.8,
                'must_close_today': True
            },
            'special_considerations': {
                'program_trading_impact': 'high',
                'check_settlement': True,
                'option_expiry_effect': 'check_if_expiry_week'
            }
        }
    
    def apply_time_strategy(self, base_signal, current_time):
        """
        시간대별 전략을 기본 신호에 적용
        """
        time_strategy = self.get_time_zone_strategy(current_time)
        
        # 신호 강도 조정
        adjusted_signal = base_signal.copy()
        
        # 시간대별 가중치 적용
        if 'signal_adjustments' in time_strategy:
            adj = time_strategy['signal_adjustments']
            adjusted_signal['strength'] = (
                base_signal['volume_signal'] * adj.get('volume_weight', 0.5) +
                base_signal['technical_signal'] * adj.get('technical_weight', 0.3) +
                base_signal['orderbook_signal'] * adj.get('order_book_weight', 0.2)
            ) * 100
        
        # 최소 신호 강도 체크
        min_strength = time_strategy.get('signal_adjustments', {}).get('min_signal_strength', 60)
        if adjusted_signal['strength'] < min_strength:
            adjusted_signal['action'] = 'no_trade'
            adjusted_signal['reason'] = f"시간대별 최소 강도 {min_strength} 미달"
        
        # 리스크 관리 적용
        if 'risk_management' in time_strategy:
            risk = time_strategy['risk_management']
            adjusted_signal['position_size'] *= risk.get('position_size_multiplier', 1.0)
            adjusted_signal['stop_loss'] *= risk.get('stop_loss_multiplier', 1.0)
            
            if risk.get('must_close_today'):
                adjusted_signal['exit_time'] = '15:15'
        
        return adjusted_signal
```

---

## 구현 우선순위 및 통합 가이드

### Phase 1: 필수 모듈 (1-2주)
1. **뉴스/공시 필터링**: 예상치 못한 손실 방지
2. **포트폴리오 집중도 관리**: 리스크 분산
3. **갭 대응 전략**: 시가 변동 대응

### Phase 2: 성능 향상 (3-4주)
4. **변동성 기반 조정**: 시장 상황 적응
5. **호가창 분석**: 단기 수급 파악
6. **상관관계 리스크**: 숨은 집중 리스크 제거

### Phase 3: 고도화 (1-2개월)
7. **ML 신호 검증**: 정확도 향상
8. **시간대별 전략**: 한국 시장 특성 활용

### 통합 시 주의사항
- 각 모듈은 독립적으로 작동 가능하도록 설계
- 모듈 간 의존성 최소화
- 성능 측정 지표 별도 구현
- A/B 테스트로 효과 검증

### 모니터링 지표
```python
MODULE_METRICS = {
    'fundamental_filter': ['filtered_trades', 'avoided_losses'],
    'portfolio_manager': ['concentration_score', 'correlation_score'],
    'gap_strategy': ['gap_handled', 'gap_profit_loss'],
    'volatility_adaptive': ['parameter_adjustments', 'risk_reduction'],
    'order_book': ['signal_accuracy', 'false_signals'],
    'ml_validator': ['prediction_accuracy', 'profit_improvement']
}
```