# Conservative Enhanced 전략 (conservative)

**YAML 프로파일**: `config/backtest_profiles.yaml > conservative`

## 📊 전략 개요

### 전략 설명
Enhanced 기능을 이식한 안정성 극대화 전략
- 11개 독립 기술지표 조합
- 횡보장 거짓신호 필터링
- 4단계 검증 프로세스
- 소액 투자자 특화 최적화

### 투자 특성
- **프로파일명**: `conservative`
- **리밸런싱 주기**: 7일
- **최대 포지션**: 4개 (분산 효과 강화)
- **시장 진입 기준**: 75점 (보수적 기준 유지)

---

## ⚙️ 파라미터 설정 (YAML 연동)

### 백테스트 파라미터
```yaml
backtest:
  initial_capital: 10000000
  rebalance_frequency: 7
  signal_ma_short: 10
  signal_ma_long: 30
  signal_rsi_period: 21
  signal_rsi_overbought: 65
  min_recent_up_days: 3
  ma_trend_factor: 1.02
  sell_threshold_ratio: 0.92
```

### 거래 파라미터  
```yaml
trading:
  market_score_threshold: 75
  max_positions: 4
  max_single_position_ratio: 0.2
```

---

## 📈 신호 생성 로직 (Enhanced)

### 4단계 검증 프로세스
```python
# 1단계: 시장 환경 (보수적 기준)
market_score_check = market_score >= 75

# 2단계: Enhanced 기술적 신호 (11개 지표)
enhanced_signal_check = enhanced_signals.total_score >= 70

# 3단계: 횡보장 거짓신호 방지 (엄격)
whipsaw_filter_check = whipsaw_analysis.risk_level < 0.4

# 4단계: 비용 대비 수익성
cost_benefit_check = expected_return > trading_costs.total

# 최종 매매 결정
buy_decision = (
    market_score_check and
    enhanced_signal_check and 
    whipsaw_filter_check and
    cost_benefit_check
)
```

### 매도 조건 (Enhanced)
```python
# 기본 매도 조건
basic_sell = ma_short < ma_long * 0.92

# Enhanced 매도 조건
enhanced_sell = (
    basic_sell or
    enhanced_signals.exit_score <= 30 or
    whipsaw_analysis.exit_recommended
)
```

---

## 🎯 Enhanced 기능 통합

### 1. Market Score Calculator
- **투명한 점수 산출**: 지수위치(40) + 기울기(30) + 변동성(30)
- **보수적 임계값**: 75점 이상에서만 매매 허용

### 2. Enhanced Signal Engine
- **11개 독립 지표**: MA, RSI, MACD, BB, Volume 등
- **동적 가중치**: 시장 상황별 자동 조정
- **신호 강도**: 70점 이상 고신뢰도 진입

### 3. Whipsaw Filter
- **횡보장 감지**: 거짓신호 위험도 0.4 미만에서만 진입
- **노이즈 제거**: 불필요한 매매 68% 감소 예상 `[미검증 — scripts/verify.py 로 측정 예정]`

### 4. Realistic Trading Costs
- **현실적 비용**: 슬리피지 0.2%, 수수료 0.015% 반영
- **수익성 검증**: 거래비용 초과 수익만 진입

### 성과 목표 (상향 조정)
- **목표 수익률**: 시장 상황별 차등 설정
- **리스크 지표**: 샤프 비율 >1.0, 최대 낙폭 <10%
- **거래 효율**: 승률 >55%, 거래당 수익 양수
- **안정성**: 연속 손실 <3회, 변동성 최소화

---

## 📊 성과 추적

### 백테스트 실행
```bash
# 단일 전략 테스트
python examples/backtesting_example.py --profile conservative

# 다중 전략 비교
python examples/backtest_with_profiles.py
```

### 모니터링 지표
- 일간 수익률 변화
- 포지션 비중 점검  
- 거래 신호 정확도
- 비용 대비 효율성

---

## ⚠️ 주의사항

### 파라미터 변경시 체크리스트
- [ ] YAML 파일 백업
- [ ] 기존 성과와 비교 분석
- [ ] 아웃오브샘플 테스트 실시
- [ ] 문서 동기화 확인

### 리스크 관리
- 과최적화 방지: 최소 1년 이상 백테스트
- 시장 환경 변화: 정기적 재검증 필요
- 실거래 차이: 슬리피지, 거래비용 현실적 반영

---

*문서 생성일: 2025-08-09*  
*YAML 연동 상태: ✅ 동기화됨*  
*다음 업데이트: 파라미터 변경시 자동*
