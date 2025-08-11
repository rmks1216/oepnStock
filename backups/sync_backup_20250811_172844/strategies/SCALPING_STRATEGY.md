# 스켈핑 전략 (scalping)

**YAML 프로파일**: `config/backtest_profiles.yaml > scalping`

## 📊 전략 개요

### 전략 설명
초단기 매매 (매일 리밸런싱)

### 투자 특성
- **프로파일명**: `scalping`
- **리밸런싱 주기**: 1일
- **최대 포지션**: 10개
- **시장 진입 기준**: 60점

---

## ⚙️ 파라미터 설정 (YAML 연동)

### 백테스트 파라미터
```yaml
backtest:
  initial_capital: 10000000
  rebalance_frequency: 1
  signal_ma_short: 2
  signal_ma_long: 8
  signal_rsi_period: 7
  signal_rsi_overbought: 80
  min_recent_up_days: 1
  ma_trend_factor: 0.99
  sell_threshold_ratio: 0.98
```

### 거래 파라미터  
```yaml
trading:
  market_score_threshold: 60
  max_positions: 10
  max_single_position_ratio: 0.1
```

---

## 📈 신호 생성 로직

### 매수 조건
```python
buy_conditions = [
    ma_short > ma_long * 0.99,
    current_rsi < 80,
    up_days >= 1,
    market_score >= 60
]
```

### 매도 조건
```python
sell_condition = ma_short < ma_long * 0.98
```

---

## 🎯 최적화 방향

### 핵심 파라미터 튜닝
1. **이동평균 조합**: MA(2, 8) → 다양한 조합 테스트
2. **리밸런싱 주기**: 1일 → ±2일 범위 테스트
3. **시장 진입 기준**: 60점 → ±5점 범위 테스트

### 성과 목표
- **목표 수익률**: 시장 상황별 차등 설정
- **리스크 지표**: 샤프 비율 >0.5, 최대 낙폭 <20%
- **거래 효율**: 승률 >45%, 거래당 수익 양수

---

## 📊 성과 추적

### 백테스트 실행
```bash
# 단일 전략 테스트
python examples/backtesting_example.py --profile scalping

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
