# oepnStock - Korean Stock Trading System

한국 주식 시장을 위한 고도화된 4단계 매매 전략 시스템

## 🎯 프로젝트 개요

oepnStock은 한국 주식 시장의 특성을 반영한 체계적인 매매 시스템입니다. 4단계 체크리스트 기반의 전략과 3단계(Phase 1-3) 보완 모듈을 통해 안정적이고 체계적인 투자를 지원합니다.

### 핵심 특징

- **4단계 체크리스트 전략**: 시장흐름파악 → 지지선감지 → 신호확인 → 리스크관리
- **한국 시장 특화**: KOSPI/KOSDAQ, 거래시간, 호가단위, 섹터 분류 등 반영
- **리스크 관리 중심**: 포트폴리오 집중도, 펀더멘털 필터, 갭 대응 전략
- **확장 가능한 구조**: Phase 1(필수) → Phase 2(성능향상) → Phase 3(고도화) 단계별 구현

## 📁 프로젝트 구조

```
oepnStock/
├── oepnstock/                    # 메인 패키지
│   ├── core/                     # 4단계 핵심 전략
│   │   ├── stage1_market_flow/   # 1단계: 시장 흐름 분석
│   │   ├── stage2_support_detection/  # 2단계: 지지선 감지
│   │   ├── stage3_signal_confirmation/  # 3단계: 신호 확인
│   │   └── stage4_risk_management/  # 4단계: 리스크 관리
│   ├── modules/                  # 보완 모듈
│   │   └── critical/            # Phase 1 필수 모듈
│   │       ├── fundamental_event_filter.py
│   │       ├── portfolio_concentration_manager.py
│   │       └── gap_trading_strategy.py
│   ├── utils/                   # 유틸리티
│   │   ├── korean_market.py     # 한국 시장 특화 기능
│   │   ├── technical_analysis.py  # 기술적 분석
│   │   └── market_data.py       # 시장 데이터 관리
│   └── config/                  # 설정 관리
├── examples/                    # 사용 예제
├── tests/                       # 테스트 코드
└── docs/                        # 문서
```

## 🚀 빠른 시작

### 설치

```bash
git clone https://github.com/your-repo/oepnStock.git
cd oepnStock
pip install -e .
```

### 기본 사용법

```python
from examples.basic_trading_example import BasicTradingSystem

# 매매 시스템 초기화
trading_system = BasicTradingSystem()

# 단일 종목 분석
result = trading_system.analyze_trading_opportunity('005930')  # 삼성전자
print(f"추천: {result['recommendation']}")
print(f"신뢰도: {result['confidence']:.1%}")

# 여러 종목 스크리닝
symbols = ['005930', '000660', '035420']
screening_result = trading_system.run_screening(symbols)
print(f"매수 후보: {len(screening_result['buy_candidates'])}개")
```

### 예제 실행

```bash
# 기본 매매 예제
python examples/basic_trading_example.py

# 전략 비교 분석
python examples/strategy_comparison.py

# 백테스팅 예제
python examples/backtesting_example.py

# 통합 테스트
python tests/test_integration.py
```

## 🔍 4단계 매매 전략

### Stage 1: 시장 흐름 분석 (Market Flow)
- **목적**: 거래 가능한 시장 환경인지 판단
- **지표**: 지수 위치(40점) + MA 기울기(30점) + 변동성(30점)
- **임계점**: 70점 이상에서 매매 허용

### Stage 2: 지지선 감지 (Support Detection)  
- **목적**: 안전한 매수 구간 식별
- **방법**: 과거 지지선 + 이동평균 + 거래량 분석
- **강화**: 겹치는 지지선(클러스터링) 우대

### Stage 3: 신호 확인 (Signal Confirmation)
- **목적**: 매수 타이밍의 적절성 검증  
- **지표**: RSI, MACD, 스토캐스틱, 볼린저밴드, 캔들패턴, 거래량
- **적응**: 시장 상황별 가중치 동적 조정

### Stage 4: 리스크 관리 (Risk Management)
- **목적**: 손실 제한 및 수익 최적화
- **방법**: 2% 규칙 → Kelly 공식 전환
- **계획**: 포지션 크기, 손절/목표가, 시나리오별 대응

## 🛡️ Phase 1 필수 보완 모듈

### Fundamental Event Filter
- **실적발표 블랙아웃**: D-3 ~ D+1 매매 금지
- **중요공시 모니터링**: 유상증자, 합병 등 위험 이벤트 차단
- **뉴스 감성 분석**: 부정적 뉴스 감지 시 포지션 조정

### Portfolio Concentration Manager  
- **포지션 수 제한**: 최대 5개 종목
- **단일 종목 비중**: 20% 한도
- **섹터 집중도**: 40% 한도  
- **상관관계 리스크**: 상관계수 0.7+ 종목군 60% 제한

### Gap Trading Strategy
- **갭 분석**: 상승/하락 갭 유형별 분류
- **대응 전략**: 되돌림 대기 vs 모멘텀 추종 vs 전면 재계산
- **리스크 조정**: 갭 크기별 포지션 조정

## 📊 사용 예제

### 기본 분석

```python
# 종목 분석
result = trading_system.analyze_trading_opportunity('005930')

# 결과 해석
if result['recommendation'] == 'BUY':
    print(f"💰 매수 추천: {result['entry_price']:,}원")
    print(f"📊 투자금액: {result['investment_amount']:,}원") 
    print(f"🛑 손절가: {result['stop_loss']:,}원")
    print(f"🎯 목표가: {result['target_prices']}")
```

### 리스크 체크

```python
# 펀더멘털 이벤트 체크
filter_decision = fundamental_filter.get_filter_decision('005930')
if not filter_decision.can_buy:
    print(f"❌ 매매 차단: {filter_decision.reason}")

# 포트폴리오 집중도 체크  
concentration_check = portfolio_manager.can_add_position(
    '005930', 1500000, current_portfolio
)
if not concentration_check.can_add:
    print(f"⚠️  집중도 제한: {concentration_check.blocking_reasons}")
```

### 갭 대응

```python
# 갭 분석
gap_analysis = gap_strategy.analyze_gap('005930', 50000, 52000)  # 4% 갭업
print(f"갭 유형: {gap_analysis.gap_type}")
print(f"채우기 확률: {gap_analysis.fill_probability:.1%}")

# 전략 결정
strategy = gap_strategy.determine_gap_strategy(gap_analysis)
print(f"권장 전략: {strategy.strategy_type}")
```

## 🧪 테스트 및 검증

### 통합 테스트 실행

```bash
python tests/test_integration.py
```

### 백테스팅 검증

```bash
python examples/backtesting_example.py
```

### 전략 성능 비교

```bash
python examples/strategy_comparison.py
```

## ⚙️ 설정 및 커스터마이징

### 기본 설정 수정

```python
from oepnstock.config import config

# 리스크 설정 조정
config.trading.max_positions = 3  # 최대 포지션 수
config.trading.max_single_position_ratio = 0.15  # 단일 종목 비중 15%
config.trading.stop_loss_ratio = 0.03  # 손절 비율 3%
```

### 사용자 정의 전략 추가

```python
class CustomStrategy:
    def analyze(self, symbol: str) -> Dict[str, Any]:
        # 사용자 정의 분석 로직
        return {
            'symbol': symbol,
            'recommendation': 'BUY',
            'confidence': 0.8
        }
```

## 🔮 로드맵

### Phase 2: 성능 향상 모듈 (예정)
- **변동성 적응 시스템**: 시장 변동성별 전략 조정
- **호가창 분석**: 실시간 매수/매도 압력 분석  
- **상관관계 리스크 관리**: 종목 간 상관관계 모니터링

### Phase 3: 고도화 모듈 (예정)
- **ML 신호 검증**: 머신러닝 기반 신호 품질 평가
- **장중 시간대별 전략**: 시간대별 차별화 전략
- **대량거래 감지**: 기관/외국인 거래 패턴 분석

### API 통합 계획
- **실시간 데이터**: KIS API, LS증권 API 연동
- **뉴스/공시**: 네이버금융, 다트 API 연동  
- **백테스팅 확장**: QuantConnect, Zipline 통합

## 📚 문서 및 가이드

### 📋 전략 문서 시스템
- **[전략 가이드](docs/STRATEGY_GUIDE.md)**: 백테스팅 전략 종합 가이드
- **[전략 검증](docs/STRATEGY_VALIDATION.md)**: 전략 검증 및 비교 방법론
- **개별 전략 문서**: `docs/strategies/` 디렉토리
  - [기본 전략](docs/strategies/DEFAULT_STRATEGY.md)
  - [적극적 전략](docs/strategies/AGGRESSIVE_STRATEGY.md)  
  - [보수적 전략](docs/strategies/CONSERVATIVE_STRATEGY.md)
  - [스켈핑 전략](docs/strategies/SCALPING_STRATEGY.md)
  - [스윙 전략](docs/strategies/SWING_STRATEGY.md)

### 🎯 백테스팅 프로파일 관리
```bash
# YAML 설정 파일
config/backtest_profiles.yaml

# 자동 문서 동기화
python utils/strategy_docs_sync.py

# 다중 전략 비교 실행
python examples/backtest_with_profiles.py
```

### 📊 전략 비교 실행
| 전략 | 리밸런싱 | MA조합 | RSI | 최대포지션 | 시장기준점 |
|---|---|---|---|---|---|
| `default` | 5일 | MA(5,20) | 14일 | 5개 | 70점 |
| `aggressive` | 3일 | MA(3,15) | 10일 | 7개 | 65점 |
| `conservative` | 7일 | MA(10,30) | 21일 | 3개 | 75점 |
| `scalping` | 1일 | MA(2,8) | 7일 | 10개 | 60점 |
| `swing` | 10일 | MA(20,60) | 30일 | 3개 | 80점 |

### 🔄 문서 업데이트 워크플로우
1. **YAML 수정** → `config/backtest_profiles.yaml`
2. **자동 동기화** → `python utils/strategy_docs_sync.py`
3. **검증 실행** → `python examples/backtest_with_profiles.py`
4. **문서 확인** → `docs/strategies/*.md` 업데이트 확인

### 📖 기존 문서
- [GUIDE.md](GUIDE.md): 상세한 전략 가이드 (한국어)
- [MODULE_GUIDE.md](MODULE_GUIDE.md): 모듈별 상세 설명 (한국어)  
- [CLAUDE.md](CLAUDE.md): 개발자를 위한 아키텍처 가이드

## ⚠️ 면책 조항

이 시스템은 교육 및 연구 목적으로 제공됩니다. 실제 투자 시에는:

1. **충분한 백테스팅**: 다양한 시장 상황에서 검증 필요
2. **리스크 관리**: 개인의 리스크 허용 범위 내에서만 사용
3. **지속적 모니터링**: 시장 환경 변화에 따른 전략 조정
4. **분산 투자**: 단일 전략에 의존하지 말고 분산 투자

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**oepnStock** - 체계적이고 안전한 한국 주식 투자를 위한 완전한 솔루션