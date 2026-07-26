# 작업 지시서 — 1순위: 데이터 검증 기반 구축

**작성일**: 2026-07-27
**용도**: 새 대화 세션에 첨부해 바로 이어서 진행
**선행 조건**: 코드·설정 파일 커밋 완료

---

## 새 세션 시작 문구

이 파일을 첨부하거나 프로젝트 지식에 올린 뒤 아래처럼 말하면 된다.

> 작업 지시서 참고해서 1순위 이어서 진행. 현재 [완료 단계]까지 했음.

`check_data.py`나 `verify.py` 출력이 있으면 그대로 붙인다.

---

## 1. 배경 — 왜 이 작업을 하는가

프로젝트는 **설계 문서와 모듈은 앞서 있으나 검증 근거가 비어 있는 상태**다.

| 문제 | 내용 |
|---|---|
| 근거 없는 수치 | 35% / 68% / 85% 개선 주장에 재현 스크립트도 결과 파일도 없다 |
| 가짜 데이터 | 백테스트가 `np.random.normal` 랜덤워크를 썼다. 패턴이 없으므로 검증 불가 |
| 문서가 앞섬 | `SimpleBacktester`가 Enhanced 모듈 4개를 호출하지 않는다 |
| 생존편향 방치 | yfinance는 상장폐지 종목을 다룰 수 없다 |

**유지할 것**: 전략별 연간 거래비용 표는 거래 횟수와 비용률로 직접 계산한
값이라 근거가 명확하다. 이 프로젝트에서 유일하게 검증된 숫자다.
(Swing 6.5% / Conservative 9.3% / Default 12.0% / Aggressive 14.8% / Scalping 52.6%)

---

## 2. 작업 체크리스트

### 완료

- [x] `oepnstock/utils/krx_data_source.py` 작성
- [x] `scripts/check_data.py` 작성
- [x] `scripts/verify.py` 작성
- [x] `.gitignore` / `.env.example` 작성
- [x] `docs/DATA_LAYER_MIGRATION.md` 작성

### 남은 작업

- [ ] **T1.** KRX 가입 + `.env` 설정
- [ ] **T2.** 기존 mock 경로 교체
- [ ] **T3.** `check_data.py` 통과
- [ ] **T4.** `verify.py` 실행 → 첫 실측치 확보
- [ ] **T5.** 문서 미검증 수치 정리

---

## T1. KRX 가입 및 환경변수 설정

1. https://data.krx.co.kr 무료 회원가입
2. `pip install pykrx`
3. `cp .env.example .env` 후 `KRX_ID`, `KRX_PW` 채우기

확인 (비밀번호는 출력하지 않는다):

```bash
python -c "import os; print('ID:', bool(os.getenv('KRX_ID')), '| PW길이:', len(os.getenv('KRX_PW') or ''))"
```

`.env` 방식을 쓰면 `check_data.py`와 `verify.py` 상단 import 위에 두 줄을 추가한다.

```python
from dotenv import load_dotenv
load_dotenv()
```

**완료 판정**: `ID: True` 가 출력된다.

---

## T2. 기존 mock 경로 교체

찾기:

```bash
grep -rn "get_data_provider\|_generate_mock_data\|MockDataProvider" --include="*.py" .
```

바꾸기:

| 기존 | 변경 후 |
|---|---|
| `from oepnstock.utils.free_data_sources import get_data_provider` | `from oepnstock.utils.krx_data_source import get_data_provider` |
| `MarketDataManager._generate_mock_data(...)` | `KRXDataProvider.get_historical_data(...)` |

**주의**: 새 `get_data_provider()`는 연결 실패 시 Mock으로 넘어가지 않고 예외를 던진다.
이건 의도된 동작이다. 조용한 대체가 가짜 결과를 만들었기 때문이다.

**완료 판정**: grep 결과에 `free_data_sources`가 남지 않는다.

---

## T3. 데이터 점검

```bash
python scripts/check_data.py
```

4개 항목을 검사한다.

1. KRX 접속 여부
2. 삼성전자 수정주가 품질 (고가·저가·종가 관계, 일간 30% 초과 변동)
3. 과거 시점 종목 리스트 조회 가능 여부
4. **기간 중 사라진 종목 수 = 생존편향 규모**

**완료 판정**: "모든 점검 통과" 출력.

**보고할 값**: 4번 항목의 사라진 종목 개수와 비율. 지금까지의 백테스트가
얼마나 낙관적이었는지 알려 주는 첫 실측치다.

### 실패 시 대응

| 증상 | 원인 | 조치 |
|---|---|---|
| KRX 로그인 실패 | 환경변수 미설정 | T1 재확인. Windows는 `setx` 후 창을 다시 연다 |
| 데이터 비어 있음 | 조회 기간이 휴장일뿐 | 기간을 넓힌다 |
| 30% 초과 변동 경고 | 수정주가 미적용 | `adjusted=True` 확인 |
| 응답 없음·차단 | 요청 과다 | `request_interval` 값을 키운다 |

---

## T4. 검증 실행

```bash
python scripts/verify.py
```

옵션:

```bash
python scripts/verify.py --profiles conservative swing
python scripts/verify.py --start 2022-01-01 --end 2023-12-31 --universe 30
```

결과는 `reports/verification_YYYYMMDD_HHMM.md`로 저장된다.

**완료 판정**: 리포트 파일이 생성되고, 최소 한 개 프로파일이 성공한다.

**예상**: 문서에 적힌 것보다 성과가 나쁘게 나올 가능성이 높다.
랜덤워크는 노이즈만 있어 극단적 결과가 잘 안 나오고, 생존편향을 걷어내면
수익률이 내려가기 때문이다. **나쁜 숫자를 보는 것이 이 단계의 목적이다.**

### 실패 시 확인할 것

`SimpleBacktester`의 인자 이름이 다를 수 있다. 아래로 실제 시그니처를 확인한다.

```bash
grep -n "def run_backtest" -A 10 examples/backtesting_example.py
```

결과 사전의 키 이름도 다를 수 있다. `verify.py`의 `METRICS` 목록을 맞춰 수정한다.

---

## T5. 문서 미검증 수치 정리

| 파일 | 대상 |
|---|---|
| `docs/PROJECT_STATUS.md` | 신호 신뢰성 35% / 거짓신호 68% / 실거래 괴리 85% |
| `docs/ENHANCED_STRATEGY_FRAMEWORK.md` | 횡보장 거짓신호 68% 감소 |
| `docs/CONSERVATIVE_ENHANCED_SUMMARY.md` | 예상 성과 개선 표 전체 |
| `docs/strategies/DEFAULT_STRATEGY.md` | 개선 사항별 효과 4개 항목 |

표기 방식 (지우지 말고 표시만 한다, 나중에 실측치로 채울 자리다):

```markdown
- 횡보장 거짓신호 감소 효과 `[미검증 — scripts/verify.py 로 측정 예정]`
```

**건드리지 말 것**: 전략별 연간 거래비용 표. 근거가 명확하다.

**완료 판정**: `grep -rn "35%\|68%\|85%" docs/`에서 `[미검증]`이 붙지 않은 항목이 없다.

---

## 3. 확립할 규칙

> **`reports/verification_*.md`에 있는 숫자만 다른 문서에 인용한다.**

이 규칙 하나가 같은 문제의 재발을 막는다. 새 주장을 문서에 쓰기 전에
"이 숫자가 어느 리포트 몇 번째 줄에서 왔는가"를 답할 수 있어야 한다.

---

## 4. 다음 단계 (2순위 — T4 결과 확보 후)

1. **Enhanced 모듈을 파이프라인에 연결한다.** `SimpleBacktester`가
   `MarketScoreCalculator`, `EnhancedSignalEngine`, `WhipsawFilter`를
   실제로 호출하도록 고친다. 미완료로 남아 있는 Phase 2가 이것이다.
2. **켬/끔을 같은 데이터로 비교한다.** 그래야 개선 효과를 숫자로 말할 수 있고,
   효과가 없으면 해당 모듈을 걷어낼 수 있다.
3. **지표 계산 정확성 테스트를 추가한다.** 현재 테스트는 반환 타입과 범위만
   본다. 손으로 계산한 RSI 값과 코드 출력을 대조하는 테스트가 필요하다.
4. **README를 동기화 대상에 포함한다.** README는 conservative 최대 포지션을
   3개로, Conservative Enhanced 문서는 4개로 적어 서로 어긋나 있다.

---

## 5. 참고 — 확인된 기술 사실

- pykrx 1.2.8부터 `KRX_ID` / `KRX_PW` 환경변수가 필수다.
- `get_market_ohlcv_by_date(fromdate, todate, ticker, freq='d', adjusted=True)`
  에서 `adjusted`가 수정주가이며 기본값이 `True`다.
- `get_market_ticker_list(date, market)`에 과거 날짜를 넣으면 그 시점의
  상장 종목이 나온다. 생존편향을 줄이는 핵심 기능이다.
