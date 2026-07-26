# 데이터 계층 교체 기록

**작성일**: 2026-07-27
**대상**: 백테스트·분석에 쓰는 시세 데이터 소스
**변경**: 랜덤워크 Mock + yfinance → pykrx (KRX 공식 데이터)

---

## 왜 바꿨나

### 문제 1: 랜덤워크 데이터로는 전략을 검증할 수 없다

기존 `MockDataProvider`와 `MarketDataManager._generate_mock_data`는
`np.random.normal`로 가격을 만들었다.

```python
returns = np.random.normal(0.001, 0.02, days)   # 기존 코드
```

랜덤워크에는 지지선도 추세도 존재하지 않는다. 정의상 다음 가격이 과거와 독립이다.
따라서 이 데이터로 낸 백테스트 결과는 전략이 좋은지 나쁜지를 전혀 말해 주지 않는다.

이 데이터로 만든 수치가 문서에 사실처럼 기록됐다는 점이 더 큰 문제였다.
(신호 신뢰성 35% 개선, 거짓신호 68% 감소, 실거래 괴리 85% 감소)
해당 수치는 전부 `[미검증]`으로 되돌린다.

### 문제 2: yfinance는 한국 종목 처리가 부정확하다

- 수정주가 반영이 불완전해서 액면분할·증자 시점에 가격이 튄다.
- 거래정지와 상장폐지 상태를 구분해 주지 않는다.
- 과거 특정 시점의 상장 종목 리스트를 알 수 없어 생존편향을 줄일 방법이 없다.

### 문제 3: 실패가 조용히 감춰졌다

기존 `get_data_provider()`는 연결에 실패하면 경고만 남기고 Mock으로 넘어갔다.

```python
if free_data.check_connection():
    return free_data
else:
    logger.warning("Using mock data provider")   # 그대로 진행됐다
    return mock_data
```

그 결과 "실데이터로 돌렸다고 생각했지만 실제로는 랜덤 데이터였던" 상황이
발생할 수 있었다. 새 구현은 연결 실패 시 예외를 던지고 멈춘다.

---

## 무엇으로 바꿨나

`oepnstock/utils/krx_data_source.py` 의 `KRXDataProvider`를 쓴다.

| 항목 | 기존 | 변경 후 |
|---|---|---|
| 데이터 출처 | 랜덤워크 / yfinance | KRX 공식 (pykrx) |
| 수정주가 | 불확실 | `adjusted=True` 명시 |
| 거래정지일 | 구분 없음 | 거래량 0인 날 제외 옵션 |
| 과거 종목 리스트 | 불가 | `get_ticker_list_at()` 지원 |
| 상장폐지 확인 | 불가 | `get_delisted_between()` 지원 |
| 조회 실패 시 | Mock 으로 대체 | 예외 발생 후 중단 |
| 반복 조회 | 매번 네트워크 | 로컬 CSV 캐시 |

### 생존편향 대응

가장 중요한 변경이다. 백테스트 **시작 시점** 기준으로 종목을 고른다.

```python
universe = provider.get_ticker_list_at(date(2022, 1, 1), market="KOSPI")
```

오늘 기준 종목 리스트를 쓰면 그 사이 상장폐지된 종목이 전부 빠진다.
사라진 종목은 대체로 성과가 나빴으므로, 빼고 계산한 수익률은 실제보다 높게 나온다.

`scripts/check_data.py`가 그 규모를 숫자로 출력한다.

---

## 사전 준비

1. `pip install pykrx`
2. https://data.krx.co.kr 무료 회원가입
3. `.env.example`을 `.env`로 복사한 뒤 `KRX_ID`, `KRX_PW`를 채운다
   (pykrx 1.2.8 이상은 로그인이 필수다)

---

## 검증 규칙

이 변경 이후 **`reports/verification_*.md`에 있는 숫자만 다른 문서에 인용한다.**

근거가 없는 수치는 문서에 사실로 적지 않는다. 이 규칙 하나가
같은 문제가 반복되는 것을 막는다.
