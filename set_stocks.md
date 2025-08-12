
-----

# 단기 매매 종목 자동 선별 기능 기획안 (v2.3)

>   - **버전**: 2.3
>   - **최종 수정**: 2025년 8월 12일
>   - **변경 사항**: 변수명 표준 정의, 최적화 파라미터 상세 설명표 추가, API 에러 응답 예시 추가.

-----

## 목차

\<details\>
\<summary\>\<strong\>전체 목차 보기/숨기기\</strong\>\</summary\>

### Part 1. 기본 스크리너 명세

  - [1. 개요 및 목표](https://www.google.com/search?q=%231-%EA%B0%9C%EC%9A%94-%EB%B0%8F-%EB%AA%A9%ED%91%9C)
  - [2. 용어 정의](https://www.google.com/search?q=%232-%EC%9A%A9%EC%96%B4-%EC%A0%95%EC%9D%98)
  - [3. 데이터 소스 및 계약](https://www.google.com/search?q=%233-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%86%8C%EC%8A%A4-%EB%B0%8F-%EA%B3%84%EC%95%BD)
  - [4. 특징(Feature) 계산 규격](https://www.google.com/search?q=%234-%ED%8A%B9%EC%A7%95feature-%EA%B3%84%EC%82%B0-%EA%B7%9C%EA%B2%A9)
  - [5. 하드 필터 (Pass/Fail)](https://www.google.com/search?q=%235-%ED%95%98%EB%93%9C-%ED%95%84%ED%84%B0-passfail)
  - [6. 점수화 (Scoring)](https://www.google.com/search?q=%236-%EC%A0%90%EC%88%98%ED%99%94-scoring)
  - [7. 출력 규격](https://www.google.com/search?q=%237-%EC%B6%9C%EB%A0%A5-%EA%B7%9C%EA%B2%A9)
  - [8. 구성 파라미터 (Configuration)](https://www.google.com/search?q=%238-%EA%B5%AC%EC%84%B1-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-configuration)
  - [9. 처리 파이프라인](https://www.google.com/search?q=%239-%EC%B2%98%EB%A6%AC-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8)
  - [10. 전략 프로파일 (프리셋)](https://www.google.com/search?q=%2310-%EC%A0%84%EB%9E%B5-%ED%94%84%EB%A1%9C%ED%8C%8C%EC%9D%BC-%ED%94%84%EB%A6%AC%EC%85%8B)
  - [11. 예외/엣지 케이스 처리](https://www.google.com/search?q=%2311-%EC%98%88%EC%99%B8%EC%97%A3%EC%A7%80-%EC%BC%80%EC%9D%B4%EC%8A%A4-%EC%B2%98%EB%A6%AC)
  - [12. 기본 백테스트 사양](https://www.google.com/search?q=%2312-%EA%B8%B0%EB%B3%B8-%EB%B0%B1%ED%85%8C%EC%8A%A4%ED%8A%B8-%EC%82%AC%EC%96%91)
  - [13. 모니터링 및 로깅](https://www.google.com/search?q=%2313-%EB%AA%A8%EB%8B%88%ED%84%B0%EB%A7%81-%EB%B0%8F-%EB%A1%9C%EA%B9%85)
  - [14. API 인터페이스](https://www.google.com/search?q=%2314-api-%EC%9D%B8%ED%84%B0%ED%8E%98%EC%9D%B4%EC%8A%A4)
  - [15. 기본 수용 기준](https://www.google.com/search?q=%2315-%EA%B8%B0%EB%B3%B8-%EC%88%98%EC%9A%A9-%EA%B8%B0%EC%A4%80)
  - [16. 구현 체크리스트](https://www.google.com/search?q=%2316-%EA%B5%AC%ED%98%84-%EC%B2%B4%ED%81%AC%EB%A6%AC%EC%8A%A4%ED%8A%B8)
  - [17. 확장 아이디어](https://www.google.com/search?q=%2317-%ED%99%95%EC%9E%A5-%EC%95%84%EC%9D%B4%EB%94%94%EC%96%B4)

### Part 2. 고급 모듈 및 상세 설계

  - [18. 파라미터 최적화 상세 설계](https://www.google.com/search?q=%2318-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%EC%B5%9C%EC%A0%81%ED%99%94-%EC%83%81%EC%84%B8-%EC%84%A4%EA%B3%84)
  - [19. 리스크 및 포지션 사이징 상세 설계](https://www.google.com/search?q=%2319-%EB%A6%AC%EC%8A%A4%ED%81%AC-%EB%B0%8F-%ED%8F%AC%EC%A7%80%EC%85%98-%EC%82%AC%EC%9D%B4%EC%A7%95-%EC%83%81%EC%84%B8-%EC%84%A4%EA%B3%84)
  - [20. ML 보조 예측 상세 설계](https://www.google.com/search?q=%2320-ml-%EB%B3%B4%EC%A1%B0-%EC%98%88%EC%B8%A1-%EC%83%81%EC%84%B8-%EC%84%A4%EA%B3%84)
  - [21. 백테스트 및 최적화 모듈 통합 설계](https://www.google.com/search?q=%2321-%EB%B0%B1%ED%85%8C%EC%8A%A4%ED%8A%B8-%EB%B0%8F-%EC%B5%9C%EC%A0%81%ED%99%94-%EB%AA%A8%EB%93%88-%ED%86%B5%ED%95%A9-%EC%84%A4%EA%B3%84)
  - [22. 포지션 사이징 계산 로직](https://www.google.com/search?q=%2322-%ED%8F%AC%EC%A7%80%EC%85%98-%EC%82%AC%EC%9D%B4%EC%A7%95-%EA%B3%84%EC%82%B0-%EB%A1%9C%EC%A7%81)
  - [23. 멀티타임프레임 및 시장 국면 연동](https://www.google.com/search?q=%2323-%EB%A9%80%ED%8B%B0%ED%83%80%EC%9E%84%ED%94%84%EB%A0%88%EC%9E%84-%EB%B0%8F-%EC%8B%9C%EC%9E%A5-%EA%B5%AD%EB%A9%B4-%EC%97%B0%EB%8F%99)
  - [24. 경량 ML 보조 예측 운영 가이드](https://www.google.com/search?q=%2324-%EA%B2%BD%EB%9F%89-ml-%EB%B3%B4%EC%A1%B0-%EC%98%88%EC%B8%A1-%EC%9A%B4%EC%98%81-%EA%B0%80%EC%9D%B4%EB%93%9C)
  - [25. 운영 대시보드 요구사항](https://www.google.com/search?q=%2325-%EC%9A%B4%EC%98%81-%EB%8C%80%EC%8B%9C%EB%B3%B4%EB%93%9C-%EC%9A%94%EA%B5%AC%EC%82%AC%ED%95%AD)
  - [26. 최종 수용 기준](https://www.google.com/search?q=%2326-%EC%B5%9C%EC%A2%85-%EC%88%98%EC%9A%A9-%EA%B8%B0%EC%A4%80)

\</details\>

-----

## Part 1. 기본 스크리너 명세

### 1\. 개요 및 목표

  - **목표**: 실시간 또는 장마감 데이터 기반으로 종목을 자동 선별, `score`와 `pass`를 계산 후 랭킹 출력.
  - **비범위**: 주문/체결 실행, 계좌 관리. 단, 손절/익절 참고값(ATR 기반)은 제공.

### 2\. 용어 정의

  - **OHLCV**: 시가, 고가, 저가, 종가, 거래량.
  - **거래대금**: `close × volume` (단위: KRW)
  - **변동폭(range\_pct)**: `(high - low) / open`
  - **VWAP**: 거래량가중평균가.
  - **ATR**: 평균 변동성 지표 (Average True Range).
  - **체결강도**: `총 매수체결량 / 총 매도체결량`.

#### 2.1. 코딩 스타일 및 변수명 규칙

> 구현의 일관성과 가독성 유지를 위해 다음 표준을 준수한다.

  - **JSON Keys, Python 변수/인스턴스**: **`snake_case`** 사용을 원칙으로 한다.
      - 예: `trade_risk_amt`, `account_equity`
  - **Python 함수/메서드**: **`snake_case`** 사용을 원칙으로 한다.
      - 예: `calculate_position_size()`, `run_backtest()`
  - **Python 클래스**: **`PascalCase`** (또는 `CapWords`) 사용을 원칙으로 한다.
      - 예: `BacktestEngine`, `RiskManager`
  - **상수 (Constants)**: **`UPPER_SNAKE_CASE`** 사용을 원칙으로 한다.
      - 예: `DEFAULT_RISK_PERCENT = 0.01`, `MAX_CONCURRENT_POSITIONS = 5`

### 3\. 데이터 소스 및 계약

  - **입력 데이터**: 실시간 시세 API, 뉴스/공시 API, 종목 메타데이터 API.
  - **JSON 스키마**: `symbol`, `bars`(OHLCV 배열), `meta`(리스크 플래그 등).
  - **품질 요구사항**: 시간 정렬, 결측치 방지, 단위·타임존 표준화.

### 4\. 특징(Feature) 계산 규격

  - **가격·거래량 지표**: 수익률, 변동폭, VWAP, ATR, 이동평균선(5·20·60), 거래량 평균·급증률.
  - **위치 지표**: VWAP 대비 %, 이평선 대비 %.
  - **모멘텀 지표**: 갭(Gap) %, 최근 N기간 강세일 수.

### 5\. 하드 필터 (Pass/Fail)

  - **유동성 필터**: `turnover ≥ min_turnover`
  - **변동폭 필터**: `min_range_pct ≤ range_pct ≤ max_range_pct`
  - **추세 필터 (MA 정배열)**: `SMA5 > SMA20 > SMA60`
  - **가격 위치 필터 (VWAP)**: `pos_vs_vwap ≥ min_pos_vs_vwap`
  - **거래량 필터 (급증)**: `vol_surge ≥ min_vol_surge`
  - **리스크 필터**: 관리/거래정지/환기/상폐 경고 등 `meta` 플래그 기반 제외.
  - **선택 필터**: 갭 모멘텀, 최근 N일 강도 조건.

### 6\. 점수화 (Scoring)

  - **방식**: 각 지표를 0\~1로 정규화(Normalization) 후 가중합(Weighted Sum).
  - **가중 항목**: 유동성, 변동폭 적합도, MA 정배열 강도, VWAP 편차, 거래량 급증 강도, 갭 모멘텀, 최근 강도, ATR 정규화 값.
  - **명세**: 가중치 및 정규화 공식은 설정 파일에 명시.

### 7\. 출력 규격

  - **기본 컬럼**: `symbol`, `timestamp`, `pass` (bool), `score` (float), 주요 지표 값.

  - **리스크 컬럼 (추가)**: `entry_hint`, `stop_ref_atr`, `tp1_ref_atr`, `shares_suggested`, `trade_risk_amt`.

  - **포맷**: JSON 직렬화.

  - **출력 JSON 예시**:

    ```json
    {
      "screen_results": [
        {
          "symbol": "005930",
          "timestamp": "2025-08-12T10:30:00Z",
          "pass": true,
          "score": 0.89,
          "indicators": {
            "close": 85000,
            "volume_surge_ratio": 3.5,
            "pos_vs_vwap_pct": 0.015,
            "atr": 1250
          },
          "risk_profile": {
            "entry_hint": 85000,
            "stop_ref_atr": 83750,
            "tp1_ref_atr": 86870,
            "shares_suggested": 56,
            "trade_risk_amt": 69800
          }
        },
        {
          "symbol": "035720",
          "timestamp": "2025-08-12T10:30:00Z",
          "pass": true,
          "score": 0.85,
          "indicators": {
            "close": 120000,
            "volume_surge_ratio": 2.8,
            "pos_vs_vwap_pct": 0.011,
            "atr": 2500
          },
          "risk_profile": {
            "entry_hint": 120000,
            "stop_ref_atr": 117500,
            "tp1_ref_atr": 123750,
            "shares_suggested": 28,
            "trade_risk_amt": 70000
          }
        }
      ]
    }
    ```

### 8\. 구성 파라미터 (Configuration)

  - **필터값**: `min_turnover`, `min_range_pct`, `max_range_pct`, `min_vol_surge`, `min_gap_up` 등.
  - **가중치**: `weights` 딕셔너리 형태로 점수 항목별 가중치 설정.
  - **지원 형식**: YAML 또는 JSON.

### 9\. 처리 파이프라인

1.  데이터 수집 및 검증 (Data Ingestion & Validation)
2.  특징 계산 (Feature Engineering)
3.  하드 필터 적용 (`pass` 여부 산출)
4.  점수 계산 (Scoring)
5.  랭킹 정렬 및 출력 (Ranking & Output)
6.  로그 저장 및 알림 (Logging & Notification)

### 10\. 전략 프로파일 (프리셋)

  - **보수형**: 낮은 변동폭, 낮은 거래량 급증 임계치.
  - **표준형**: 기본값.
  - **공격형**: 높은 변동폭, 높은 거래량 급증 임계치.
  - **시가갭 특화**: 갭 모멘텀 가중치 상향.

### 11\. 예외/엣지 케이스 처리

  - 거래량 0 또는 결측치 처리.
  - 상/하한가 근접 시 변동성 계산 페널티.
  - 가격 단절 이벤트(유무상증자, 배당, 액면분할) 보정.
  - 거래 정지 종목 `pass=false` 강제.

### 12\. 기본 백테스트 사양

  - **레이블링**: T+1 수익률 또는 R-multiple (`(매도가-매수가)/손절폭`).
  - **핵심 성능지표**: 승률, 손익비, 기대값, MDD, Profit Factor.
  - **검증**: 롤링 워크포워드 방식, 과최적화 방지 설계.
  - **연계**: 파라미터 최적화 기능과 연동.

### 13\. 모니터링 및 로깅

  - 단계별 계산 결과 및 소요 시간 로그.
  - `pass` 통과 종목의 변화 추적.
  - 데이터 지연 및 결측 발생 시 알림.
  - 데이터 및 성능 드리프트 모니터링.

### 14\. API 인터페이스

  - **함수형**: `screen_universe(universe, config) → DataFrame`.
  - **REST API**: `POST /screener/run`, `GET /screener/config`.

#### 14.1. API 에러 응답 규격

> API 호출 실패 또는 로직 수행 중 예외 발생 시, 다음 표준 JSON 구조로 응답한다.

  - **400 Bad Request (잘못된 요청)**: 입력값 오류, 필수 파라미터 누락 등
    ```json
    {
      "error": {
        "code": 400,
        "type": "InvalidParameter",
        "message": "Required parameter 'symbols' is missing."
      }
    }
    ```
  - **500 Internal Server Error (서버 내부 오류)**: 특징 계산 실패, 데이터베이스 연결 오류 등
    ```json
    {
      "error": {
        "code": 500,
        "type": "CalculationError",
        "message": "Failed to calculate ATR for symbol '005930'. See logs for details.",
        "trace_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
      }
    }
    ```
  - **503 Service Unavailable (서비스 사용 불가)**: 외부 데이터 소스(증권사 API) 장애
    ```json
    {
      "error": {
        "code": 503,
        "type": "DataSourceUnavailable",
        "message": "Upstream stock data API is currently unavailable. Please try again later."
      }
    }
    ```

### 15\. 기본 수용 기준

  - **결정론**: 동일 입력 → 동일 결과 보장.
  - **랭킹 규칙**: `pass=false`인 종목은 랭킹에서 제외.
  - **설정 반영**: 구성(config) 변경 시 다음 평가 주기부터 반영.
  - **테스트**: 모든 경계값에 대한 단위 테스트(Unit Test) 필수.

### 16\. 구현 체크리스트

  - [ ] 데이터 정합성 검증 (타임존, 정렬, 결측치)
  - [ ] 롤링 지표 미래 데이터 누수 방지 (`shift`, `min_periods` 등)
  - [ ] 하드 필터 로직 함수화 및 단위 테스트
  - [ ] 점수화 로직 함수화 및 단위 테스트
  - [ ] 설정값 외부화 (YAML/JSON 로더)
  - [ ] 로깅 및 모니터링 모듈 연동
  - [ ] 프리셋 변경 실시간 반영 검증
  - [ ] 백테스트와 실시간 파이프라인 호환성 검증
  - [ ] API 엔드포인트 호출 및 응답 테스트 (성공/실패 케이스 모두)
  - [ ] 멀티타임프레임 데이터 병합 처리 로직
  - [ ] 뉴스/이슈 데이터 파싱 및 종목 매칭 테스트
  - [ ] 파라미터 탐색 러너(Grid/Random/Bayesian) 구현 및 워크포워드 통합
  - [ ] 백테스트 목적함수/제약조건 테스트(경계값)
  - [ ] 포지션 사이징 로직 단위 테스트(틱 사이즈·수수료 포함)

### 17\. 확장 아이디어

  - 뉴스 키워드·감성 점수를 점수화 항목에 추가.
  - 시장 국면(상승/하락/횡보) 자동 감지 후 프리셋 자동 전환.
  - 규칙 스코어와 ML 모델 예측 확률을 결합하는 하이브리드 모델.
  - 호가창 데이터(매수/매도 잔량) 및 체결강도 기반 추가 지표.
  - 멀티타임프레임 조건을 복합적으로 적용 (예: 일봉 추세 + 분봉 신호).
  - 테마·섹터 순환 분석을 통한 섹터별 가중치 부여.
  - VIX, 공포·탐욕 지수 등 시장 심리 지표 반영.

-----

## Part 2. 고급 모듈 및 상세 설계

### 18\. 파라미터 최적화 상세 설계

  - **18.1. 목적**: `min_turnover`, `weights.*` 등의 임계값/가중치를 데이터 기반으로 탐색하여 **수익-리스크 균형을 최적화**.
  - **18.2. 탐색 기법**:
      - **Grid Search**: 소규모 범위, 재현성 높음(초기 검증용).
      - **Random Search**: 고차원 매개변수에 효율적(기본 권장).
      - **베이지안 최적화 (TPE/GP)**: 탐색 효율 최상 (e.g., Optuna 라이브러리 활용).
      - **(선택) 진화 전략/유전 알고리즘**: 비선형·불연속 공간 탐색.
  - **18.3. 목적함수(Objective Function)**:
      - **기본**: `정규화 기대값 = ProfitFactor × (1 - MDD_norm)` 최대화
          - `ProfitFactor = 총이익 / 총손실`
          - `MDD_norm = MDD / 목표MDD` (예: 목표 15% -\> 0.15)
      - **제약 조건(Constraints)**:
          - `승률 ≥ 45%`
          - `거래수 ≥ 300 (전체 테스트 기간)`
          - `월별 손익 음수 비율 ≤ 30%`
  - **18.4. 검증 절차 (Walk-Forward Validation)**:
    1.  **시계열 분할**: (Train:Valid) = (6개월:1개월) 구간을 롤링(Rolling).
    2.  각 분할에서 최적 파라미터 탐색 후, 검증 구간(Valid)의 성과 기록.
    3.  전체 구간의 **평균 성능**과 \*\*성능의 표준편차(견고성)\*\*를 동시 고려하여 최종 파라미터 선정.
  - **18.5. 과최적화 방지**:
      - **K-Fold 시계열 교차검증** (데이터를 섞지 않음).
      - **얼리 스톱(Early Stopping)**: N회 연속 성능 개선이 없으면 탐색 중단.
      - **복잡도 페널티**: 파라미터 복잡도(사용된 피처 수 등) 증가 시 목적함수에 페널티 부여.
      - **갱신 주기 제한**: 월 1회 또는 분기 1회만 파라미터 갱신.
  - **18.6. 설정 예시 (YAML)**:
    ```yaml
    opt:
      algo: bayesian # grid|random|bayesian
      trials: 200
      objective: "profit_factor * (1 - mdd/0.15)"
      constraints:
        min_win_rate: 0.45
        min_trades: 300
        max_negative_month_ratio: 0.30
      walkforward:
        train_months: 6
        valid_months: 1
        rolls: 8
      early_stop_patience: 20
      regularization:
        complexity_penalty: 0.02 # 항목 수 x 0.02 감점
    ```
  - **18.7. 최적화 파라미터 상세 설명**:

| 파라미터 | 설명 | 단위 / 타입 | 예시 |
| :--- | :--- | :--- | :--- |
| `algo` | 사용할 최적화 알고리즘 | `string` | `bayesian` |
| `trials` | 최적화 시도 횟수 | `integer` | `200` |
| `objective` | 최대화할 목적함수 공식 | `string` (수식) | `"profit_factor * (1 - mdd/0.15)"` |
| `constraints.min_win_rate` | 최소 승률 제약 조건 | `float` (비율) | `0.45` |
| `constraints.min_trades` | 최소 거래 횟수 제약 조건 | `integer` | `300` |
| `constraints.max_negative_month_ratio`| 월별 손실 최대 비율 제약 조건 | `float` (비율) | `0.30` |
| `walkforward.train_months` | 워크포워드 학습 구간 길이 | `integer` (월) | `6` |
| `walkforward.valid_months` | 워크포워드 검증 구간 길이 | `integer` (월) | `1` |
| `walkforward.rolls` | 워크포워드 롤링 횟수 | `integer` | `8` |
| `early_stop_patience` | N회 연속 성능 개선 없을 시 조기 종료 | `integer` | `20` |
| `regularization.complexity_penalty` | 모델 복잡도에 부여하는 페널티 계수 | `float` | `0.02` |

### 19\. 리스크 및 포지션 사이징 상세 설계

  - **19.1. 계좌 리스크 원칙**:
      - **1회 거래 리스크**: `account_risk_pct = 0.5% ~ 1.0%` (기본 0.7%)
      - **최대 동시 보유 종목 수**: `max_concurrent_positions = 5`
      - **섹터/테마 집중도 제한**: 단일 섹터 당 `≤ 40%` 자본 할당.
  - **19.2. 진입/손절/익절 규칙**:
      - **진입(Entry)**: 스크리너 로직에 따라 결정.
      - **손절(Stop-Loss)**: `stop = entry - k_ATR × ATR(14)` (기본 `k_ATR=1.0`)
      - **1차 익절(Take-Profit)**: `tp1 = entry + 1.5 × ATR(14)`
      - **트레일링 스탑(Trailing Stop)**: `ATR(14)` 기반으로 고점 대비 하락 시 청산.
  - **19.3. 포지션 사이징 공식**:
      - `주당 예상 손실액 = entry - stop`
      - `총 허용 손실액 = account_equity × account_risk_pct`
      - **최종 매수 수량**: `shares = floor( 총 허용 손실액 / (주당 예상 손실액 + 거래비용) )`
  - **19.4. 포트폴리오 레벨 제어**:
      - **일간 손실 제한 (DDL)**: 당일 손실 -2% 도달 시 신규 매매 중단(쿨다운).
      - **주간 손실 제한 (WDL)**: 주간 손실 -5% 도달 시 남은 주간은 '보수형' 프리셋 강제.
      - **상관관계 제어**: 동일 테마 내 고상관(ρ≥0.7) 종목 동시 보유 금지.
  - **19.5. 계산 예시: ATR 및 포지션 사이징**:
    1.  **ATR(14) 계산 예시**:

          - `True Range (TR)` = `max[(당일 고가 - 당일 저가), abs(당일 고가 - 전일 종가), abs(당일 저가 - 전일 종가)]`
          - `ATR(14)` = `(이전 ATR(14) × 13 + 현재 TR) / 14` (EMA 방식)
          - **상황**: 전일 종가 10,000원, 당일 시가 10,200원, 고가 10,800원, 저가 10,100원
          - `TR` = `max[(10800-10100), abs(10800-10000), abs(10100-10000)]` = `max[700, 800, 100]` = `800`
          - 이전 `ATR(14)`가 650이었다면, 현재 `ATR(14)`는 `(650 * 13 + 800) / 14` ≈ `660.7`원이 됩니다.

    2.  **포지션 사이징 계산 예시**:

          - **계좌 정보**: 계좌 평가금액(`account_equity`) 10,000,000원, 1회 허용 리스크(`account_risk_pct`) 1.0%
          - **종목 정보**: 진입가(`entry`) 20,000원, 계산된 `ATR` 660원, 손절계수(`k_ATR`) 1.5
          - **계산**:
              - `총 허용 손실액` = 10,000,000원 × 1.0% = **100,000원**
              - `손절가` = 20,000원 - (1.5 × 660원) = **19,010원**
              - `주당 예상 손실액` = 20,000원 - 19,010원 = **990원**
              - `최종 매수 수량` = floor(100,000원 / 990원) = **101주**
              - (거래비용을 주당 10원으로 가정 시) `수량` = floor(100,000원 / (990원 + 10원)) = **100주**

### 20\. ML 보조 예측 상세 설계

  - **20.1. 목적**: 규칙 기반 신호에 **확률적 근거**를 더해 **랭킹 정밀도** 향상.
  - **20.2. 레이블(Label) 정의**:
      - `y = 1` if `T+1`일 고가가 진입가 대비 `+x%` 도달 (예: `+0.7%`).
      - (간소화) `y = 1` if `T+1`일 종가 \> `T`일 종가.
  - **20.3. 피처(Feature) 정의 (Look-ahead Bias 방지)**:
      - **가격**: `ret_1`, `range_pct`, `pos_vs_vwap`, `pos_vs_sma5/20/60`.
      - **거래량**: `vol_surge`, `volume_zscore(20)`.
      - **변동성**: `ATR_n`.
      - **패턴**: `gap_pct`, `recent_strength(5)`.
      - **(선택)** 섹터 강도, 시장 지수 대비 상대강도.
  - **20.4. 모델 선정**:
      - **1단계 (기본)**: **Logistic Regression** (설명력 높고, 안정적이며, 빠름).
      - **2단계 (선택)**: **LightGBM/XGBoost** (비선형 관계 포착, 클래스 가중치로 불균형 데이터 처리).
      - **확률 보정 (Calibration)**: Platt Scaling 또는 Isotonic Regression.
  - **20.5. 점수 결합**:
      - `final_score = α × rule_score + (1-α) × p_up` (기본 `α=0.7`).
      - **게이팅(Gating)**: `p_up < 0.45` 이면 `pass=false`로 강제 탈락시키는 옵션.
  - **20.6. 학습 및 배포 (MLOps)**:
      - **학습 주기**: **주 1회** 또는 월 1회 재학습.
      - **버전 관리**: 피처, 데이터, 모델의 버전을 해시값으로 기록하여 재현성 확보.
      - **모니터링**: **데이터 드리프트(PSI)**, **성능 드리프트(AUC)** 감지 시 알림.

### 21\. 백테스트 및 최적화 모듈 통합 설계

  - **21.1. 목표**: 규칙/가중치/리스크 파라미터의 **성과 검증** 및 **자동 최적화**.
  - **21.2. 데이터 요구사항**:
      - **시세 데이터**: 분봉 포함 권장, 수정주가 사용 필수.
      - **거래 비용**: 수수료, 세금, 슬리피지(Slippage)를 bp 단위로 설정.
      - **이벤트 데이터**: 거래정지, 상하한가, 분할/병합 캘린더 반영 시 정확도 향상.
  - **21.3. 시뮬레이션 엔진 사양**:
      - **포트폴리오 단위** 백테스트 (동시 N개 포지션 보유 및 평가).
      - **주문 체결 모델**:
          - 진입/청산은 해당 봉의 고가/저가 범위 내에서 체결 가정.
          - 슬리피지: `체결가 = 목표가 × (1 ± slippage_bps / 10000)` 적용.
          - KRX 가격 단위(틱 사이즈)에 맞춰 가격 라운딩.
      - **쿨다운**: DDL/WDL 등 리스크 제한 발동 시 전략 일시 정지.
  - **21.4. 리포팅 및 재현성**:
      - **결과물 (Artifacts)**:
          - `params.yaml` (최적 파라미터)
          - `metrics.json` (핵심 성과 지표)
          - `equity_curve.csv` (자산 곡선), `trades.csv` (거래 내역)
      - **버전 관리**: `데이터 해시 + 코드 커밋 해시 + 설정 파일 해시`를 함께 기록하여 완전한 재현성 보장.
  - **21.5. 워크플로 다이어그램**:
    ```mermaid
    graph TD
        A[시계열 데이터] --> B{워크포워드 분할\n(Train / Validation)};
        B --> C[Train 데이터];
        B --> D[Validation 데이터];
        
        subgraph 최적화 루프 (Optimization Loop on Train Data)
            C --> E{1. 파라미터 후보군 생성};
            E --> F[2. 백테스트 실행];
            F --> G{3. 목적함수 평가};
            G --> H{4. 성능 개선?};
            H -- No --> I[종료];
            H -- Yes --> E;
        end

        I --> J[최적 파라미터 선정];
        J --> K[Final Backtest];
        D --> K;
        K --> L[성과 리포트];
        
        style F fill:#f9f,stroke:#333,stroke-width:2px
        style K fill:#ccf,stroke:#333,stroke-width:2px
    ```

### 22\. 포지션 사이징 계산 로직

  - **22.1. 입력/출력 인터페이스**:
    ```yaml
    input:
      account_equity: 100000000 # 계좌 평가금액 (KRW)
      account_risk_pct: 0.007    # 1회 허용 리스크 (0.7%)
      entry_price: 52300       # 진입가
      atr: 1100                # ATR 값 (가격 단위)
      k_atr_stop: 1.0            # 손절 ATR 계수
      fee_bps: 7                 # 왕복 수수료+세금 (bp)
      slippage_bps: 5            # 왕복 슬리피지 (bp)
    output:
      shares: 123                # 추천 매수 수량
      stop_price: 51200            # 계산된 손절가
      tp1_price: 53950           # 계산된 1차 익절가
      trade_risk_amt: 700000     # 이번 거래의 최대 손실 예상액 (KRW)
    ```
  - **22.2. 계산 공식 상세**:
    1.  **손절/익절가 계산**:
          - `stop = round_to_tick(entry - k_atr_stop * atr)`
          - `tp1 = round_to_tick(entry + 1.5 * atr)`
    2.  **허용 손실금액 계산**:
          - `risk_capital = account_equity * account_risk_pct`
    3.  **주당 예상 손실 (비용 포함)**:
          - `cost_per_share = entry * (fee_bps + slippage_bps) / 10000`
          - `per_share_loss = (entry - stop) + cost_per_share`
    4.  **수량 계산**:
          - `shares = floor( risk_capital / max(per_share_loss, ε) )`

### 23\. 멀티타임프레임 및 시장 국면 연동

  - **23.1. 멀티타임프레임 규칙**:
      - **분봉 (1/5/15분)** → 민감한 진입 신호 포착.
      - **일봉** → 장기 추세 검증.
      - **결합 규칙**: `최종 pass = (분봉 신호 발생) AND (일봉 추세 필터 통과)`
      - *예시: 분봉에서 VWAP 상향 돌파 + 거래량 급증, 동시에 일봉에서 5일선 \> 20일선 \> 60일선 정배열 상태.*
  - **23.2. 시장 국면(Regime) 감지**:
      - **판단 지표**: KOSPI/KOSDAQ 지수의 20일, 60일 이동평균선 기울기.
      - **국면 분류**: 상승장 / 횡보장 / 하락장.
      - **자동 대응**: 감지된 국면에 따라 최적화된 **전략 프로파일(프리셋) 자동 전환**.
          - *예시: 하락장 감지 시, '공격형' 프로파일을 '보수형'으로 자동 변경.*

### 24\. 경량 ML 보조 예측 운영 가이드

  - **24.1. 해석 가능성 (XAI)**:
      - **1순위**: Logistic Regression의 계수(coefficients) 분석.
      - **2순위**: LightGBM의 `SHAP` 또는 `Feature Importance` 플롯을 통해 모델의 주요 판단 근거 시각화.
  - **24.2. 드리프트 대응**:
      - **입력 데이터 분포(PSI)** 또는 **예측 확률 분포**가 임계치 이상으로 변동 시 알림.
      - **대응**:
        1.  점수 결합 가중치 `α`를 상향하여 규칙 기반 점수의 비중을 높임 (안정성 확보).
        2.  모델 재학습 파이프라인을 트리거.

### 25\. 운영 대시보드 요구사항

  - **실시간 현황**:
      - 상위 N개 추천 종목 리스트 (`score`, `pass` 여부).
      - 현재 시장 국면 (상승/횡보/하락).
      - 실시간 체결강도 상위 요약.
  - **리스크 관리**:
      - 현재 총 노출 금액 (Total Exposure).
      - 섹터별 노출 비중.
      - 일간/주간 손익 현황 및 DDL/WDL 발동 여부.
  - **시스템 품질**:
      - API 데이터 피드 지연 시간.
      - 데이터 결측률.
      - 실측 슬리피지 (실제 체결가 vs 목표가).
  - **ML 모델 (선택)**:
      - 가장 영향력 있는 피처 Top 10.
      - 최근 데이터 드리프트 지표.
      - 최근 재학습 일자 및 성과 로그.

### 26\. 최종 수용 기준

  - **성과 기준**: 워크포워드 최적화 결과가 최근 12개월 백테스트에서 아래 기준을 모두 충족해야 함.
      - `Profit Factor ≥ 1.3`
      - `MDD ≤ 15%`
      - `월별 수익 음수 비율 ≤ 30%`
  - **실거래 검증**: 모의투자(Paper Trading) 샌드박스에서 4주간 운용 시, 백테스트 대비 **슬리피지 오차가 30% 이내**여야 함.
  - **감사 가능성**: 모든 파라미터 및 모델 갱신은 \*\*변경 로그(Changelog)와 버전(해시)\*\*으로 추적 및 감사가 가능해야 함.