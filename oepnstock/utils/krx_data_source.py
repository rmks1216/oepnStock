"""
KRX Data Source
pykrx 기반 실제 한국거래소 데이터 소스.

이 모듈은 free_data_sources.py 의 MockDataProvider(랜덤워크)와
FreeDataProvider(yfinance)를 대체한다.

교체 이유:
    1. 랜덤워크 데이터에는 기술적 패턴이 존재하지 않으므로 전략 검증이 불가능하다.
    2. yfinance 는 한국 종목의 수정주가/거래정지/상장폐지 처리가 부정확하다.
    3. pykrx 는 KRX 공식 데이터를 쓰므로 수정주가가 정확하고,
       과거 시점의 상장 종목 리스트를 조회할 수 있어 생존편향을 줄일 수 있다.

사전 준비:
    1) pip install pykrx
    2) https://data.krx.co.kr 회원가입
    3) 환경변수 설정 (pykrx 1.2.8 이상 필수)
       Windows : setx KRX_ID "아이디"  /  setx KRX_PW "비밀번호"
       macOS   : export KRX_ID="아이디" / export KRX_PW="비밀번호"
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# pykrx 는 import 하는 순간 KRX_ID / KRX_PW 로 로그인한다.
# 그래서 .env 파일을 pykrx import 보다 먼저 읽어 환경변수에 올린다.
# 이미 셸에 설정된 환경변수가 있으면 그 값을 덮어쓰지 않는다.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

try:
    from pykrx import stock as krx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pykrx 가 설치되어 있지 않습니다. 먼저 `pip install pykrx` 를 실행하세요."
    ) from exc

from .logger import get_logger

logger = get_logger(__name__)


# pykrx 가 돌려주는 한글 컬럼명을 프로젝트 표준(소문자 영문)으로 바꾼다.
_OHLCV_COLUMN_MAP: Dict[str, str] = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
    "거래대금": "value",
    "등락률": "change_pct",
}

_INDEX_TICKER: Dict[str, str] = {
    "KOSPI": "1001",
    "KOSDAQ": "2001",
}

# get_data() 에 종목을 지정하지 않았을 때 쓰는 종목.
# 오늘 기준 대형주라 생존편향이 있다. 검증용 백테스트는 종목을 직접 넘긴다.
DEFAULT_SYMBOLS: List[str] = ["005930", "000660", "035420", "051910", "068270"]


class KRXDataProvider:
    """KRX 공식 데이터를 조회하고 로컬에 캐시하는 데이터 제공자.

    KRX 서버는 짧은 시간에 요청이 몰리면 응답을 끊는다.
    그래서 조회한 결과를 CSV 로 저장해 두고, 같은 요청이 다시 오면
    네트워크를 타지 않고 파일에서 읽는다.

    Attributes:
        cache_dir: 조회 결과를 저장할 디렉터리.
        request_interval: 연속 요청 사이에 쉬는 시간(초).
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        request_interval: float = 0.6,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_interval = request_interval
        self._last_request_at: float = 0.0

        if not os.getenv("KRX_ID") or not os.getenv("KRX_PW"):
            logger.warning(
                "KRX_ID / KRX_PW 환경변수가 없습니다. "
                "pykrx 1.2.8 이상은 KRX 로그인이 필요하므로 조회가 실패할 수 있습니다."
            )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """연속 요청 사이에 최소 간격을 둔다. KRX 차단을 막기 위함이다."""
        elapsed = time.time() - self._last_request_at
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request_at = time.time()

    @staticmethod
    def _to_krx_date(value: date | datetime | str) -> str:
        """날짜를 pykrx 가 받는 'YYYYMMDD' 문자열로 바꾼다."""
        if isinstance(value, str):
            return value.replace("-", "")
        return value.strftime("%Y%m%d")

    def _cache_path(self, kind: str, key: str, start: str, end: str) -> Path:
        return self.cache_dir / f"{kind}_{key}_{start}_{end}.csv"

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index.name = "date"
            logger.debug("캐시에서 읽음: %s (%d행)", path.name, len(df))
            return df
        except Exception as exc:
            logger.warning("캐시 파일을 읽지 못해 다시 조회합니다: %s (%s)", path.name, exc)
            return None

    def _write_cache(self, path: Path, df: pd.DataFrame) -> None:
        try:
            df.to_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            logger.warning("캐시 저장 실패: %s (%s)", path.name, exc)

    @staticmethod
    def _normalize(df: pd.DataFrame, drop_halted: bool) -> pd.DataFrame:
        """컬럼명을 표준화하고 거래정지일을 정리한다.

        Args:
            df: pykrx 가 돌려준 원본 데이터프레임.
            drop_halted: True 이면 거래량 0 인 날(거래정지 등)을 제거한다.
        """
        df = df.rename(columns=_OHLCV_COLUMN_MAP)
        df.index.name = "date"

        if drop_halted and "volume" in df.columns:
            halted = int((df["volume"] == 0).sum())
            if halted:
                logger.info("거래량 0인 %d일을 제외했습니다(거래정지 추정).", halted)
                df = df[df["volume"] > 0]

        return df.sort_index()

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def get_historical_data(
        self,
        symbol: str,
        start_date: date | str,
        end_date: date | str,
        use_cache: bool = True,
        drop_halted: bool = True,
    ) -> Optional[pd.DataFrame]:
        """개별 종목의 일봉 수정주가를 조회한다.

        Args:
            symbol: 6자리 종목코드. 예) '005930'
            start_date: 조회 시작일.
            end_date: 조회 종료일.
            use_cache: True 이면 캐시를 우선 사용한다.
            drop_halted: True 이면 거래량 0인 날을 제거한다.

        Returns:
            open/high/low/close/volume 컬럼을 가진 데이터프레임.
            조회에 실패하면 None.
        """
        start = self._to_krx_date(start_date)
        end = self._to_krx_date(end_date)
        path = self._cache_path("ohlcv", symbol, start, end)

        if use_cache:
            cached = self._read_cache(path)
            if cached is not None:
                return cached

        try:
            self._throttle()
            # adjusted=True 가 수정주가다. 액면분할/증자 왜곡을 막으려면 반드시 켠다.
            raw = krx.get_market_ohlcv_by_date(
                fromdate=start, todate=end, ticker=symbol, freq="d", adjusted=True
            )
        except Exception as exc:
            logger.error("종목 %s 조회 실패: %s", symbol, exc)
            return None

        if raw is None or raw.empty:
            logger.warning("종목 %s 데이터가 비어 있습니다(%s~%s).", symbol, start, end)
            return None

        df = self._normalize(raw, drop_halted)
        self._write_cache(path, df)
        logger.info("종목 %s: %d일치 조회 완료", symbol, len(df))
        return df

    def get_index_data(
        self,
        index_name: str,
        start_date: date | str,
        end_date: date | str,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """지수 일봉을 조회한다.

        Args:
            index_name: 'KOSPI' 또는 'KOSDAQ'.
            start_date: 조회 시작일.
            end_date: 조회 종료일.
            use_cache: True 이면 캐시를 우선 사용한다.

        Returns:
            open/high/low/close/volume 컬럼을 가진 데이터프레임. 실패 시 None.
        """
        ticker = _INDEX_TICKER.get(index_name.upper())
        if ticker is None:
            logger.error("지원하지 않는 지수입니다: %s", index_name)
            return None

        start = self._to_krx_date(start_date)
        end = self._to_krx_date(end_date)
        path = self._cache_path("index", index_name.upper(), start, end)

        if use_cache:
            cached = self._read_cache(path)
            if cached is not None:
                return cached

        try:
            self._throttle()
            raw = krx.get_index_ohlcv_by_date(fromdate=start, todate=end, ticker=ticker)
        except Exception as exc:
            logger.error("지수 %s 조회 실패: %s", index_name, exc)
            return None

        if raw is None or raw.empty:
            logger.warning("지수 %s 데이터가 비어 있습니다.", index_name)
            return None

        df = self._normalize(raw, drop_halted=False)
        self._write_cache(path, df)
        logger.info("지수 %s: %d일치 조회 완료", index_name, len(df))
        return df

    def get_ticker_list_at(
        self, as_of: date | str, market: str = "KOSPI"
    ) -> List[str]:
        """특정 과거 시점에 상장돼 있던 종목 코드를 돌려준다.

        생존편향을 줄이는 핵심 메서드다.
        오늘 기준 종목 리스트로 과거를 백테스트하면 그 사이 상장폐지된 종목이
        전부 빠지므로 수익률이 구조적으로 부풀려진다.
        백테스트 시작일 기준으로 이 메서드를 써서 종목을 고르면 그 왜곡이 줄어든다.

        Args:
            as_of: 기준 시점.
            market: 'KOSPI', 'KOSDAQ', 'ALL' 중 하나.

        Returns:
            종목코드 리스트. 실패 시 빈 리스트.
        """
        as_of_str = self._to_krx_date(as_of)
        try:
            self._throttle()
            tickers = krx.get_market_ticker_list(date=as_of_str, market=market)
        except Exception as exc:
            logger.error("%s 시점 %s 종목 리스트 조회 실패: %s", as_of_str, market, exc)
            return []

        logger.info("%s 기준 %s 상장 종목 %d개", as_of_str, market, len(tickers))
        return list(tickers)

    def get_delisted_between(
        self, start_date: date | str, end_date: date | str, market: str = "KOSPI"
    ) -> List[str]:
        """기간 중 사라진 종목 코드를 돌려준다.

        시작 시점에는 있었지만 종료 시점에는 없는 종목이 상장폐지 후보다.
        생존편향이 실제로 얼마나 큰지 눈으로 확인할 때 쓴다.

        Args:
            start_date: 기간 시작일.
            end_date: 기간 종료일.
            market: 'KOSPI', 'KOSDAQ', 'ALL' 중 하나.

        Returns:
            기간 중 사라진 종목코드 리스트.
        """
        before = set(self.get_ticker_list_at(start_date, market))
        after = set(self.get_ticker_list_at(end_date, market))
        gone = sorted(before - after)
        logger.info("기간 중 사라진 종목 %d개 (%s)", len(gone), market)
        return gone

    def get_market_data(
        self,
        symbols: List[str],
        start_date: date | str,
        end_date: date | str,
    ) -> Dict[str, pd.DataFrame]:
        """여러 종목을 한 번에 조회한다.

        Args:
            symbols: 종목코드 리스트.
            start_date: 조회 시작일.
            end_date: 조회 종료일.

        Returns:
            {종목코드: 데이터프레임} 형태의 사전. 실패한 종목은 빠진다.
        """
        result: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, start_date, end_date)
            if df is not None and not df.empty:
                result[symbol] = df
            else:
                logger.warning("종목 %s 를 건너뜁니다.", symbol)

        logger.info("요청 %d개 중 %d개 조회 성공", len(symbols), len(result))
        return result

    def get_recent_data(self, symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
        """오늘부터 거슬러 올라간 최근 일봉을 조회한다.

        페이퍼 트레이딩 예제가 쓰던 yfinance 의 period='1mo' 같은 조회를 대신한다.
        백테스트에서는 쓰지 않는다. 백테스트는 기준일이 과거이므로
        get_historical_data 에 기간을 명시해야 한다.

        Args:
            symbol: 6자리 종목코드.
            days: 오늘부터 거슬러 올라갈 달력 일수.

        Returns:
            일봉 데이터프레임. 실패 시 None.
        """
        end = date.today()
        start = end - timedelta(days=days)
        # 오늘 데이터는 장중에 바뀌므로 캐시를 쓰지 않는다.
        return self.get_historical_data(symbol, start, end, use_cache=False)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """가장 최근 영업일 종가를 돌려준다.

        주의: 실시간 체결가가 아니다. KRX 일봉 데이터라 장중에는
        당일 진행 중인 값이거나 전일 종가다. 실시간 시세가 필요하면
        증권사 API 를 연결해야 한다.

        Args:
            symbol: 6자리 종목코드.

        Returns:
            최근 종가. 실패 시 None.
        """
        df = self.get_recent_data(symbol, days=10)
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])

    def get_data(
        self,
        start_date: date | str,
        end_date: date | str,
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """여러 종목 일봉을 긴 형식(date, symbol, OHLCV 컬럼)으로 돌려준다.

        AdvancedBacktester 가 data_source.get_data(start, end) 로 부르는 형식에 맞춘 것이다.

        Args:
            start_date: 조회 시작일.
            end_date: 조회 종료일.
            symbols: 종목코드 리스트. None 이면 DEFAULT_SYMBOLS 를 쓴다.

        Returns:
            date, symbol, open, high, low, close, volume 컬럼의 데이터프레임.

        Raises:
            RuntimeError: 한 종목도 조회하지 못했을 때.
        """
        symbols = symbols or DEFAULT_SYMBOLS
        frames = []
        for symbol, df in self.get_market_data(symbols, start_date, end_date).items():
            part = df[["open", "high", "low", "close", "volume"]].copy()
            part["symbol"] = symbol
            frames.append(part.reset_index())

        if not frames:
            raise RuntimeError(
                f"{start_date}~{end_date} 기간에 조회된 종목이 없습니다: {symbols}"
            )

        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(["date", "symbol"]).reset_index(drop=True)

    def check_connection(self) -> bool:
        """KRX 조회가 실제로 되는지 확인한다.

        Returns:
            최근 영업일 KOSPI 지수를 받아오면 True.
        """
        end = date.today()
        start = end - timedelta(days=10)
        df = self.get_index_data("KOSPI", start, end, use_cache=False)
        ok = df is not None and not df.empty
        if ok:
            logger.info("KRX 연결 정상 (최근 종가 %.2f)", df["close"].iloc[-1])
        else:
            logger.error("KRX 연결 실패. KRX_ID / KRX_PW 환경변수를 확인하세요.")
        return ok


# 프로젝트 전역에서 쓰는 단일 인스턴스
krx_data = KRXDataProvider()


def get_data_provider() -> KRXDataProvider:
    """실데이터 제공자를 돌려준다.

    기존 free_data_sources.get_data_provider() 는 연결이 안 되면 조용히
    랜덤워크 Mock 으로 넘어갔다. 그 동작이 "검증된 것처럼 보이는 가짜 결과"를
    만들어 냈으므로, 여기서는 대체하지 않고 실패를 그대로 드러낸다.

    Returns:
        KRXDataProvider 인스턴스.

    Raises:
        RuntimeError: KRX 조회가 되지 않을 때.
    """
    if not krx_data.check_connection():
        raise RuntimeError(
            "KRX 데이터를 가져올 수 없습니다. 검증은 실데이터로만 수행합니다.\n"
            "확인할 것: (1) pip install pykrx  (2) KRX_ID / KRX_PW 환경변수  (3) 네트워크"
        )
    return krx_data
