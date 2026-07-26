"""
데이터 점검 스크립트 (1단계)

이 스크립트는 백테스트를 돌리기 전에 "데이터가 진짜인지" 먼저 확인한다.
아래 4가지를 순서대로 검사한다.

    1. KRX 접속이 되는가
    2. 개별 종목 수정주가를 받아오는가
    3. 과거 시점 종목 리스트가 나오는가 (생존편향 확인용)
    4. 기간 중 상장폐지된 종목이 몇 개인지

실행 방법:
    python scripts/check_data.py

이 스크립트가 전부 통과하기 전에는 백테스트 결과를 믿으면 안 된다.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oepnstock.utils.krx_data_source import KRXDataProvider  # noqa: E402

# 점검 대상 설정
TEST_SYMBOL = "005930"      # 삼성전자
TEST_SYMBOL_NAME = "삼성전자"
START = date(2022, 1, 1)
END = date(2023, 12, 31)


def line(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    """데이터 점검을 순서대로 수행하고 결과를 출력한다.

    Returns:
        모든 점검을 통과하면 0, 하나라도 실패하면 1.
    """
    provider = KRXDataProvider()
    failed: list[str] = []

    # ----------------------------------------------------------
    line("1. KRX 접속 확인")
    # ----------------------------------------------------------
    if provider.check_connection():
        print("통과: KRX 에서 지수 데이터를 받아왔습니다.")
    else:
        print("실패: KRX 에 접속하지 못했습니다.")
        print("  확인 1) pip install pykrx 가 됐는지")
        print("  확인 2) KRX_ID / KRX_PW 환경변수를 설정했는지")
        print("  확인 3) data.krx.co.kr 에 가입했는지")
        return 1

    # ----------------------------------------------------------
    line(f"2. 개별 종목 수정주가 확인 ({TEST_SYMBOL_NAME})")
    # ----------------------------------------------------------
    df = provider.get_historical_data(TEST_SYMBOL, START, END)
    if df is None or df.empty:
        print("실패: 종목 데이터를 받지 못했습니다.")
        failed.append("종목 조회")
    else:
        print(f"행 수        : {len(df)}일")
        print(f"기간         : {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"컬럼         : {list(df.columns)}")
        print(f"결측치       : {int(df.isna().sum().sum())}개")
        print(f"종가 범위    : {df['close'].min():,.0f} ~ {df['close'].max():,.0f}원")
        print()
        print("최근 3일:")
        print(df[["open", "high", "low", "close", "volume"]].tail(3).to_string())

        # 데이터 품질 자동 검사
        problems = []
        if df["close"].le(0).any():
            problems.append("종가가 0 이하인 날이 있다")
        if (df["high"] < df["low"]).any():
            problems.append("고가가 저가보다 낮은 날이 있다")
        if (df["close"] > df["high"]).any() or (df["close"] < df["low"]).any():
            problems.append("종가가 고가-저가 범위를 벗어난 날이 있다")

        # 하루에 30% 넘게 튀면 수정주가가 안 먹었을 가능성이 있다
        jump = df["close"].pct_change().abs()
        big_jump = int((jump > 0.30).sum())
        if big_jump:
            problems.append(f"일간 30% 초과 변동이 {big_jump}일 있다(수정주가 미적용 의심)")

        if problems:
            print()
            print("경고:")
            for p in problems:
                print(f"  - {p}")
            failed.append("데이터 품질")
        else:
            print()
            print("통과: 데이터 품질 검사 이상 없음.")

    # ----------------------------------------------------------
    line("3. 과거 시점 종목 리스트 확인")
    # ----------------------------------------------------------
    past = provider.get_ticker_list_at(START, market="KOSPI")
    today = provider.get_ticker_list_at(date.today(), market="KOSPI")

    if not past or not today:
        print("실패: 종목 리스트를 받지 못했습니다.")
        failed.append("종목 리스트")
    else:
        print(f"{START} 기준 KOSPI 상장 종목 : {len(past)}개")
        print(f"오늘 기준 KOSPI 상장 종목    : {len(today)}개")
        print("통과: 과거 시점 종목 리스트를 조회할 수 있습니다.")

    # ----------------------------------------------------------
    line("4. 생존편향 규모 확인")
    # ----------------------------------------------------------
    gone = provider.get_delisted_between(START, date.today(), market="KOSPI")
    if past:
        ratio = len(gone) / len(past) * 100
        print(f"{START} 이후 KOSPI 에서 사라진 종목: {len(gone)}개 ({ratio:.1f}%)")
        print()
        print("해석:")
        print(f"  오늘 종목 리스트로 과거를 백테스트하면 이 {len(gone)}개가 전부 빠진다.")
        print("  이 종목들은 대체로 성과가 나빠서 사라졌으므로,")
        print("  빼고 계산한 수익률은 실제보다 높게 나온다. 이것이 생존편향이다.")
        if gone:
            print(f"  예시: {', '.join(gone[:10])}")

    # ----------------------------------------------------------
    line("점검 결과")
    # ----------------------------------------------------------
    if failed:
        print("실패한 항목:", ", ".join(failed))
        print("→ 이 문제를 먼저 해결한 뒤 백테스트를 돌리세요.")
        return 1

    print("모든 점검 통과.")
    print("→ 다음 단계: python scripts/verify.py 를 실행하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
