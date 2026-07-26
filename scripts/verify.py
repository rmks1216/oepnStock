"""
검증 스크립트 (2단계)

이 프로젝트에서 문서에 적을 수 있는 숫자는 오직 이 스크립트가 출력한 숫자뿐이다.
다른 어떤 경로로 나온 수치도 문서에 사실로 적지 않는다.

하는 일:
    1. 실제 KRX 데이터를 쓰는지 먼저 확인한다 (Mock 이면 즉시 중단).
    2. 백테스트 시작 시점 기준으로 종목을 골라 생존편향을 줄인다.
    3. 지정한 전략 프로파일별로 백테스트를 돌린다.
    4. 수익률/샤프/최대낙폭/승률/비용을 표로 출력한다.
    5. 같은 내용을 reports/ 아래 마크다운 파일로 남긴다.

실행 방법:
    python scripts/verify.py
    python scripts/verify.py --profiles conservative swing
    python scripts/verify.py --start 2022-01-01 --end 2023-12-31

주의:
    이 스크립트를 돌리기 전에 scripts/check_data.py 가 전부 통과해야 한다.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

# 프로젝트 루트를 import 경로에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from oepnstock.utils.krx_data_source import KRXDataProvider  # noqa: E402

DEFAULT_PROFILES = ["default", "aggressive", "conservative", "swing"]
DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2023, 12, 31)
DEFAULT_UNIVERSE_SIZE = 20

REPORT_DIR = ROOT / "reports"


# ----------------------------------------------------------------------
# 안전장치
# ----------------------------------------------------------------------

def assert_real_data() -> KRXDataProvider:
    """실제 데이터인지 확인한다. 아니면 즉시 중단한다.

    이 프로젝트가 과거에 랜덤워크 데이터로 만든 수치를 문서에 사실처럼 적었다.
    같은 일이 반복되지 않도록, 여기서 실패하면 검증 자체를 진행하지 않는다.

    Returns:
        연결이 확인된 KRXDataProvider.

    Raises:
        SystemExit: KRX 데이터를 받을 수 없을 때.
    """
    provider = KRXDataProvider()
    if not provider.check_connection():
        print("중단: KRX 실데이터를 받을 수 없습니다.")
        print("      먼저 python scripts/check_data.py 를 실행해 원인을 확인하세요.")
        raise SystemExit(1)
    return provider


def build_universe(
    provider: KRXDataProvider, as_of: date, size: int
) -> List[str]:
    """백테스트 시작 시점 기준으로 종목을 고른다.

    오늘 기준 종목 리스트를 쓰면 그 사이 상장폐지된 종목이 빠져서
    수익률이 실제보다 높게 나온다(생존편향). 시작 시점 리스트를 쓰면 그 왜곡이 줄어든다.

    Args:
        provider: 데이터 제공자.
        as_of: 백테스트 시작일.
        size: 뽑을 종목 수.

    Returns:
        종목코드 리스트.
    """
    tickers = provider.get_ticker_list_at(as_of, market="KOSPI")
    if not tickers:
        print("경고: 과거 종목 리스트를 받지 못해 기본 종목으로 대체합니다.")
        return ["005930", "000660", "035420", "005380", "051910"][:size]

    universe = sorted(tickers)[:size]
    print(f"  종목 유니버스: {as_of} 기준 KOSPI 상장 종목 중 {len(universe)}개")
    return universe


# ----------------------------------------------------------------------
# 백테스트 실행
# ----------------------------------------------------------------------

def run_one_profile(
    profile: str, symbols: List[str], start: date, end: date
) -> Dict[str, Any]:
    """전략 프로파일 하나를 백테스트한다.

    Args:
        profile: 프로파일 이름. 예) 'conservative'
        symbols: 대상 종목 리스트.
        start: 시작일.
        end: 종료일.

    Returns:
        결과 사전. 실패하면 'error' 키가 들어 있다.
    """
    try:
        from examples.backtesting_example import SimpleBacktester
        from oepnstock.utils.config import ConfigManager

        config_manager = ConfigManager()
        custom_config = config_manager.apply_backtest_profile(profile)

        backtester = SimpleBacktester(custom_config=custom_config)
        result = backtester.run_backtest(
            symbols=symbols, start_date=start, end_date=end
        )

        if not isinstance(result, dict):
            return {"error": f"결과 형식이 사전이 아닙니다: {type(result)}"}
        return result

    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}


def pick(result: Dict[str, Any], key: str) -> Any:
    """결과에서 지표를 꺼낸다. 없으면 None 을 돌려준다.

    지표가 없으면 '측정 안 됨'으로 표시해야지, 0 으로 채우면 안 된다.
    0 으로 채우면 측정하지 않은 것과 측정해서 0 이 나온 것을 구별할 수 없다.
    """
    value = result.get(key)
    return value if isinstance(value, (int, float)) else None


def fmt_pct(value: Any) -> str:
    return "측정 안 됨" if value is None else f"{value:.2%}"


def fmt_num(value: Any) -> str:
    return "측정 안 됨" if value is None else f"{value:.2f}"


def fmt_int(value: Any) -> str:
    return "측정 안 됨" if value is None else f"{int(value)}"


# ----------------------------------------------------------------------
# 리포트 출력
# ----------------------------------------------------------------------

METRICS = [
    ("total_return", "총수익률", fmt_pct),
    ("annualized_return", "연환산수익률", fmt_pct),
    ("sharpe_ratio", "샤프비율", fmt_num),
    ("max_drawdown", "최대낙폭", fmt_pct),
    ("win_rate", "승률", fmt_pct),
    ("total_trades", "거래횟수", fmt_int),
    ("cost_ratio", "거래비용비율", fmt_pct),
]


def build_report(
    results: Dict[str, Dict[str, Any]],
    symbols: List[str],
    start: date,
    end: date,
) -> str:
    """검증 결과를 마크다운 문자열로 만든다."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: List[str] = []

    lines.append("# 백테스트 검증 리포트")
    lines.append("")
    lines.append(f"- 생성 시각: {now}")
    lines.append("- 데이터 출처: KRX 공식 데이터 (pykrx, 수정주가 적용)")
    lines.append(f"- 검증 기간: {start} ~ {end}")
    lines.append(f"- 대상 종목: {len(symbols)}개 ({start} 기준 상장 종목에서 선정)")
    lines.append("")
    lines.append("> 이 문서의 숫자만 다른 문서에 인용할 수 있다.")
    lines.append("> 여기 없는 수치는 근거가 없는 것이므로 문서에 적지 않는다.")
    lines.append("")

    ok = {k: v for k, v in results.items() if "error" not in v}
    ng = {k: v for k, v in results.items() if "error" in v}

    if ok:
        lines.append("## 전략별 성과")
        lines.append("")
        header = "| 지표 | " + " | ".join(ok.keys()) + " |"
        sep = "|---|" + "---|" * len(ok)
        lines.append(header)
        lines.append(sep)
        for key, label, formatter in METRICS:
            cells = [formatter(pick(r, key)) for r in ok.values()]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    if ng:
        lines.append("## 실행 실패한 프로파일")
        lines.append("")
        for name, r in ng.items():
            lines.append(f"- `{name}`: {r['error']}")
        lines.append("")

    lines.append("## 아직 측정하지 않은 것")
    lines.append("")
    lines.append("- Enhanced 모듈(Market Score / Enhanced Signal / Whipsaw)의 켬-끔 비교")
    lines.append("- Walk-Forward 분석 (과최적화 점검)")
    lines.append("- 상장폐지 종목을 포함한 전체 유니버스 백테스트")
    lines.append("")
    lines.append("위 항목은 측정 전까지 어떤 개선 효과도 주장하지 않는다.")
    lines.append("")

    return "\n".join(lines)


def print_console(results: Dict[str, Dict[str, Any]]) -> None:
    """콘솔에 결과 표를 출력한다."""
    ok = {k: v for k, v in results.items() if "error" not in v}

    print()
    print("=" * 70)
    print("검증 결과")
    print("=" * 70)

    if not ok:
        print("성공한 백테스트가 없습니다.")
    else:
        name_width = 16
        col_width = 14
        print("지표".ljust(name_width) + "".join(n.rjust(col_width) for n in ok))
        print("-" * (name_width + col_width * len(ok)))
        for key, label, formatter in METRICS:
            row = label.ljust(name_width)
            row += "".join(formatter(pick(r, key)).rjust(col_width) for r in ok.values())
            print(row)

    for name, r in results.items():
        if "error" in r:
            print()
            print(f"[실패] {name}: {r['error']}")


# ----------------------------------------------------------------------
# 진입점
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="실데이터 기반 전략 검증")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES,
                        help="검증할 전략 프로파일 이름들")
    parser.add_argument("--start", type=str, default=DEFAULT_START.isoformat(),
                        help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=DEFAULT_END.isoformat(),
                        help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--universe", type=int, default=DEFAULT_UNIVERSE_SIZE,
                        help="대상 종목 수")
    return parser.parse_args()


def main() -> int:
    """검증을 수행하고 리포트를 저장한다.

    Returns:
        성공한 백테스트가 하나라도 있으면 0, 없으면 1.
    """
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print("=" * 70)
    print("실데이터 검증 시작")
    print("=" * 70)

    provider = assert_real_data()
    print("  데이터 출처: KRX 공식 데이터 (수정주가 적용)")

    symbols = build_universe(provider, start, args.universe)
    print(f"  검증 기간: {start} ~ {end}")
    print(f"  대상 프로파일: {', '.join(args.profiles)}")

    results: Dict[str, Dict[str, Any]] = {}
    for profile in args.profiles:
        print()
        print(f"  실행 중: {profile} ...")
        results[profile] = run_one_profile(profile, symbols, start, end)

    print_console(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORT_DIR / f"verification_{stamp}.md"
    report_path.write_text(build_report(results, symbols, start, end), encoding="utf-8")

    print()
    print(f"리포트 저장: {report_path}")
    print("이 파일에 있는 숫자만 다른 문서에 인용하세요.")

    return 0 if any("error" not in r for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
