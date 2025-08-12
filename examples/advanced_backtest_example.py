"""
고급 백테스트 시스템 사용 예제
Walk-Forward Analysis, 몬테카를로 시뮬레이션, 성과 비교
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from oepnstock.backtest.advanced_backtester import AdvancedBacktester
from oepnstock.backtest.walk_forward_analyzer import WalkForwardAnalyzer
from oepnstock.backtest.monte_carlo_simulator import MonteCarloSimulator
from oepnstock.backtest.performance_metrics import PerformanceMetrics

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockStrategy:
    """테스트용 가상 전략"""
    
    def __init__(self, lookback_period: int = 20, 
                 entry_threshold: float = 0.02,
                 exit_threshold: float = 0.01,
                 position_size: float = 0.1):
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """가상 신호 생성"""
        # 단순한 모멘텀 전략
        returns = data['close'].pct_change(self.lookback_period)
        
        signals = pd.Series(0, index=data.index)
        signals[returns > self.entry_threshold] = 1  # 매수
        signals[returns < -self.exit_threshold] = -1  # 매도
        
        return signals


class AdvancedBacktestDemo:
    """고급 백테스트 데모"""
    
    def __init__(self):
        self.backtester = AdvancedBacktester()
        self.walk_forward = WalkForwardAnalyzer(optimization_metric='sharpe_ratio')
        self.monte_carlo = MonteCarloSimulator(random_seed=42)
    
    def run_comprehensive_demo(self):
        """종합 백테스트 데모 실행"""
        logger.info("=== 고급 백테스트 시스템 데모 시작 ===")
        
        # 1. 기본 데이터 생성
        market_data = self._generate_sample_data()
        
        # 2. 자본금별 백테스트
        logger.info("\n1. 자본금별 종합 백테스트")
        self.demo_capital_based_backtest(market_data)
        
        # 3. Walk-Forward Analysis
        logger.info("\n2. Walk-Forward Analysis")
        self.demo_walk_forward_analysis(market_data)
        
        # 4. 몬테카를로 시뮬레이션
        logger.info("\n3. 몬테카를로 시뮬레이션")
        self.demo_monte_carlo_simulation(market_data)
        
        # 5. 성과 지표 비교
        logger.info("\n4. 성과 지표 종합 분석")
        self.demo_performance_metrics(market_data)
        
        logger.info("\n=== 고급 백테스트 시스템 데모 완료 ===")
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """샘플 데이터 생성"""
        logger.info("샘플 시장 데이터 생성 중...")
        
        # 2년간 일일 데이터
        dates = pd.date_range('2022-01-01', '2024-01-01', freq='D')
        dates = dates[dates.weekday < 5]  # 주말 제외
        
        np.random.seed(42)
        n_days = len(dates)
        
        # 시장 데이터 생성
        base_price = 50000
        returns = np.random.normal(0.0005, 0.02, n_days)  # 일일 0.05% 평균, 2% 변동성
        
        # 추세와 주기성 추가
        trend = np.linspace(0, 0.1, n_days)  # 상승 추세
        seasonality = 0.02 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # 연간 주기
        
        adjusted_returns = returns + trend / 252 + seasonality / 252
        
        # 가격 시계열 생성
        prices = [base_price]
        for ret in adjusted_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # 다중 종목 데이터
        symbols = ['005930', '000660', '035420', '051910', '068270']
        data_list = []
        
        for symbol in symbols:
            symbol_multiplier = np.random.uniform(0.8, 1.2)  # 종목별 가격 차이
            symbol_prices = [p * symbol_multiplier for p in prices]
            
            for i, date in enumerate(dates):
                if i < len(symbol_prices):
                    price = symbol_prices[i]
                    data_list.append({
                        'date': date,
                        'symbol': symbol,
                        'open': price * (1 + np.random.normal(0, 0.005)),
                        'high': price * (1 + abs(np.random.normal(0, 0.01))),
                        'low': price * (1 - abs(np.random.normal(0, 0.01))),
                        'close': price,
                        'volume': np.random.randint(100000, 1000000),
                        'returns': adjusted_returns[i] if i < len(adjusted_returns) else 0
                    })
        
        return pd.DataFrame(data_list)
    
    def demo_capital_based_backtest(self, market_data: pd.DataFrame):
        """자본금별 백테스트 데모"""
        logger.info("자본금별 종합 백테스트 실행 중...")
        
        strategy = MockStrategy()
        capital_levels = [1_000_000, 3_000_000, 5_000_000, 10_000_000]
        
        results = self.backtester.run_comprehensive_backtest(strategy, capital_levels)
        
        print("\n📊 자본금별 백테스트 결과")
        print("=" * 50)
        
        for capital_range, result in results.items():
            print(f"\n{capital_range}:")
            print(f"  • 총 수익률: {result.total_return:.2%}")
            print(f"  • 연간 수익률: {result.annual_return:.2%}")
            print(f"  • 샤프 비율: {result.sharpe_ratio:.2f}")
            print(f"  • 최대 드로다운: {result.max_drawdown:.2%}")
            print(f"  • 총 거래: {result.total_trades}회")
            print(f"  • 승률: {result.win_rate:.1%}")
        
        # 결과 저장
        self.backtester.save_results(results, "capital_based_backtest.json")
    
    def demo_walk_forward_analysis(self, market_data: pd.DataFrame):
        """Walk-Forward Analysis 데모"""
        logger.info("Walk-Forward Analysis 실행 중...")
        
        # 파라미터 그리드
        param_grid = {
            'lookback_period': [10, 15, 20, 25, 30],
            'entry_threshold': [0.01, 0.015, 0.02, 0.025],
            'exit_threshold': [0.005, 0.01, 0.015],
            'position_size': [0.05, 0.1, 0.15]
        }
        
        try:
            wf_result = self.walk_forward.run_walk_forward_analysis(
                strategy_class=MockStrategy,
                data=market_data,
                param_grid=param_grid,
                train_window=252,  # 1년
                test_window=63,   # 3개월
                step_size=63,     # 3개월
                initial_capital=5_000_000
            )
            
            # 결과 출력
            print("\n🔄 Walk-Forward Analysis 결과")
            print("=" * 50)
            print(f"전략: {wf_result.strategy_name}")
            print(f"분석 기간: {len(wf_result.periods)}개 기간")
            print(f"전체 수익률: {wf_result.overall_return:.2%}")
            print(f"전체 샤프 비율: {wf_result.overall_sharpe:.2f}")
            print(f"최대 드로다운: {wf_result.overall_max_drawdown:.2%}")
            print(f"평균 훈련 샤프: {wf_result.avg_train_sharpe:.2f}")
            print(f"평균 테스트 샤프: {wf_result.avg_test_sharpe:.2f}")
            print(f"성과 감소: {wf_result.performance_decay:.2f}")
            
            # 파라미터 안정성
            print(f"\n🔧 파라미터 안정성:")
            for param, stability in wf_result.param_stability.items():
                print(f"  • {param}: {stability:.2f}")
            
            # 리포트 생성
            report = self.walk_forward.generate_walk_forward_report(wf_result)
            print(f"\n{report}")
            
            # 결과 저장
            self.walk_forward.save_walk_forward_results(
                wf_result, "walk_forward_demo"
            )
            
        except Exception as e:
            logger.error(f"Walk-Forward Analysis 실패: {e}")
    
    def demo_monte_carlo_simulation(self, market_data: pd.DataFrame):
        """몬테카를로 시뮬레이션 데모"""
        logger.info("몬테카를로 시뮬레이션 실행 중...")
        
        strategy_params = {
            'lookback_period': 20,
            'entry_threshold': 0.02,
            'exit_threshold': 0.01,
            'position_size': 0.1
        }
        
        try:
            mc_result = self.monte_carlo.run_monte_carlo_simulation(
                strategy_class=MockStrategy,
                base_data=market_data,
                strategy_params=strategy_params,
                n_simulations=1000,
                initial_capital=5_000_000,
                simulation_days=252,
                scenario_types=['parametric_bootstrap', 'block_bootstrap', 'stress_test'],
                parallel=False  # 예제에서는 순차 실행
            )
            
            # 결과 출력
            print("\n🎲 몬테카를로 시뮬레이션 결과")
            print("=" * 50)
            print(f"시뮬레이션 횟수: {mc_result.n_simulations:,}회")
            print(f"평균 수익률: {mc_result.mean_return:.2%}")
            print(f"중간값 수익률: {mc_result.median_return:.2%}")
            print(f"표준편차: {mc_result.std_return:.2%}")
            print(f"손실 확률: {mc_result.probability_of_loss:.1%}")
            print(f"파산 확률: {mc_result.probability_of_ruin:.1%}")
            
            print(f"\n📈 분위수 분석:")
            print(f"  • 5%ile (최악): {mc_result.percentile_5:.2%}")
            print(f"  • 25%ile: {mc_result.percentile_25:.2%}")
            print(f"  • 75%ile: {mc_result.percentile_75:.2%}")
            print(f"  • 95%ile (최고): {mc_result.percentile_95:.2%}")
            
            print(f"\n🎯 극단 시나리오:")
            print(f"  • 최고 수익률: {mc_result.best_scenario.total_return:.2%}")
            print(f"  • 최악 수익률: {mc_result.worst_scenario.total_return:.2%}")
            
            # 리포트 생성
            report = self.monte_carlo.generate_monte_carlo_report(mc_result)
            print(f"\n{report}")
            
            # 결과 저장
            self.monte_carlo.save_monte_carlo_results(
                mc_result, "monte_carlo_demo"
            )
            
        except Exception as e:
            logger.error(f"몬테카를로 시뮬레이션 실패: {e}")
    
    def demo_performance_metrics(self, market_data: pd.DataFrame):
        """성과 지표 데모"""
        logger.info("성과 지표 계산 및 분석 중...")
        
        # 가상 수익률 시계열 생성
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0.0005, 0.02, 252),
            index=pd.date_range('2023-01-01', periods=252, freq='D')
        )
        
        # 벤치마크 수익률 (시장 지수)
        benchmark_returns = pd.Series(
            np.random.normal(0.0003, 0.015, 252),
            index=returns.index
        )
        
        # 종합 성과 지표 계산
        performance_report = PerformanceMetrics.calculate_comprehensive_metrics(
            returns, benchmark_returns
        )
        
        print("\n📈 종합 성과 분석")
        print("=" * 50)
        
        print(f"기본 수익률 지표:")
        print(f"  • 총 수익률: {performance_report.total_return:.2%}")
        print(f"  • 연간 수익률: {performance_report.annual_return:.2%}")
        print(f"  • 변동성: {performance_report.volatility:.2%}")
        
        print(f"\n리스크 조정 수익률:")
        print(f"  • 샤프 비율: {performance_report.sharpe_ratio:.2f}")
        print(f"  • 소르티노 비율: {performance_report.sortino_ratio:.2f}")
        print(f"  • 칼마 비율: {performance_report.calmar_ratio:.2f}")
        print(f"  • 오메가 비율: {performance_report.omega_ratio:.2f}")
        
        print(f"\n리스크 지표:")
        print(f"  • 최대 드로다운: {performance_report.max_drawdown:.2%}")
        print(f"  • VaR (95%): {performance_report.var_95:.2%}")
        print(f"  • CVaR (95%): {performance_report.cvar_95:.2%}")
        print(f"  • 꼬리 비율: {performance_report.tail_ratio:.2f}")
        
        print(f"\n벤치마크 비교:")
        print(f"  • 상관계수: {performance_report.market_correlation:.2f}")
        print(f"  • 베타: {performance_report.beta:.2f}")
        print(f"  • 알파: {performance_report.alpha:.2%}")
        print(f"  • 정보 비율: {performance_report.information_ratio:.2f}")
        
        # 롤링 성과 지표
        rolling_metrics = PerformanceMetrics.calculate_rolling_metrics(returns, window=63)
        if not rolling_metrics.empty:
            print(f"\n📊 롤링 성과 (최근):")
            latest = rolling_metrics.iloc[-1]
            print(f"  • 3개월 샤프: {latest['sharpe_ratio']:.2f}")
            print(f"  • 3개월 변동성: {latest['volatility']:.2%}")
            print(f"  • 3개월 최대DD: {latest['max_drawdown']:.2%}")
        
        # 성과 요약 리포트
        summary = PerformanceMetrics.generate_performance_summary(returns, benchmark_returns)
        print(f"\n{summary}")
        
        # 스트레스 테스트
        stress_scenarios = {
            'market_crash': -0.20,
            'volatility_spike': 0.50,
            'correlation_breakdown': 0.30
        }
        
        stress_results = PerformanceMetrics.stress_test_analysis(returns, stress_scenarios)
        print(f"\n⚠️ 스트레스 테스트:")
        for scenario, results in stress_results.items():
            print(f"  • {scenario}:")
            print(f"    - 수익률: {results['total_return']:.2%}")
            print(f"    - 샤프: {results['sharpe_ratio']:.2f}")
            print(f"    - 최대DD: {results['max_drawdown']:.2%}")


def main():
    """메인 실행 함수"""
    demo = AdvancedBacktestDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()