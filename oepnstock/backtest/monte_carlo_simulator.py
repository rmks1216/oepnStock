"""
몬테카를로 시뮬레이션 엔진
다양한 시나리오 하에서 전략 성과 검증
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime
import concurrent.futures
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

from .performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloScenario:
    """몬테카를로 시나리오"""
    scenario_id: int
    returns: pd.Series
    final_capital: float
    max_drawdown: float
    sharpe_ratio: float
    total_return: float
    volatility: float
    var_95: float
    trade_count: int = 0
    max_consecutive_losses: int = 0


@dataclass
class MonteCarloResult:
    """몬테카를로 시뮬레이션 결과"""
    strategy_name: str
    n_simulations: int
    initial_capital: float
    
    # 시나리오 결과들
    scenarios: List[MonteCarloScenario]
    
    # 통계적 요약
    mean_return: float
    median_return: float
    std_return: float
    
    # 분위수 분석
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    
    # 리스크 지표
    probability_of_loss: float
    probability_of_ruin: float  # 50% 이상 손실 확률
    expected_shortfall_5: float
    
    # 성과 지표 분포
    sharpe_distribution: List[float]
    drawdown_distribution: List[float]
    
    # 최고/최악 시나리오
    best_scenario: MonteCarloScenario
    worst_scenario: MonteCarloScenario
    
    # 신뢰구간
    return_confidence_95: Tuple[float, float]
    sharpe_confidence_95: Tuple[float, float]


class MonteCarloSimulator:
    """몬테카를로 시뮬레이션 엔진"""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info("Monte Carlo Simulator initialized")
    
    def run_monte_carlo_simulation(self,
                                 strategy_class: type,
                                 base_data: pd.DataFrame,
                                 strategy_params: Dict[str, Any],
                                 n_simulations: int = 1000,
                                 initial_capital: float = 10_000_000,
                                 simulation_days: int = 252,
                                 scenario_types: List[str] = None,
                                 parallel: bool = True) -> MonteCarloResult:
        """몬테카를로 시뮬레이션 실행"""
        
        if scenario_types is None:
            scenario_types = ['parametric_bootstrap', 'block_bootstrap', 'stress_test']
        
        logger.info(f"Starting Monte Carlo simulation with {n_simulations} scenarios")
        logger.info(f"Scenario types: {scenario_types}")
        
        # 시나리오 생성
        scenarios_data = self._generate_scenarios(
            base_data, n_simulations, simulation_days, scenario_types
        )
        
        # 시뮬레이션 실행
        if parallel and n_simulations > 100:
            scenarios = self._run_parallel_simulation(
                strategy_class, scenarios_data, strategy_params, initial_capital
            )
        else:
            scenarios = self._run_sequential_simulation(
                strategy_class, scenarios_data, strategy_params, initial_capital
            )
        
        # 결과 분석
        return self._analyze_monte_carlo_results(
            strategy_class.__name__, scenarios, initial_capital, n_simulations
        )
    
    def _generate_scenarios(self,
                          base_data: pd.DataFrame,
                          n_simulations: int,
                          simulation_days: int,
                          scenario_types: List[str]) -> List[pd.DataFrame]:
        """다양한 시나리오 데이터 생성"""
        
        scenarios_data = []
        simulations_per_type = n_simulations // len(scenario_types)
        
        for scenario_type in scenario_types:
            type_scenarios = getattr(self, f'_generate_{scenario_type}_scenarios')(
                base_data, simulations_per_type, simulation_days
            )
            scenarios_data.extend(type_scenarios)
        
        # 부족한 경우 추가 생성
        while len(scenarios_data) < n_simulations:
            additional = self._generate_parametric_bootstrap_scenarios(
                base_data, 1, simulation_days
            )[0]
            scenarios_data.append(additional)
        
        return scenarios_data[:n_simulations]
    
    def _generate_parametric_bootstrap_scenarios(self,
                                               base_data: pd.DataFrame,
                                               n_scenarios: int,
                                               simulation_days: int) -> List[pd.DataFrame]:
        """모수적 부트스트랩 시나리오"""
        scenarios = []
        
        # 기본 통계량 추정
        if 'returns' in base_data.columns:
            returns = base_data['returns'].dropna()
        else:
            # 가격 데이터에서 수익률 계산
            returns = base_data['close'].pct_change().dropna()
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        for _ in range(n_scenarios):
            # 정규분포에서 수익률 샘플링
            simulated_returns = np.random.normal(
                mean_return, std_return, simulation_days
            )
            
            # 가격 시계열 생성
            base_price = 50000  # 기준 가격
            prices = [base_price]
            
            for ret in simulated_returns:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            # DataFrame 생성
            dates = pd.date_range('2024-01-01', periods=len(prices), freq='D')
            scenario_data = pd.DataFrame({
                'date': dates,
                'close': prices[:-1],  # 마지막 가격 제외
                'returns': simulated_returns,
                'volume': np.random.randint(100000, 1000000, len(simulated_returns))
            })
            
            scenarios.append(scenario_data)
        
        return scenarios
    
    def _generate_block_bootstrap_scenarios(self,
                                          base_data: pd.DataFrame,
                                          n_scenarios: int,
                                          simulation_days: int) -> List[pd.DataFrame]:
        """블록 부트스트랩 시나리오 (시계열 패턴 보존)"""
        scenarios = []
        
        if 'returns' in base_data.columns:
            returns = base_data['returns'].dropna()
        else:
            returns = base_data['close'].pct_change().dropna()
        
        block_size = 20  # 블록 크기 (20일)
        
        for _ in range(n_scenarios):
            simulated_returns = []
            
            while len(simulated_returns) < simulation_days:
                # 랜덤 블록 시작점 선택
                if len(returns) > block_size:
                    start_idx = np.random.randint(0, len(returns) - block_size)
                    block = returns.iloc[start_idx:start_idx + block_size]
                    simulated_returns.extend(block.values)
                else:
                    # 데이터가 부족한 경우 전체 사용
                    simulated_returns.extend(returns.values)
            
            # 필요한 길이만큼 자르기
            simulated_returns = simulated_returns[:simulation_days]
            
            # 가격 시계열 생성
            base_price = 50000
            prices = [base_price]
            
            for ret in simulated_returns:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            # DataFrame 생성
            dates = pd.date_range('2024-01-01', periods=len(prices), freq='D')
            scenario_data = pd.DataFrame({
                'date': dates,
                'close': prices[:-1],
                'returns': simulated_returns,
                'volume': np.random.randint(100000, 1000000, len(simulated_returns))
            })
            
            scenarios.append(scenario_data)
        
        return scenarios
    
    def _generate_stress_test_scenarios(self,
                                      base_data: pd.DataFrame,
                                      n_scenarios: int,
                                      simulation_days: int) -> List[pd.DataFrame]:
        """스트레스 테스트 시나리오"""
        scenarios = []
        
        if 'returns' in base_data.columns:
            base_returns = base_data['returns'].dropna()
        else:
            base_returns = base_data['close'].pct_change().dropna()
        
        mean_return = base_returns.mean()
        std_return = base_returns.std()
        
        # 다양한 스트레스 상황 정의
        stress_conditions = [
            {'name': 'market_crash', 'return_shock': -0.03, 'vol_multiplier': 2.0},
            {'name': 'high_volatility', 'return_shock': 0, 'vol_multiplier': 3.0},
            {'name': 'persistent_decline', 'return_shock': -0.001, 'vol_multiplier': 1.5},
            {'name': 'extreme_negative', 'return_shock': -0.05, 'vol_multiplier': 2.5}
        ]
        
        scenarios_per_stress = max(1, n_scenarios // len(stress_conditions))
        
        for condition in stress_conditions:
            for _ in range(scenarios_per_stress):
                # 스트레스 조건 적용
                stressed_mean = mean_return + condition['return_shock']
                stressed_std = std_return * condition['vol_multiplier']
                
                # 수익률 생성
                simulated_returns = np.random.normal(
                    stressed_mean, stressed_std, simulation_days
                )
                
                # 극단적 이벤트 추가 (10% 확률로 -5% 이상 손실)
                for i in range(len(simulated_returns)):
                    if np.random.random() < 0.1:
                        simulated_returns[i] = min(simulated_returns[i], -0.05)
                
                # 가격 시계열 생성
                base_price = 50000
                prices = [base_price]
                
                for ret in simulated_returns:
                    new_price = prices[-1] * (1 + ret)
                    prices.append(new_price)
                
                # DataFrame 생성
                dates = pd.date_range('2024-01-01', periods=len(prices), freq='D')
                scenario_data = pd.DataFrame({
                    'date': dates,
                    'close': prices[:-1],
                    'returns': simulated_returns,
                    'volume': np.random.randint(100000, 1000000, len(simulated_returns)),
                    'stress_type': condition['name']
                })
                
                scenarios.append(scenario_data)
        
        return scenarios[:n_scenarios]
    
    def _run_parallel_simulation(self,
                               strategy_class: type,
                               scenarios_data: List[pd.DataFrame],
                               strategy_params: Dict[str, Any],
                               initial_capital: float) -> List[MonteCarloScenario]:
        """병렬 시뮬레이션 실행"""
        
        max_workers = min(cpu_count(), len(scenarios_data))
        scenarios = []
        
        logger.info(f"Running parallel simulation with {max_workers} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            futures = []
            for i, scenario_data in enumerate(scenarios_data):
                future = executor.submit(
                    self._run_single_scenario,
                    strategy_class, scenario_data, strategy_params, 
                    initial_capital, i
                )
                futures.append(future)
            
            # 결과 수집
            for future in concurrent.futures.as_completed(futures):
                try:
                    scenario = future.result()
                    scenarios.append(scenario)
                except Exception as e:
                    logger.error(f"Scenario failed: {e}")
        
        return scenarios
    
    def _run_sequential_simulation(self,
                                 strategy_class: type,
                                 scenarios_data: List[pd.DataFrame],
                                 strategy_params: Dict[str, Any],
                                 initial_capital: float) -> List[MonteCarloScenario]:
        """순차 시뮬레이션 실행"""
        
        scenarios = []
        
        for i, scenario_data in enumerate(scenarios_data):
            try:
                scenario = self._run_single_scenario(
                    strategy_class, scenario_data, strategy_params,
                    initial_capital, i
                )
                scenarios.append(scenario)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Completed {i + 1}/{len(scenarios_data)} scenarios")
                    
            except Exception as e:
                logger.error(f"Scenario {i} failed: {e}")
        
        return scenarios
    
    def _run_single_scenario(self,
                           strategy_class: type,
                           scenario_data: pd.DataFrame,
                           strategy_params: Dict[str, Any],
                           initial_capital: float,
                           scenario_id: int) -> MonteCarloScenario:
        """단일 시나리오 실행"""
        
        # 전략 인스턴스 생성
        strategy = strategy_class(**strategy_params)
        
        # 간단한 백테스트 실행
        portfolio_values = self._run_scenario_backtest(
            strategy, scenario_data, initial_capital
        )
        
        # 수익률 계산
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # 성과 지표 계산
        final_capital = portfolio_values[-1]
        total_return = (final_capital / initial_capital) - 1
        
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(returns) if len(returns) > 0 else 0
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(returns) if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        var_95 = PerformanceMetrics.calculate_var(returns, 0.95) if len(returns) > 0 else 0
        
        return MonteCarloScenario(
            scenario_id=scenario_id,
            returns=returns,
            final_capital=final_capital,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_return=total_return,
            volatility=volatility,
            var_95=var_95,
            trade_count=len(scenario_data),  # 단순화
            max_consecutive_losses=0  # 추후 계산
        )
    
    def _run_scenario_backtest(self,
                             strategy,
                             scenario_data: pd.DataFrame,
                             initial_capital: float) -> List[float]:
        """시나리오별 백테스트"""
        
        portfolio_values = [initial_capital]
        
        # 간단한 랜덤 워크 기반 백테스트 (실제 구현에서는 전략 로직 사용)
        for _, row in scenario_data.iterrows():
            if 'returns' in row:
                daily_return = row['returns']
            else:
                daily_return = np.random.normal(0.0005, 0.02)
            
            # 거래 비용 적용 (0.15%)
            adjusted_return = daily_return - 0.0015
            
            new_value = portfolio_values[-1] * (1 + adjusted_return)
            portfolio_values.append(new_value)
        
        return portfolio_values[1:]  # 첫 번째 값 제외
    
    def _analyze_monte_carlo_results(self,
                                   strategy_name: str,
                                   scenarios: List[MonteCarloScenario],
                                   initial_capital: float,
                                   n_simulations: int) -> MonteCarloResult:
        """몬테카를로 결과 분석"""
        
        if not scenarios:
            raise ValueError("No valid scenarios to analyze")
        
        # 수익률 분포
        returns = [s.total_return for s in scenarios]
        sharpe_ratios = [s.sharpe_ratio for s in scenarios]
        drawdowns = [s.max_drawdown for s in scenarios]
        
        # 기본 통계량
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        
        # 분위수
        percentile_5 = np.percentile(returns, 5)
        percentile_25 = np.percentile(returns, 25)
        percentile_75 = np.percentile(returns, 75)
        percentile_95 = np.percentile(returns, 95)
        
        # 리스크 지표
        probability_of_loss = sum(1 for r in returns if r < 0) / len(returns)
        probability_of_ruin = sum(1 for r in returns if r < -0.5) / len(returns)
        
        # Expected Shortfall (5% 최악 시나리오의 평균)
        worst_5_percent = sorted(returns)[:max(1, len(returns) // 20)]
        expected_shortfall_5 = np.mean(worst_5_percent)
        
        # 최고/최악 시나리오
        best_scenario = max(scenarios, key=lambda s: s.total_return)
        worst_scenario = min(scenarios, key=lambda s: s.total_return)
        
        # 신뢰구간 (95%)
        return_confidence_95 = (np.percentile(returns, 2.5), np.percentile(returns, 97.5))
        sharpe_confidence_95 = (np.percentile(sharpe_ratios, 2.5), np.percentile(sharpe_ratios, 97.5))
        
        return MonteCarloResult(
            strategy_name=strategy_name,
            n_simulations=len(scenarios),
            initial_capital=initial_capital,
            scenarios=scenarios,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            percentile_5=percentile_5,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            percentile_95=percentile_95,
            probability_of_loss=probability_of_loss,
            probability_of_ruin=probability_of_ruin,
            expected_shortfall_5=expected_shortfall_5,
            sharpe_distribution=sharpe_ratios,
            drawdown_distribution=drawdowns,
            best_scenario=best_scenario,
            worst_scenario=worst_scenario,
            return_confidence_95=return_confidence_95,
            sharpe_confidence_95=sharpe_confidence_95
        )
    
    def generate_monte_carlo_report(self, result: MonteCarloResult) -> str:
        """몬테카를로 분석 리포트 생성"""
        
        report = f"""
        🎲 몬테카를로 시뮬레이션 리포트
        ═══════════════════════════════
        전략: {result.strategy_name}
        시뮬레이션 횟수: {result.n_simulations:,}회
        초기 자본: {result.initial_capital:,}원
        
        📊 수익률 분포
        • 평균 수익률: {result.mean_return:.2%}
        • 중간값 수익률: {result.median_return:.2%}
        • 표준편차: {result.std_return:.2%}
        
        📈 분위수 분석
        • 5%ile (최악): {result.percentile_5:.2%}
        • 25%ile: {result.percentile_25:.2%}
        • 75%ile: {result.percentile_75:.2%}
        • 95%ile (최고): {result.percentile_95:.2%}
        
        ⚠️ 리스크 분석
        • 손실 확률: {result.probability_of_loss:.1%}
        • 파산 확률 (50% 이상 손실): {result.probability_of_ruin:.1%}
        • Expected Shortfall (5%): {result.expected_shortfall_5:.2%}
        
        🎯 95% 신뢰구간
        • 수익률: {result.return_confidence_95[0]:.2%} ~ {result.return_confidence_95[1]:.2%}
        • 샤프비율: {result.sharpe_confidence_95[0]:.2f} ~ {result.sharpe_confidence_95[1]:.2f}
        
        🏆 극단 시나리오
        • 최고 수익률: {result.best_scenario.total_return:.2%}
        • 최악 수익률: {result.worst_scenario.total_return:.2%}
        
        💡 리스크 등급
        """
        
        # 리스크 등급 판정
        if result.probability_of_loss < 0.3 and result.probability_of_ruin < 0.05:
            risk_grade = "안전 (낮은 리스크)"
        elif result.probability_of_loss < 0.5 and result.probability_of_ruin < 0.15:
            risk_grade = "보통 (중간 리스크)"
        else:
            risk_grade = "위험 (높은 리스크)"
        
        report += f"\n        • 전체 등급: {risk_grade}"
        
        # 권장사항
        report += f"""
        
        📋 권장사항
        """
        
        if result.probability_of_ruin > 0.1:
            report += "\n        • 포지션 크기 축소 권장 - 파산 위험 높음"
        
        if result.std_return > 0.5:
            report += "\n        • 변동성 관리 필요 - 수익률 편차 과도"
        
        if result.mean_return < 0:
            report += "\n        • 전략 재검토 필요 - 기댓값 음수"
        
        return report.strip()
    
    def compare_monte_carlo_results(self, results: List[MonteCarloResult]) -> pd.DataFrame:
        """여러 몬테카를로 결과 비교"""
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                'Strategy': result.strategy_name,
                'Mean_Return': result.mean_return,
                'Median_Return': result.median_return,
                'Std_Return': result.std_return,
                'Probability_of_Loss': result.probability_of_loss,
                'Probability_of_Ruin': result.probability_of_ruin,
                'Expected_Shortfall_5': result.expected_shortfall_5,
                '95%_VaR': result.percentile_5,
                'Best_Case': result.percentile_95,
                'Worst_Case': result.percentile_5,
                'Sharpe_Mean': np.mean(result.sharpe_distribution),
                'Simulations': result.n_simulations
            })
        
        return pd.DataFrame(comparison_data)
    
    def save_monte_carlo_results(self, result: MonteCarloResult,
                               filename: str, save_dir: str = "monte_carlo_results"):
        """몬테카를로 결과 저장"""
        import os
        import pickle
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 전체 결과 저장
        with open(os.path.join(save_dir, f"{filename}_full.pkl"), 'wb') as f:
            pickle.dump(result, f)
        
        # 요약 통계 저장
        summary_data = {
            'returns': [s.total_return for s in result.scenarios],
            'sharpe_ratios': [s.sharpe_ratio for s in result.scenarios],
            'max_drawdowns': [s.max_drawdown for s in result.scenarios],
            'final_capitals': [s.final_capital for s in result.scenarios]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(save_dir, f"{filename}_summary.csv"), index=False)
        
        # 리포트 저장
        report = self.generate_monte_carlo_report(result)
        with open(os.path.join(save_dir, f"{filename}_report.txt"), 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Monte Carlo results saved to {save_dir}/{filename}")