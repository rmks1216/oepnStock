"""
Walk-Forward Analysis 구현
시간 경과에 따른 전략 성과 검증 및 파라미터 최적화
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

from .performance_metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardPeriod:
    """Walk-Forward 기간"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimal_params: Dict[str, Any] = field(default_factory=dict)
    train_performance: Dict[str, float] = field(default_factory=dict)
    test_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Walk-Forward Analysis 결과"""
    strategy_name: str
    periods: List[WalkForwardPeriod]
    
    # 종합 성과
    overall_return: float
    overall_sharpe: float
    overall_max_drawdown: float
    
    # 기간별 통계
    avg_train_sharpe: float
    avg_test_sharpe: float
    performance_decay: float  # 훈련 vs 테스트 성과 차이
    
    # 파라미터 안정성
    param_stability: Dict[str, float]
    optimal_param_frequency: Dict[str, Dict[str, int]]
    
    # 시계열 결과
    equity_curve: pd.Series
    period_returns: pd.DataFrame
    rolling_metrics: pd.DataFrame


class WalkForwardAnalyzer:
    """Walk-Forward Analysis 엔진"""
    
    def __init__(self, optimization_metric: str = 'sharpe_ratio'):
        self.optimization_metric = optimization_metric
        self.supported_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 
            'total_return', 'max_drawdown'
        ]
        
        if optimization_metric not in self.supported_metrics:
            raise ValueError(f"Unsupported metric: {optimization_metric}")
        
        logger.info(f"Walk-Forward Analyzer initialized with metric: {optimization_metric}")
    
    def run_walk_forward_analysis(self, 
                                 strategy_class: type,
                                 data: pd.DataFrame,
                                 param_grid: Dict[str, List],
                                 train_window: int = 252,  # 1년
                                 test_window: int = 63,    # 3개월
                                 step_size: int = 63,     # 3개월
                                 min_train_size: int = 126,  # 최소 6개월
                                 initial_capital: float = 10_000_000) -> WalkForwardResult:
        """Walk-Forward Analysis 실행"""
        
        logger.info(f"Starting Walk-Forward Analysis")
        logger.info(f"Train window: {train_window}, Test window: {test_window}, Step: {step_size}")
        
        # 데이터 검증
        if len(data) < train_window + test_window:
            raise ValueError(f"Insufficient data: need {train_window + test_window}, got {len(data)}")
        
        # Walk-Forward 기간 생성
        periods = self._generate_walk_forward_periods(
            data, train_window, test_window, step_size, min_train_size
        )
        
        logger.info(f"Generated {len(periods)} Walk-Forward periods")
        
        # 각 기간별 최적화 및 테스트
        results = []
        overall_equity = pd.Series(dtype=float)
        
        for i, period in enumerate(periods):
            logger.info(f"Processing period {i+1}/{len(periods)}: "
                       f"{period.train_start.date()} ~ {period.test_end.date()}")
            
            try:
                # 훈련 데이터에서 파라미터 최적화
                train_data = self._get_period_data(data, period.train_start, period.train_end)
                optimal_params = self._optimize_parameters(
                    strategy_class, train_data, param_grid, initial_capital
                )
                period.optimal_params = optimal_params
                
                # 훈련 성과 평가
                train_performance = self._evaluate_strategy(
                    strategy_class, train_data, optimal_params, initial_capital
                )
                period.train_performance = train_performance
                
                # 테스트 데이터에서 성과 검증
                test_data = self._get_period_data(data, period.test_start, period.test_end)
                test_performance = self._evaluate_strategy(
                    strategy_class, test_data, optimal_params, initial_capital
                )
                period.test_performance = test_performance
                
                # 테스트 기간 자산 곡선 추가
                test_equity = self._generate_equity_curve(
                    strategy_class, test_data, optimal_params, initial_capital
                )
                
                if len(overall_equity) == 0:
                    overall_equity = test_equity.copy()
                else:
                    # 이전 기간 마지막 가치에서 연결
                    scaling_factor = overall_equity.iloc[-1] / test_equity.iloc[0]
                    scaled_test_equity = test_equity * scaling_factor
                    overall_equity = pd.concat([overall_equity[:-1], scaled_test_equity])
                
                results.append(period)
                
            except Exception as e:
                logger.error(f"Failed to process period {i+1}: {e}")
                continue
        
        if not results:
            raise RuntimeError("No periods were successfully processed")
        
        # 결과 종합 분석
        return self._analyze_walk_forward_results(
            strategy_class.__name__, results, overall_equity, param_grid
        )
    
    def _generate_walk_forward_periods(self, 
                                     data: pd.DataFrame,
                                     train_window: int,
                                     test_window: int, 
                                     step_size: int,
                                     min_train_size: int) -> List[WalkForwardPeriod]:
        """Walk-Forward 기간 생성"""
        periods = []
        dates = sorted(data['date'].unique()) if 'date' in data.columns else data.index.unique()
        
        start_idx = 0
        while start_idx + train_window + test_window <= len(dates):
            # 훈련 기간
            train_start_idx = start_idx
            train_end_idx = start_idx + train_window
            
            # 실제 훈련 크기 확인
            actual_train_size = train_end_idx - train_start_idx
            if actual_train_size < min_train_size:
                break
            
            # 테스트 기간
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_window, len(dates))
            
            # 실제 테스트 크기 확인
            if test_end_idx - test_start_idx < 10:  # 최소 10일
                break
            
            period = WalkForwardPeriod(
                train_start=dates[train_start_idx],
                train_end=dates[train_end_idx - 1],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx - 1]
            )
            
            periods.append(period)
            start_idx += step_size
        
        return periods
    
    def _get_period_data(self, data: pd.DataFrame, 
                        start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """기간별 데이터 추출"""
        if 'date' in data.columns:
            mask = (data['date'] >= start_date) & (data['date'] <= end_date)
            return data[mask].copy()
        else:
            # 인덱스가 날짜인 경우
            return data.loc[start_date:end_date].copy()
    
    def _optimize_parameters(self, 
                           strategy_class: type,
                           train_data: pd.DataFrame,
                           param_grid: Dict[str, List],
                           initial_capital: float) -> Dict[str, Any]:
        """파라미터 최적화"""
        
        # 그리드 서치를 위한 파라미터 조합 생성
        param_combinations = self._generate_param_combinations(param_grid)
        
        best_params = None
        best_score = float('-inf')
        
        logger.debug(f"Testing {len(param_combinations)} parameter combinations")
        
        for params in param_combinations:
            try:
                performance = self._evaluate_strategy(
                    strategy_class, train_data, params, initial_capital
                )
                
                score = performance.get(self.optimization_metric, float('-inf'))
                
                # 최대화 목표인 경우 (샤프 비율 등)
                if self.optimization_metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'total_return']:
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                # 최소화 목표인 경우 (드로다운)
                elif self.optimization_metric == 'max_drawdown':
                    if -score > best_score:  # 음수로 변환하여 최대화
                        best_score = -score
                        best_params = params.copy()
                        
            except Exception as e:
                logger.debug(f"Parameter combination failed: {params}, error: {e}")
                continue
        
        if best_params is None:
            # 기본 파라미터 사용
            best_params = {key: values[0] for key, values in param_grid.items()}
            logger.warning("No valid parameter combination found, using defaults")
        
        return best_params
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """파라미터 그리드에서 모든 조합 생성"""
        if not param_grid:
            return [{}]
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        
        def generate_recursive(index: int, current_params: Dict[str, Any]):
            if index == len(keys):
                combinations.append(current_params.copy())
                return
            
            key = keys[index]
            for value in values[index]:
                current_params[key] = value
                generate_recursive(index + 1, current_params)
        
        generate_recursive(0, {})
        return combinations
    
    def _evaluate_strategy(self, 
                          strategy_class: type,
                          data: pd.DataFrame,
                          params: Dict[str, Any],
                          initial_capital: float) -> Dict[str, float]:
        """전략 성과 평가"""
        try:
            # 전략 인스턴스 생성
            strategy = strategy_class(**params)
            
            # 백테스트 실행 (간소화된 버전)
            equity_curve = self._run_simple_backtest(strategy, data, initial_capital)
            
            if len(equity_curve) < 2:
                return {metric: 0 for metric in self.supported_metrics}
            
            # 수익률 계산
            returns = equity_curve.pct_change().dropna()
            
            # 성과 지표 계산
            performance = {}
            
            if len(returns) > 0:
                performance['total_return'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
                performance['sharpe_ratio'] = PerformanceMetrics.calculate_sharpe_ratio(returns)
                performance['sortino_ratio'] = PerformanceMetrics.calculate_sortino_ratio(returns)
                performance['calmar_ratio'] = PerformanceMetrics.calculate_calmar_ratio(returns)
                performance['max_drawdown'] = PerformanceMetrics.calculate_max_drawdown(returns)
                performance['volatility'] = returns.std() * np.sqrt(252)
            else:
                performance = {metric: 0 for metric in self.supported_metrics}
            
            return performance
            
        except Exception as e:
            logger.error(f"Strategy evaluation failed: {e}")
            return {metric: 0 for metric in self.supported_metrics}
    
    def _run_simple_backtest(self, strategy, data: pd.DataFrame, 
                           initial_capital: float) -> pd.Series:
        """간단한 백테스트 실행"""
        portfolio_values = [initial_capital]
        
        # 가상의 간단한 백테스트 (실제 구현에서는 전략 로직 사용)
        for i in range(1, len(data)):
            # 임의의 수익률 생성 (실제로는 전략 신호 기반)
            daily_return = np.random.normal(0.0005, 0.02)  # 일일 0.05% 평균 수익, 2% 변동성
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)
        
        # 날짜 인덱스 생성
        if 'date' in data.columns:
            dates = data['date'].iloc[:len(portfolio_values)]
        else:
            dates = data.index[:len(portfolio_values)]
        
        return pd.Series(portfolio_values, index=dates)
    
    def _generate_equity_curve(self, strategy_class: type, data: pd.DataFrame,
                             params: Dict[str, Any], initial_capital: float) -> pd.Series:
        """자산 곡선 생성"""
        strategy = strategy_class(**params)
        return self._run_simple_backtest(strategy, data, initial_capital)
    
    def _analyze_walk_forward_results(self, 
                                    strategy_name: str,
                                    periods: List[WalkForwardPeriod],
                                    overall_equity: pd.Series,
                                    param_grid: Dict[str, List]) -> WalkForwardResult:
        """Walk-Forward 결과 분석"""
        
        # 전체 성과
        overall_returns = overall_equity.pct_change().dropna()
        overall_return = (overall_equity.iloc[-1] / overall_equity.iloc[0]) - 1
        overall_sharpe = PerformanceMetrics.calculate_sharpe_ratio(overall_returns)
        overall_max_drawdown = PerformanceMetrics.calculate_max_drawdown(overall_returns)
        
        # 기간별 성과 통계
        train_sharpes = [p.train_performance.get('sharpe_ratio', 0) for p in periods]
        test_sharpes = [p.test_performance.get('sharpe_ratio', 0) for p in periods]
        
        avg_train_sharpe = np.mean(train_sharpes) if train_sharpes else 0
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0
        performance_decay = avg_train_sharpe - avg_test_sharpe
        
        # 파라미터 안정성 분석
        param_stability = self._analyze_parameter_stability(periods, param_grid)
        optimal_param_frequency = self._analyze_parameter_frequency(periods, param_grid)
        
        # 기간별 수익률 DataFrame
        period_data = []
        for i, period in enumerate(periods):
            period_data.append({
                'period': i + 1,
                'train_start': period.train_start,
                'train_end': period.train_end,
                'test_start': period.test_start,
                'test_end': period.test_end,
                'train_sharpe': period.train_performance.get('sharpe_ratio', 0),
                'test_sharpe': period.test_performance.get('sharpe_ratio', 0),
                'train_return': period.train_performance.get('total_return', 0),
                'test_return': period.test_performance.get('total_return', 0),
                'optimal_params': str(period.optimal_params)
            })
        
        period_returns = pd.DataFrame(period_data)
        
        # 롤링 성과 지표
        rolling_metrics = self._calculate_rolling_metrics(overall_equity)
        
        return WalkForwardResult(
            strategy_name=strategy_name,
            periods=periods,
            overall_return=overall_return,
            overall_sharpe=overall_sharpe,
            overall_max_drawdown=overall_max_drawdown,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            performance_decay=performance_decay,
            param_stability=param_stability,
            optimal_param_frequency=optimal_param_frequency,
            equity_curve=overall_equity,
            period_returns=period_returns,
            rolling_metrics=rolling_metrics
        )
    
    def _analyze_parameter_stability(self, 
                                   periods: List[WalkForwardPeriod],
                                   param_grid: Dict[str, List]) -> Dict[str, float]:
        """파라미터 안정성 분석"""
        stability = {}
        
        for param_name in param_grid.keys():
            param_values = []
            for period in periods:
                if param_name in period.optimal_params:
                    param_values.append(period.optimal_params[param_name])
            
            if param_values:
                # 숫자형 파라미터의 경우 변동계수 계산
                try:
                    numeric_values = [float(v) for v in param_values]
                    if len(set(numeric_values)) > 1:
                        stability[param_name] = np.std(numeric_values) / np.mean(numeric_values)
                    else:
                        stability[param_name] = 0  # 모든 값이 동일
                except (ValueError, TypeError):
                    # 비숫자형 파라미터의 경우 고유값 비율
                    unique_ratio = len(set(param_values)) / len(param_values)
                    stability[param_name] = unique_ratio
            else:
                stability[param_name] = 1.0  # 데이터 없음
        
        return stability
    
    def _analyze_parameter_frequency(self, 
                                   periods: List[WalkForwardPeriod],
                                   param_grid: Dict[str, List]) -> Dict[str, Dict[str, int]]:
        """파라미터 사용 빈도 분석"""
        frequency = {}
        
        for param_name in param_grid.keys():
            frequency[param_name] = {}
            
            for period in periods:
                if param_name in period.optimal_params:
                    value = period.optimal_params[param_name]
                    frequency[param_name][str(value)] = frequency[param_name].get(str(value), 0) + 1
        
        return frequency
    
    def _calculate_rolling_metrics(self, equity_curve: pd.Series, 
                                 window: int = 63) -> pd.DataFrame:
        """롤링 성과 지표 계산"""
        if len(equity_curve) < window:
            return pd.DataFrame()
        
        returns = equity_curve.pct_change().dropna()
        rolling_data = []
        
        for i in range(window, len(returns) + 1):
            period_returns = returns.iloc[i-window:i]
            
            metrics = {
                'date': returns.index[i-1],
                'rolling_sharpe': PerformanceMetrics.calculate_sharpe_ratio(period_returns),
                'rolling_volatility': period_returns.std() * np.sqrt(252),
                'rolling_max_dd': PerformanceMetrics.calculate_max_drawdown(period_returns),
                'rolling_return': (1 + period_returns).prod() - 1
            }
            
            rolling_data.append(metrics)
        
        return pd.DataFrame(rolling_data).set_index('date')
    
    def generate_walk_forward_report(self, result: WalkForwardResult) -> str:
        """Walk-Forward 분석 리포트 생성"""
        
        report = f"""
        📊 Walk-Forward Analysis 리포트
        ═══════════════════════════════
        전략: {result.strategy_name}
        분석 기간: {len(result.periods)}개 기간
        
        🎯 전체 성과
        • 총 수익률: {result.overall_return:.2%}
        • 샤프 비율: {result.overall_sharpe:.2f}
        • 최대 드로다운: {result.overall_max_drawdown:.2%}
        
        📈 기간별 성과
        • 평균 훈련 샤프: {result.avg_train_sharpe:.2f}
        • 평균 테스트 샤프: {result.avg_test_sharpe:.2f}
        • 성과 감소: {result.performance_decay:.2f}
        
        🔧 파라미터 안정성
        """
        
        for param, stability in result.param_stability.items():
            report += f"\n        • {param}: {stability:.2f}"
        
        report += f"""
        
        📊 성과 안정성 평가
        • 성과 감소율: {(result.performance_decay / result.avg_train_sharpe * 100) if result.avg_train_sharpe != 0 else 0:.1f}%
        • 테스트 성과 안정성: {'우수' if result.avg_test_sharpe > 0.5 else '보통' if result.avg_test_sharpe > 0 else '미흡'}
        
        💡 권장사항
        """
        
        if result.performance_decay > 0.5:
            report += "\n        • 과최적화 위험 - 파라미터 범위 축소 검토"
        
        if result.avg_test_sharpe < 0:
            report += "\n        • 전략 재검토 필요 - 테스트 성과 부진"
        
        if any(stability > 0.8 for stability in result.param_stability.values()):
            report += "\n        • 파라미터 변동성 과도 - 안정성 개선 필요"
        
        return report.strip()
    
    def compare_walk_forward_results(self, results: List[WalkForwardResult]) -> pd.DataFrame:
        """여러 Walk-Forward 결과 비교"""
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                'Strategy': result.strategy_name,
                'Overall_Return': result.overall_return,
                'Overall_Sharpe': result.overall_sharpe,
                'Max_Drawdown': result.overall_max_drawdown,
                'Avg_Train_Sharpe': result.avg_train_sharpe,
                'Avg_Test_Sharpe': result.avg_test_sharpe,
                'Performance_Decay': result.performance_decay,
                'Stability_Score': np.mean(list(result.param_stability.values())),
                'Periods_Count': len(result.periods)
            })
        
        return pd.DataFrame(comparison_data)
    
    def save_walk_forward_results(self, result: WalkForwardResult, 
                                 filename: str, save_dir: str = "walk_forward_results"):
        """Walk-Forward 결과 저장"""
        import os
        import pickle
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 전체 결과 저장
        with open(os.path.join(save_dir, f"{filename}_full.pkl"), 'wb') as f:
            pickle.dump(result, f)
        
        # 요약 결과 CSV 저장
        result.period_returns.to_csv(os.path.join(save_dir, f"{filename}_periods.csv"))
        result.rolling_metrics.to_csv(os.path.join(save_dir, f"{filename}_rolling.csv"))
        
        # 리포트 저장
        report = self.generate_walk_forward_report(result)
        with open(os.path.join(save_dir, f"{filename}_report.txt"), 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Walk-Forward results saved to {save_dir}/{filename}")