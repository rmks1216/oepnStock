"""
고급 성과 지표 계산
다양한 리스크 조정 수익률 및 통계적 지표
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """성과 리포트"""
    # 기본 수익률 지표
    total_return: float
    annual_return: float
    volatility: float
    
    # 리스크 조정 수익률
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # 꼬리 위험 지표
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    tail_ratio: float
    
    # 드로다운 분석
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int
    recovery_factor: float
    
    # 분포 특성
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    # 상관관계 및 베타
    market_correlation: float
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float


class PerformanceMetrics:
    """고급 성과 지표 계산"""
    
    @staticmethod
    def calculate_comprehensive_metrics(returns: pd.Series, 
                                      benchmark_returns: pd.Series = None,
                                      risk_free_rate: float = 0.03) -> PerformanceReport:
        """종합 성과 지표 계산"""
        
        # 기본 검증
        if len(returns) == 0:
            raise ValueError("Returns series is empty")
        
        returns = returns.dropna()
        
        # 기본 수익률 지표
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 리스크 조정 수익률
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate)
        calmar_ratio = PerformanceMetrics.calculate_calmar_ratio(returns)
        omega_ratio = PerformanceMetrics.calculate_omega_ratio(returns, risk_free_rate / 252)
        
        # VaR 및 CVaR
        var_95 = PerformanceMetrics.calculate_var(returns, confidence=0.95)
        var_99 = PerformanceMetrics.calculate_var(returns, confidence=0.99)
        cvar_95 = PerformanceMetrics.calculate_cvar(returns, confidence=0.95)
        cvar_99 = PerformanceMetrics.calculate_cvar(returns, confidence=0.99)
        tail_ratio = PerformanceMetrics.calculate_tail_ratio(returns)
        
        # 드로다운 분석
        drawdown_stats = PerformanceMetrics.analyze_drawdowns(returns)
        
        # 분포 특성
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # 벤치마크 비교 (있는 경우)
        market_correlation = 0
        beta = 0
        alpha = 0
        tracking_error = 0
        information_ratio = 0
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            benchmark_returns = benchmark_returns.dropna()
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 1:
                market_correlation = aligned_returns.corr(aligned_benchmark)
                beta = PerformanceMetrics.calculate_beta(aligned_returns, aligned_benchmark)
                alpha = PerformanceMetrics.calculate_alpha(
                    aligned_returns, aligned_benchmark, risk_free_rate, beta
                )
                tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(252)
                information_ratio = PerformanceMetrics.calculate_information_ratio(
                    aligned_returns, aligned_benchmark
                )
        
        return PerformanceReport(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            tail_ratio=tail_ratio,
            max_drawdown=drawdown_stats['max_drawdown'],
            avg_drawdown=drawdown_stats['avg_drawdown'],
            drawdown_duration=drawdown_stats['max_duration'],
            recovery_factor=drawdown_stats['recovery_factor'],
            skewness=skewness,
            kurtosis=kurtosis,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            market_correlation=market_correlation,
            beta=beta,
            alpha=alpha,
            tracking_error=tracking_error,
            information_ratio=information_ratio
        )
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """샤프 비율 계산"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns.mean() - risk_free_rate / 252
        volatility = returns.std()
        
        if volatility == 0:
            return 0
        
        return excess_returns / volatility * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, 
                               target_return: float = 0.03) -> float:
        """소르티노 비율 (하방 리스크만 고려)"""
        if len(returns) == 0:
            return 0
        
        target_daily = target_return / 252
        excess_returns = returns.mean() - target_daily
        
        downside_returns = returns[returns < target_daily]
        if len(downside_returns) == 0:
            return np.inf if excess_returns > 0 else 0
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return np.inf if excess_returns > 0 else 0
        
        return excess_returns / downside_std * np.sqrt(252)
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """칼마 비율 = 연간 수익률 / 최대 드로다운"""
        if len(returns) == 0:
            return 0
        
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(returns)
        
        if max_drawdown == 0:
            return np.inf if annual_return > 0 else 0
        
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0) -> float:
        """오메가 비율"""
        if len(returns) == 0:
            return 0
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return np.inf if gains.sum() > 0 else 1
        
        return gains.sum() / losses.sum()
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Value at Risk"""
        if len(returns) == 0:
            return 0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0
        
        var = PerformanceMetrics.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
        """꼬리 비율 (극단적 수익/손실 비율)"""
        if len(returns) == 0:
            return 0
        
        sorted_returns = np.sort(returns)
        n = len(returns)
        
        tail_size = max(1, int(n * percentile))
        right_tail = sorted_returns[-tail_size:].mean()  # 상위 꼬리
        left_tail = sorted_returns[:tail_size].mean()    # 하위 꼬리
        
        if left_tail == 0:
            return np.inf if right_tail > 0 else 0
        
        return abs(right_tail / left_tail)
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """최대 드로다운"""
        if len(returns) == 0:
            return 0
        
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        return drawdown.min()
    
    @staticmethod
    def analyze_drawdowns(returns: pd.Series) -> Dict[str, float]:
        """드로다운 분석"""
        if len(returns) == 0:
            return {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'max_duration': 0,
                'recovery_factor': 0
            }
        
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        # 드로다운 기간 분석
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_in_dd in in_drawdown:
            if is_in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        max_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # 회복 팩터 (총 수익률 / 최대 드로다운)
        total_return = cumulative_returns.iloc[-1] - 1
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_duration': max_duration,
            'recovery_factor': recovery_factor
        }
    
    @staticmethod
    def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """베타 계산"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0
        
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) < 2:
            return 0
        
        covariance = aligned_returns.cov(aligned_benchmark)
        benchmark_variance = aligned_benchmark.var()
        
        if benchmark_variance == 0:
            return 0
        
        return covariance / benchmark_variance
    
    @staticmethod
    def calculate_alpha(returns: pd.Series, benchmark_returns: pd.Series,
                       risk_free_rate: float, beta: float) -> float:
        """알파 계산 (CAPM)"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0
        
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return 0
        
        portfolio_return = aligned_returns.mean() * 252
        benchmark_return = aligned_benchmark.mean() * 252
        
        expected_return = risk_free_rate + beta * (benchmark_return - risk_free_rate)
        alpha = portfolio_return - expected_return
        
        return alpha
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> float:
        """정보 비율"""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0
        
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return 0
        
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return excess_returns.mean() / tracking_error * np.sqrt(252)
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """롤링 성과 지표"""
        if len(returns) < window:
            logger.warning(f"Not enough data for rolling metrics (need {window}, got {len(returns)})")
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(window, len(returns) + 1):
            period_returns = returns.iloc[i-window:i]
            
            metrics = {
                'date': returns.index[i-1],
                'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(period_returns),
                'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(period_returns),
                'max_drawdown': PerformanceMetrics.calculate_max_drawdown(period_returns),
                'volatility': period_returns.std() * np.sqrt(252),
                'var_95': PerformanceMetrics.calculate_var(period_returns, 0.95)
            }
            
            rolling_data.append(metrics)
        
        return pd.DataFrame(rolling_data).set_index('date')
    
    @staticmethod
    def performance_attribution(returns: pd.Series, 
                               factor_returns: pd.DataFrame) -> Dict[str, float]:
        """성과 기여도 분석"""
        if len(returns) == 0 or factor_returns.empty:
            return {}
        
        # 수익률과 팩터 수익률 정렬
        aligned_data = pd.concat([returns, factor_returns], axis=1, join='inner')
        
        if len(aligned_data) < 10:  # 최소 데이터 요구사항
            return {}
        
        y = aligned_data.iloc[:, 0]  # 포트폴리오 수익률
        X = aligned_data.iloc[:, 1:]  # 팩터 수익률
        
        try:
            # 선형 회귀
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # 각 팩터의 기여도
            attribution = {}
            for i, factor in enumerate(X.columns):
                attribution[factor] = model.coef_[i] * X[factor].mean() * 252
            
            # 알파 (절편)
            attribution['alpha'] = model.intercept_ * 252
            
            # R-squared
            attribution['r_squared'] = model.score(X, y)
            
            return attribution
            
        except Exception as e:
            logger.error(f"Performance attribution failed: {e}")
            return {}
    
    @staticmethod
    def stress_test_analysis(returns: pd.Series, 
                           stress_scenarios: Dict[str, float]) -> Dict[str, float]:
        """스트레스 테스트"""
        results = {}
        
        for scenario_name, shock_size in stress_scenarios.items():
            # 시나리오별 스트레스 적용
            if scenario_name == "market_crash":
                # 시장 급락 시나리오 (-20% 충격)
                stressed_returns = returns - shock_size
            elif scenario_name == "volatility_spike":
                # 변동성 급증 시나리오
                stressed_returns = returns * (1 + shock_size)
            elif scenario_name == "correlation_breakdown":
                # 상관관계 붕괴 시나리오
                stressed_returns = returns + np.random.normal(0, shock_size, len(returns))
            else:
                # 일반적인 충격
                stressed_returns = returns + shock_size
            
            # 스트레스 받은 포트폴리오의 성과
            stressed_sharpe = PerformanceMetrics.calculate_sharpe_ratio(stressed_returns)
            stressed_max_dd = PerformanceMetrics.calculate_max_drawdown(stressed_returns)
            stressed_var = PerformanceMetrics.calculate_var(stressed_returns, 0.95)
            
            results[scenario_name] = {
                'sharpe_ratio': stressed_sharpe,
                'max_drawdown': stressed_max_dd,
                'var_95': stressed_var,
                'total_return': (1 + stressed_returns).prod() - 1
            }
        
        return results
    
    @staticmethod
    def generate_performance_summary(returns: pd.Series, 
                                   benchmark_returns: pd.Series = None) -> str:
        """성과 요약 리포트 생성"""
        report = PerformanceMetrics.calculate_comprehensive_metrics(
            returns, benchmark_returns
        )
        
        summary = f"""
        📊 성과 분석 리포트
        ═══════════════════════
        
        🎯 수익률 지표
        • 총 수익률: {report.total_return:.2%}
        • 연간 수익률: {report.annual_return:.2%}
        • 변동성: {report.volatility:.2%}
        
        ⚖️ 리스크 조정 수익률
        • 샤프 비율: {report.sharpe_ratio:.2f}
        • 소르티노 비율: {report.sortino_ratio:.2f}
        • 칼마 비율: {report.calmar_ratio:.2f}
        • 오메가 비율: {report.omega_ratio:.2f}
        
        📉 리스크 지표
        • 최대 드로다운: {report.max_drawdown:.2%}
        • VaR (95%): {report.var_95:.2%}
        • CVaR (95%): {report.cvar_95:.2%}
        • 꼬리 비율: {report.tail_ratio:.2f}
        
        📈 분포 특성
        • 왜도: {report.skewness:.2f}
        • 첨도: {report.kurtosis:.2f}
        • 정규성 검정 p-value: {report.jarque_bera_pvalue:.4f}
        
        {'📊 벤치마크 비교' if benchmark_returns is not None else ''}
        {f'• 상관계수: {report.market_correlation:.2f}' if benchmark_returns is not None else ''}
        {f'• 베타: {report.beta:.2f}' if benchmark_returns is not None else ''}
        {f'• 알파: {report.alpha:.2%}' if benchmark_returns is not None else ''}
        {f'• 정보 비율: {report.information_ratio:.2f}' if benchmark_returns is not None else ''}
        """
        
        return summary.strip()