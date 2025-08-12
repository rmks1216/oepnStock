"""
고급 백테스트 엔진
다양한 시나리오, 자본금 규모별 백테스트 및 최적화
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import json
import concurrent.futures
from pathlib import Path
import pickle

from ..utils.realistic_trading_costs import RealisticTradingCosts

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """백테스트 결과"""
    strategy_name: str
    capital_range: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # 기본 성과 지표
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # 거래 통계
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # 시간별 분석
    best_month: Tuple[str, float]
    worst_month: Tuple[str, float]
    best_trade: Dict[str, Any]
    worst_trade: Dict[str, Any]
    
    # 리스크 지표
    var_95: float  # 95% VaR
    cvar_95: float  # Conditional VaR
    maximum_consecutive_losses: int
    
    # 시계열 데이터
    equity_curve: pd.Series
    monthly_returns: pd.Series
    daily_returns: pd.Series
    drawdown_series: pd.Series
    
    # 메타데이터
    market_conditions: List[str] = field(default_factory=list)
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'strategy_name': self.strategy_name,
            'capital_range': self.capital_range,
            'period': f"{self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}",
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'var_95': self.var_95,
            'max_consecutive_losses': self.maximum_consecutive_losses
        }


class AdvancedBacktester:
    """고급 백테스트 엔진"""
    
    def __init__(self, data_source=None, cache_dir: str = "backtest_cache"):
        self.data_source = data_source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 결과 캐시
        self.results_cache = {}
        
        # 비용 계산기
        self.cost_calculator = RealisticTradingCosts()
        
        # 시장 시나리오 정의
        self.market_scenarios = self._define_market_scenarios()
        
        logger.info("Advanced backtester initialized")
    
    def _define_market_scenarios(self) -> Dict[str, Dict]:
        """시장 시나리오 정의"""
        return {
            "bull_market": {
                "name": "강세장",
                "period": ("2020-04-01", "2021-06-30"),
                "weight": 0.2,
                "characteristics": ["strong_uptrend", "low_volatility", "high_momentum"]
            },
            "bear_market": {
                "name": "약세장", 
                "period": ("2022-01-01", "2022-10-31"),
                "weight": 0.3,
                "characteristics": ["strong_downtrend", "high_volatility", "negative_sentiment"]
            },
            "sideways_market": {
                "name": "횡보장",
                "period": ("2023-01-01", "2023-06-30"),
                "weight": 0.3,
                "characteristics": ["range_bound", "medium_volatility", "mixed_signals"]
            },
            "high_volatility": {
                "name": "고변동성",
                "period": ("2020-03-01", "2020-04-30"),
                "weight": 0.2,
                "characteristics": ["extreme_volatility", "panic_selling", "rapid_recovery"]
            }
        }
    
    def run_comprehensive_backtest(self, strategy, capital_levels: List[float] = None) -> Dict[str, BacktestResult]:
        """종합 백테스트 실행"""
        if capital_levels is None:
            capital_levels = [1_000_000, 3_000_000, 5_000_000, 10_000_000]
        
        results = {}
        
        logger.info(f"Running comprehensive backtest for {len(capital_levels)} capital levels")
        
        for capital in capital_levels:
            capital_range = self._get_capital_range_name(capital)
            
            # 각 시장 시나리오별 백테스트
            scenario_results = {}
            for scenario_name, scenario_config in self.market_scenarios.items():
                try:
                    result = self._run_scenario_backtest(
                        strategy, capital, scenario_name, scenario_config
                    )
                    scenario_results[scenario_name] = result
                    logger.info(f"Completed {scenario_name} for {capital_range}")
                except Exception as e:
                    logger.error(f"Failed {scenario_name} for {capital_range}: {e}")
            
            # 시나리오별 결과 종합
            if scenario_results:
                combined_result = self._combine_scenario_results(
                    scenario_results, capital, capital_range
                )
                results[capital_range] = combined_result
        
        return results
    
    def _get_capital_range_name(self, capital: float) -> str:
        """자본금 범위 이름 반환"""
        if capital <= 1_000_000:
            return "100만원 이하"
        elif capital <= 3_000_000:
            return "100-300만원"
        elif capital <= 5_000_000:
            return "300-500만원"
        elif capital <= 10_000_000:
            return "500-1000만원"
        else:
            return "1000만원 이상"
    
    def _run_scenario_backtest(self, strategy, capital: float, 
                              scenario_name: str, scenario_config: Dict) -> BacktestResult:
        """시나리오별 백테스트 실행"""
        start_time = datetime.now()
        
        # 캐시 키 생성
        cache_key = f"{strategy.__class__.__name__}_{capital}_{scenario_name}"
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        # 데이터 로드
        start_date, end_date = scenario_config["period"]
        market_data = self._load_market_data(start_date, end_date)
        
        if market_data.empty:
            raise ValueError(f"No data available for period {start_date} ~ {end_date}")
        
        # 전략 실행
        portfolio_values, trades, metrics = self._execute_strategy(
            strategy, market_data, capital
        )
        
        # 결과 분석
        result = self._analyze_backtest_results(
            portfolio_values, trades, metrics, capital, 
            scenario_name, scenario_config, start_time
        )
        
        # 캐시 저장
        self.results_cache[cache_key] = result
        
        return result
    
    def _load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """시장 데이터 로드"""
        try:
            if self.data_source:
                return self.data_source.get_data(start_date, end_date)
            else:
                # 가상 데이터 생성 (실제 구현에서는 실제 데이터 사용)
                return self._generate_mock_data(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            return pd.DataFrame()
    
    def _generate_mock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """테스트용 가상 데이터 생성"""
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = dates[dates.weekday < 5]  # 주말 제외
        
        n_days = len(dates)
        np.random.seed(42)  # 재현 가능한 결과
        
        # 가상 종목 데이터 생성
        symbols = ['005930', '000660', '035420', '051910', '068270']
        data_list = []
        
        for symbol in symbols:
            # 랜덤 워크 + 트렌드
            base_price = 50000 + np.random.randint(0, 100000)
            returns = np.random.normal(0.0005, 0.02, n_days)  # 일일 수익률
            
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            for i, date in enumerate(dates):
                data_list.append({
                    'date': date,
                    'symbol': symbol,
                    'open': prices[i] * (1 + np.random.normal(0, 0.005)),
                    'high': prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                    'low': prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                    'close': prices[i],
                    'volume': np.random.randint(100000, 1000000)
                })
        
        return pd.DataFrame(data_list)
    
    def _execute_strategy(self, strategy, market_data: pd.DataFrame, 
                         initial_capital: float) -> Tuple[pd.Series, List[Dict], Dict]:
        """전략 실행"""
        portfolio_values = []
        trades = []
        current_capital = initial_capital
        positions = {}
        
        # 날짜별로 전략 실행
        for date in market_data['date'].unique():
            daily_data = market_data[market_data['date'] == date]
            
            # 전략 신호 생성 (가상 구현)
            signals = self._generate_strategy_signals(daily_data, positions)
            
            # 거래 실행
            for signal in signals:
                trade_result = self._execute_trade(signal, current_capital, positions)
                if trade_result:
                    trades.append(trade_result)
                    current_capital = trade_result['remaining_capital']
            
            # 포트폴리오 가치 계산
            portfolio_value = self._calculate_portfolio_value(
                current_capital, positions, daily_data
            )
            portfolio_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': current_capital,
                'positions_value': portfolio_value - current_capital
            })
        
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        metrics = {
            'final_capital': portfolio_df['value'].iloc[-1],
            'max_positions': len(positions),
            'total_trades': len(trades)
        }
        
        return portfolio_df['value'], trades, metrics
    
    def _generate_strategy_signals(self, daily_data: pd.DataFrame, 
                                  positions: Dict) -> List[Dict]:
        """전략 신호 생성 (가상 구현)"""
        signals = []
        
        for _, row in daily_data.iterrows():
            symbol = row['symbol']
            price = row['close']
            
            # 간단한 랜덤 신호 (실제로는 전략 로직 사용)
            if np.random.random() < 0.05:  # 5% 확률로 매수
                if symbol not in positions:
                    signals.append({
                        'action': 'buy',
                        'symbol': symbol,
                        'price': price,
                        'quantity': 100,
                        'date': row['date']
                    })
            elif symbol in positions and np.random.random() < 0.03:  # 3% 확률로 매도
                signals.append({
                    'action': 'sell',
                    'symbol': symbol,
                    'price': price,
                    'quantity': positions[symbol]['quantity'],
                    'date': row['date']
                })
        
        return signals
    
    def _execute_trade(self, signal: Dict, current_capital: float, 
                      positions: Dict) -> Optional[Dict]:
        """거래 실행"""
        action = signal['action']
        symbol = signal['symbol']
        price = signal['price']
        quantity = signal['quantity']
        
        # 거래 비용 계산 (간소화된 버전)
        trade_value = price * quantity
        if action == 'buy':
            costs = trade_value * 0.0015  # 매수 비용 0.15%
        else:
            costs = trade_value * 0.0038  # 매도 비용 0.38% (수수료 + 거래세)
        
        if action == 'buy':
            total_cost = trade_value + costs
            if current_capital >= total_cost:
                # 매수 실행
                positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'purchase_date': signal['date']
                }
                return {
                    'action': action,
                    'symbol': symbol,
                    'price': price,
                    'quantity': quantity,
                    'trade_value': trade_value,
                    'costs': costs,
                    'remaining_capital': current_capital - total_cost,
                    'date': signal['date']
                }
        
        elif action == 'sell' and symbol in positions:
            # 매도 실행
            net_proceeds = trade_value - costs
            pnl = net_proceeds - (positions[symbol]['avg_price'] * quantity)
            
            del positions[symbol]
            
            return {
                'action': action,
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'trade_value': trade_value,
                'costs': costs,
                'pnl': pnl,
                'remaining_capital': current_capital + net_proceeds,
                'date': signal['date']
            }
        
        return None
    
    def _calculate_portfolio_value(self, cash: float, positions: Dict, 
                                  daily_data: pd.DataFrame) -> float:
        """포트폴리오 가치 계산"""
        total_value = cash
        
        for symbol, position in positions.items():
            current_price_data = daily_data[daily_data['symbol'] == symbol]
            if not current_price_data.empty:
                current_price = current_price_data['close'].iloc[0]
                position_value = current_price * position['quantity']
                total_value += position_value
        
        return total_value
    
    def _analyze_backtest_results(self, portfolio_values: pd.Series, trades: List[Dict], 
                                 metrics: Dict, initial_capital: float,
                                 scenario_name: str, scenario_config: Dict,
                                 start_time: datetime) -> BacktestResult:
        """백테스트 결과 분석"""
        # 수익률 계산
        returns = portfolio_values.pct_change().dropna()
        
        # 기본 성과 지표
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # 샤프 비율 (무위험 수익률 3% 가정)
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 소르티노 비율
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # 최대 낙폭
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 거래 통계
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # VaR 계산
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0
        
        # 최고/최저 거래
        best_trade = max(trades, key=lambda x: x.get('pnl', 0)) if trades else {}
        worst_trade = min(trades, key=lambda x: x.get('pnl', 0)) if trades else {}
        
        # 월별 수익률
        monthly_returns = portfolio_values.resample('M').last().pct_change().dropna()
        best_month = (monthly_returns.idxmax().strftime('%Y-%m'), monthly_returns.max()) if len(monthly_returns) > 0 else ("", 0)
        worst_month = (monthly_returns.idxmin().strftime('%Y-%m'), monthly_returns.min()) if len(monthly_returns) > 0 else ("", 0)
        
        # 연속 손실
        consecutive_losses = self._calculate_consecutive_losses(trades)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return BacktestResult(
            strategy_name=f"Strategy_{scenario_name}",
            capital_range=self._get_capital_range_name(initial_capital),
            start_date=portfolio_values.index[0],
            end_date=portfolio_values.index[-1],
            initial_capital=initial_capital,
            final_capital=portfolio_values.iloc[-1],
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            best_month=best_month,
            worst_month=worst_month,
            best_trade=best_trade,
            worst_trade=worst_trade,
            var_95=var_95,
            cvar_95=cvar_95,
            maximum_consecutive_losses=consecutive_losses,
            equity_curve=portfolio_values,
            monthly_returns=monthly_returns,
            daily_returns=returns,
            drawdown_series=drawdown,
            market_conditions=scenario_config.get("characteristics", []),
            parameters_used={"scenario": scenario_name},
            execution_time=execution_time
        )
    
    def _calculate_consecutive_losses(self, trades: List[Dict]) -> int:
        """연속 손실 계산"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _combine_scenario_results(self, scenario_results: Dict[str, BacktestResult], 
                                 capital: float, capital_range: str) -> BacktestResult:
        """시나리오별 결과 가중 평균으로 종합"""
        total_weight = sum(self.market_scenarios[scenario]["weight"] 
                          for scenario in scenario_results.keys())
        
        # 가중 평균 계산
        weighted_metrics = {}
        for metric in ['total_return', 'annual_return', 'volatility', 'sharpe_ratio', 
                      'max_drawdown', 'win_rate', 'profit_factor']:
            weighted_value = sum(
                getattr(result, metric) * self.market_scenarios[scenario]["weight"] 
                for scenario, result in scenario_results.items()
            ) / total_weight
            weighted_metrics[metric] = weighted_value
        
        # 기타 통계 합계
        total_trades = sum(result.total_trades for result in scenario_results.values())
        total_winning = sum(result.winning_trades for result in scenario_results.values())
        
        # 대표 결과 생성
        representative_result = list(scenario_results.values())[0]
        
        return BacktestResult(
            strategy_name=f"Combined_Strategy",
            capital_range=capital_range,
            start_date=representative_result.start_date,
            end_date=representative_result.end_date,
            initial_capital=capital,
            final_capital=capital * (1 + weighted_metrics['total_return']),
            total_return=weighted_metrics['total_return'],
            annual_return=weighted_metrics['annual_return'],
            volatility=weighted_metrics['volatility'],
            sharpe_ratio=weighted_metrics['sharpe_ratio'],
            sortino_ratio=weighted_metrics.get('sortino_ratio', 0),
            max_drawdown=weighted_metrics['max_drawdown'],
            total_trades=total_trades,
            winning_trades=total_winning,
            losing_trades=total_trades - total_winning,
            win_rate=weighted_metrics['win_rate'],
            avg_win=0,  # 개별 계산 필요
            avg_loss=0,  # 개별 계산 필요
            profit_factor=weighted_metrics['profit_factor'],
            best_month=("", 0),
            worst_month=("", 0),
            best_trade={},
            worst_trade={},
            var_95=0,
            cvar_95=0,
            maximum_consecutive_losses=0,
            equity_curve=pd.Series(),
            monthly_returns=pd.Series(),
            daily_returns=pd.Series(),
            drawdown_series=pd.Series(),
            market_conditions=["combined_scenarios"],
            parameters_used={"combined": True, "scenarios": list(scenario_results.keys())},
            execution_time=sum(result.execution_time for result in scenario_results.values())
        )
    
    def compare_strategies(self, strategies: List, capital_levels: List[float] = None) -> pd.DataFrame:
        """전략 비교"""
        if capital_levels is None:
            capital_levels = [1_000_000, 3_000_000, 5_000_000, 10_000_000]
        
        comparison_results = []
        
        for strategy in strategies:
            strategy_name = strategy.__class__.__name__
            results = self.run_comprehensive_backtest(strategy, capital_levels)
            
            for capital_range, result in results.items():
                comparison_results.append({
                    'Strategy': strategy_name,
                    'Capital_Range': capital_range,
                    'Total_Return': result.total_return,
                    'Annual_Return': result.annual_return,
                    'Sharpe_Ratio': result.sharpe_ratio,
                    'Max_Drawdown': result.max_drawdown,
                    'Win_Rate': result.win_rate,
                    'Total_Trades': result.total_trades,
                    'Profit_Factor': result.profit_factor
                })
        
        return pd.DataFrame(comparison_results)
    
    def save_results(self, results: Dict[str, BacktestResult], filename: str):
        """결과 저장"""
        save_path = self.cache_dir / filename
        
        # 직렬화 가능한 형태로 변환
        serializable_results = {}
        for key, result in results.items():
            serializable_results[key] = result.to_dict()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Backtest results saved to {save_path}")
    
    def load_results(self, filename: str) -> Dict[str, Dict]:
        """결과 로드"""
        load_path = self.cache_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Results file not found: {load_path}")
        
        with open(load_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        logger.info(f"Backtest results loaded from {load_path}")
        return results