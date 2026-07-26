"""
Backtesting Example
백테스팅 예제: 4단계 전략의 과거 성과 검증 (간단한 워크포워드 분석)
"""

import sys
import os
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oepnstock.core.stage1_market_flow import MarketFlowAnalyzer
from oepnstock.core.stage2_support_detection import SupportDetector
from oepnstock.core.stage3_signal_confirmation import SignalConfirmator
from oepnstock.core.stage4_risk_management import RiskManager
from oepnstock.modules.critical import (
    FundamentalEventFilter,
    PortfolioConcentrationManager,
    GapTradingStrategy
)
from oepnstock.utils import MarketDataManager, get_logger
from oepnstock.utils.krx_data_source import get_data_provider
from oepnstock.config.settings import config

logger = get_logger(__name__)


class SimpleBacktester:
    """
    간단한 백테스팅 시스템
    
    주의: 이는 개념 검증용 예제입니다.
    실제 백테스팅에서는 생존편향, 전진편향, 거래비용 등을 더 정교하게 고려해야 합니다.
    """
    
    def __init__(self, custom_config=None):
        # Use provided config or default global config
        self.config = custom_config if custom_config else config
        
        # Trading system components
        self.market_analyzer = MarketFlowAnalyzer()
        self.support_detector = SupportDetector()
        self.signal_confirmator = SignalConfirmator()
        self.risk_manager = RiskManager()
        
        # Critical modules
        self.fundamental_filter = FundamentalEventFilter()
        self.portfolio_manager = PortfolioConcentrationManager()
        self.gap_strategy = GapTradingStrategy()
        
        # Data manager
        self.data_manager = MarketDataManager()
        self.data_provider = get_data_provider()
        
        # Backtesting settings from config
        self.initial_capital = self.config.backtest.initial_capital
        self.trading_costs = {
            'commission': self.config.costs.commission_buy,
            'tax': self.config.costs.tax,
            'slippage': self.config.costs.slippage_market
        }
        
    def run_backtest(self, symbols: List[str] = None, start_date: date = None, 
                    end_date: date = None, rebalance_frequency: int = None) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            symbols: 테스트할 종목 리스트 (None이면 설정값 사용)
            start_date: 백테스트 시작일 (None이면 설정값 사용)
            end_date: 백테스트 종료일 (None이면 설정값 사용)
            rebalance_frequency: 리밸런싱 주기 (일) (None이면 설정값 사용)
            
        Returns:
            Dict: 백테스트 결과
        """
        # Use config defaults if not provided
        if symbols is None:
            symbols = self.config.backtest.test_symbols
        if start_date is None:
            start_date = date.fromisoformat(self.config.backtest.default_start_date)
        if end_date is None:
            end_date = date.fromisoformat(self.config.backtest.default_end_date)
        if rebalance_frequency is None:
            rebalance_frequency = self.config.backtest.rebalance_frequency
            
        logger.info(f"Starting backtest: {start_date} to {end_date}, {len(symbols)} symbols")
        
        # 백테스트 상태 초기화
        portfolio = {
            'cash': self.initial_capital,
            'total_value': self.initial_capital,
            'positions': [],
            'history': []
        }
        
        # 거래 기록
        trade_history = []
        daily_returns = []
        
        # 날짜별로 백테스트 실행
        current_date = start_date
        rebalance_counter = 0
        
        try:
            while current_date <= end_date:
                # 주말 스킵
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue
                
                # 포트폴리오 가치 업데이트
                portfolio_value = self._update_portfolio_value(portfolio, current_date)
                
                # 리밸런싱 시점
                if rebalance_counter % rebalance_frequency == 0:
                    new_trades = self._rebalance_portfolio(
                        portfolio, symbols, current_date
                    )
                    trade_history.extend(new_trades)
                
                # 일일 수익률 기록
                daily_return = (portfolio_value - portfolio['total_value']) / portfolio['total_value']
                daily_returns.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'daily_return': daily_return,
                    'position_count': len(portfolio['positions'])
                })
                
                portfolio['total_value'] = portfolio_value
                rebalance_counter += 1
                current_date += timedelta(days=1)
            
            # 백테스트 결과 분석
            results = self._analyze_backtest_results(
                daily_returns, trade_history, portfolio
            )
            
            logger.info(f"Backtest completed - Total Return: {results['total_return']:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {
                'error': str(e),
                'total_return': 0,
                'trades': len(trade_history)
            }
    
    def _update_portfolio_value(self, portfolio: Dict, current_date: date) -> float:
        """포트폴리오 가치 업데이트"""
        total_value = portfolio['cash']
        
        # Mock 가격 업데이트 (실제로는 시장 데이터 사용)
        for position in portfolio['positions']:
            # 간단한 랜덤 워크로 가격 변동 시뮬레이션
            np.random.seed(hash(f"{position['symbol']}{current_date}") % 2**32)
            daily_change = np.random.normal(0.001, 0.02)  # 평균 0.1% 상승, 2% 변동성
            
            new_price = position['current_price'] * (1 + daily_change)
            new_price = max(new_price, position['avg_price'] * 0.5)  # 최대 50% 하락 제한
            
            position['current_price'] = new_price
            position['market_value'] = position['quantity'] * new_price
            total_value += position['market_value']
        
        return total_value
    
    def _rebalance_portfolio(self, portfolio: Dict, symbols: List[str], 
                           current_date: date) -> List[Dict]:
        """포트폴리오 리밸런싱"""
        new_trades = []
        
        try:
            # 각 종목에 대해 매매 신호 생성
            for symbol in symbols:
                signal = self._generate_trading_signal(symbol, current_date)
                
                if signal['action'] == 'BUY':
                    trade = self._execute_buy_order(portfolio, signal)
                    if trade:
                        new_trades.append(trade)
                        
                elif signal['action'] == 'SELL':
                    trade = self._execute_sell_order(portfolio, signal)
                    if trade:
                        new_trades.append(trade)
            
            return new_trades
            
        except Exception as e:
            logger.warning(f"Error in rebalancing: {e}")
            return []
    
    def _generate_trading_signal(self, symbol: str, current_date: date) -> Dict[str, Any]:
        """매매 신호 생성 (간소화된 버전)"""
        try:
            # 실제 시장 데이터 사용
            stock_data = self._get_real_price_data(symbol, current_date)
            
            # 데이터 충분성 확인
            if len(stock_data) < 30:
                # 데이터가 부족하면 기본 신호 반환
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'price': stock_data['close'].iloc[-1] if len(stock_data) > 0 else 50000,
                    'date': current_date
                }
            
            # 설정 기반 전략 규칙
            # 1. 단기 평균 > 장기 평균 (상승 추세)
            # 2. RSI < 과매수 임계값 (과매수 아님)
            # 3. 최근 N일 중 M일 이상 상승
            
            ma_short = stock_data['close'].rolling(self.config.backtest.signal_ma_short).mean().dropna()
            ma_long = stock_data['close'].rolling(self.config.backtest.signal_ma_long).mean().dropna()
            
            if len(ma_short) == 0 or len(ma_long) == 0:
                return {
                    'symbol': symbol,
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'price': stock_data['close'].iloc[-1],
                    'date': current_date
                }
            
            ma_short_current = ma_short.iloc[-1]
            ma_long_current = ma_long.iloc[-1]
            
            # 설정 기반 RSI 계산
            delta = stock_data['close'].diff()
            rsi_period = self.config.backtest.signal_rsi_period
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            
            # RSI 계산 시 안전장치
            rsi_data = gain / loss
            rsi = 100 - (100 / (1 + rsi_data))
            rsi_clean = rsi.dropna()
            
            current_rsi = rsi_clean.iloc[-1] if len(rsi_clean) > 0 else 50
            
            # 최근 상승일 계산 (설정 기반)
            if len(stock_data) >= 3:
                recent_changes = stock_data['close'].pct_change().tail(3)
                up_days = (recent_changes > 0).sum()
            else:
                up_days = 1
            
            # 설정 기반 신호 생성
            buy_conditions = [
                ma_short_current > ma_long_current * self.config.backtest.ma_trend_factor,  # 상승 추세
                current_rsi < self.config.backtest.signal_rsi_overbought,     # 과매수 아님
                up_days >= self.config.backtest.min_recent_up_days            # 최근 상승 모멘텀
            ]
            
            if all(buy_conditions):
                action = 'BUY'
                confidence = 0.8
            elif ma_short_current < ma_long_current * self.config.backtest.sell_threshold_ratio:
                action = 'SELL'
                confidence = 0.6
            else:
                action = 'HOLD'
                confidence = 0.3
            
            return {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'price': stock_data['close'].iloc[-1],
                'date': current_date,
                'ma_short': ma_short_current,
                'ma_long': ma_long_current,
                'rsi': current_rsi
            }
            
        except Exception as e:
            logger.warning(f"Error generating signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0.0,
                'price': 50000,  # Default price
                'date': current_date
            }
    
    def _get_real_price_data(self, symbol: str, end_date: date) -> pd.DataFrame:
        """실제 가격 데이터 조회 (KRX)"""
        start_date = end_date - timedelta(days=90)  # 이전 period="3mo"와 동일한 조회 범위
        try:
            hist_data = self.data_provider.get_historical_data(symbol, start_date, end_date)

            if hist_data is not None and not hist_data.empty:
                hist_data = hist_data[hist_data.index <= pd.Timestamp(end_date)]
                return hist_data[['close']]

        except Exception as e:
            logger.warning(f"Failed to get real data for {symbol}: {e}")

        # 조회 실패 시 빈 데이터프레임 반환 (Mock으로 대체하지 않음)
        return pd.DataFrame(columns=['close'])
    
    def _execute_buy_order(self, portfolio: Dict, signal: Dict) -> Dict:
        """매수 주문 실행"""
        try:
            symbol = signal['symbol']
            price = signal['price']
            
            # 이미 보유 중이면 스킵
            existing_position = next((p for p in portfolio['positions'] 
                                    if p['symbol'] == symbol), None)
            if existing_position:
                return None
            
            # 포트폴리오 집중도 체크
            max_position_size = portfolio['total_value'] * 0.2  # 최대 20%
            investment_amount = min(max_position_size, portfolio['cash'] * 0.8)
            
            if investment_amount < price:  # 최소 1주도 살 수 없으면
                return None
            
            # 거래 비용 계산
            commission = investment_amount * self.trading_costs['commission']
            slippage = investment_amount * self.trading_costs['slippage']
            total_cost = commission + slippage
            
            # 실제 투자 가능 금액
            net_investment = investment_amount - total_cost
            quantity = int(net_investment // price)
            
            if quantity <= 0:
                return None
            
            # 실제 거래 금액
            actual_investment = quantity * price + total_cost
            
            if actual_investment > portfolio['cash']:
                return None
            
            # 포지션 추가
            portfolio['positions'].append({
                'symbol': symbol,
                'quantity': quantity,
                'avg_price': price,
                'current_price': price,
                'market_value': quantity * price,
                'entry_date': signal['date']
            })
            
            # 현금 차감
            portfolio['cash'] -= actual_investment
            
            return {
                'type': 'BUY',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'amount': actual_investment,
                'date': signal['date'],
                'commission': commission,
                'slippage': slippage
            }
            
        except Exception as e:
            logger.warning(f"Error executing buy order for {signal['symbol']}: {e}")
            return None
    
    def _execute_sell_order(self, portfolio: Dict, signal: Dict) -> Dict:
        """매도 주문 실행"""
        try:
            symbol = signal['symbol']
            price = signal['price']
            
            # 보유 포지션 찾기
            position = next((p for p in portfolio['positions'] 
                           if p['symbol'] == symbol), None)
            if not position:
                return None
            
            quantity = position['quantity']
            gross_amount = quantity * price
            
            # 거래 비용 계산
            commission = gross_amount * self.trading_costs['commission']
            tax = gross_amount * self.trading_costs['tax']  # 매도세
            slippage = gross_amount * self.trading_costs['slippage']
            total_cost = commission + tax + slippage
            
            net_amount = gross_amount - total_cost
            
            # 현금 증가
            portfolio['cash'] += net_amount
            
            # 포지션 제거
            portfolio['positions'].remove(position)
            
            # 수익률 계산
            total_investment = position['avg_price'] * quantity
            profit = net_amount - total_investment
            return_rate = profit / total_investment
            
            return {
                'type': 'SELL',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'amount': net_amount,
                'date': signal['date'],
                'profit': profit,
                'return_rate': return_rate,
                'commission': commission,
                'tax': tax,
                'slippage': slippage,
                'holding_days': (signal['date'] - position['entry_date']).days
            }
            
        except Exception as e:
            logger.warning(f"Error executing sell order for {signal['symbol']}: {e}")
            return None
    
    def _analyze_backtest_results(self, daily_returns: List[Dict], 
                                trade_history: List[Dict], 
                                final_portfolio: Dict) -> Dict[str, Any]:
        """백테스트 결과 분석"""
        
        if not daily_returns:
            return {'error': 'No trading data'}
        
        # 기본 통계
        df = pd.DataFrame(daily_returns)
        total_return = (final_portfolio['total_value'] - self.initial_capital) / self.initial_capital
        
        # 일일 수익률 통계
        daily_returns_series = df['daily_return']
        avg_daily_return = daily_returns_series.mean()
        volatility = daily_returns_series.std()
        
        # 샤프 비율 (무위험 수익률 2%로 가정)
        risk_free_rate = 0.02 / 252  # 일일 무위험 수익률
        sharpe_ratio = (avg_daily_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 최대 낙폭 (Maximum Drawdown)
        cumulative_returns = (1 + daily_returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 거래 통계
        buy_trades = [t for t in trade_history if t['type'] == 'BUY']
        sell_trades = [t for t in trade_history if t['type'] == 'SELL']
        
        profitable_trades = [t for t in sell_trades if t['profit'] > 0]
        win_rate = len(profitable_trades) / len(sell_trades) if sell_trades else 0
        
        avg_profit_per_trade = np.mean([t['profit'] for t in sell_trades]) if sell_trades else 0
        avg_return_per_trade = np.mean([t['return_rate'] for t in sell_trades]) if sell_trades else 0
        
        # 거래 비용 총액
        total_commission = sum(t.get('commission', 0) for t in trade_history)
        total_tax = sum(t.get('tax', 0) for t in trade_history)
        total_slippage = sum(t.get('slippage', 0) for t in trade_history)
        total_costs = total_commission + total_tax + total_slippage
        
        return {
            # 전체 성과
            'initial_capital': self.initial_capital,
            'final_value': final_portfolio['total_value'],
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(daily_returns)) - 1,
            
            # 리스크 지표
            'volatility': volatility * np.sqrt(252),  # 연환산 변동성
            'sharpe_ratio': sharpe_ratio * np.sqrt(252),  # 연환산 샤프 비율
            'max_drawdown': max_drawdown,
            
            # 거래 통계
            'total_trades': len(trade_history),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit_per_trade,
            'avg_return_per_trade': avg_return_per_trade,
            
            # 비용 통계
            'total_costs': total_costs,
            'cost_ratio': total_costs / self.initial_capital,
            'commission': total_commission,
            'tax': total_tax,
            'slippage': total_slippage,
            
            # 포트폴리오 상태
            'final_cash': final_portfolio['cash'],
            'final_positions': len(final_portfolio['positions']),
            'cash_ratio': final_portfolio['cash'] / final_portfolio['total_value'],
            
            # 상세 데이터
            'daily_returns': daily_returns[-10:],  # 최근 10일만
            'sample_trades': sell_trades[-5:] if len(sell_trades) >= 5 else sell_trades,
            'backtest_period_days': len(daily_returns)
        }


def main():
    """메인 실행 함수"""
    print("=== oepnStock Backtesting Example (Config-Based) ===")
    print()
    
    # 백테스터 초기화 (설정 기반)
    backtester = SimpleBacktester()
    
    # 설정에서 테스트 파라미터 가져오기
    test_symbols = config.backtest.test_symbols
    symbol_names = config.backtest.symbol_names
    start_date = date.fromisoformat(config.backtest.default_start_date)
    end_date = date.fromisoformat(config.backtest.default_end_date)
    
    print(f"📊 백테스트 기간: {start_date} ~ {end_date}")
    print(f"📈 테스트 종목: {len(test_symbols)}개")
    print(f"💰 초기 자본: {backtester.initial_capital:,}원")
    print(f"🔄 리밸런싱 주기: {config.backtest.rebalance_frequency}일")
    print(f"📈 신호 설정: MA({config.backtest.signal_ma_short},{config.backtest.signal_ma_long}), RSI({config.backtest.signal_rsi_period})")
    print()
    
    # 백테스트 실행 (설정 기반 - 파라미터 생략하면 설정값 사용)
    print("🔄 백테스트 실행 중...")
    results = backtester.run_backtest()
    
    if 'error' in results:
        print(f"❌ 백테스트 오류: {results['error']}")
        return
    
    print("✅ 백테스트 완료!")
    print()
    
    # 결과 출력
    print("📈 전체 성과:")
    print("-" * 50)
    print(f"초기 자본: {results['initial_capital']:,.0f}원")
    print(f"최종 가치: {results['final_value']:,.0f}원")
    print(f"총 수익률: {results['total_return']:.2%}")
    print(f"연환산 수익률: {results['annualized_return']:.2%}")
    print()
    
    print("📊 리스크 지표:")
    print("-" * 50)
    print(f"변동성: {results['volatility']:.2%}")
    print(f"샤프 비율: {results['sharpe_ratio']:.2f}")
    print(f"최대 낙폭: {results['max_drawdown']:.2%}")
    print()
    
    print("💼 거래 통계:")
    print("-" * 50)
    print(f"총 거래 횟수: {results['total_trades']}회")
    print(f"매수 거래: {results['buy_trades']}회")
    print(f"매도 거래: {results['sell_trades']}회")
    print(f"승률: {results['win_rate']:.1%}")
    print(f"거래당 평균 수익: {results['avg_profit_per_trade']:,.0f}원")
    print(f"거래당 평균 수익률: {results['avg_return_per_trade']:.2%}")
    print()
    
    print("💸 거래 비용:")
    print("-" * 50)
    print(f"총 거래 비용: {results['total_costs']:,.0f}원")
    print(f"비용 비율: {results['cost_ratio']:.2%}")
    print(f"수수료: {results['commission']:,.0f}원")
    print(f"거래세: {results['tax']:,.0f}원")
    print(f"슬리피지: {results['slippage']:,.0f}원")
    print()
    
    print("🏦 포트폴리오 상태:")
    print("-" * 50)
    print(f"현금: {results['final_cash']:,.0f}원")
    print(f"보유 종목 수: {results['final_positions']}개")
    print(f"현금 비율: {results['cash_ratio']:.1%}")
    print()
    
    # 샘플 거래 내역
    if results['sample_trades']:
        print("📋 최근 거래 내역 (샘플):")
        print("-" * 80)
        print(f"{'날짜':<12} {'종목':<8} {'수량':<8} {'가격':<10} {'수익률':<8} {'보유일':<6}")
        print("-" * 80)
        
        for trade in results['sample_trades']:
            symbol = trade['symbol']
            name = symbol_names.get(symbol, symbol)
            date_str = trade['date'].strftime('%Y-%m-%d')
            
            print(f"{date_str:<12} {name:<8} {trade['quantity']:,}주 "
                  f"{trade['price']:,.0f}원 {trade['return_rate']:>6.1%} "
                  f"{trade['holding_days']:>4}일")
    
    print()
    
    # 성과 평가
    print("🎯 성과 평가:")
    print("-" * 50)
    
    benchmark_return = 0.05  # 5% 벤치마크 (연환산)
    outperformed = results['annualized_return'] > benchmark_return
    
    print(f"벤치마크 (5%) 대비: {'초과 ✅' if outperformed else '미달 ❌'}")
    print(f"초과/미달 폭: {(results['annualized_return'] - benchmark_return):.2%}")
    
    if results['sharpe_ratio'] > 1.0:
        print("샤프 비율: 우수 ✅ (1.0 이상)")
    elif results['sharpe_ratio'] > 0.5:
        print("샤프 비율: 보통 ⚠️  (0.5-1.0)")
    else:
        print("샤프 비율: 개선 필요 ❌ (0.5 미만)")
    
    if abs(results['max_drawdown']) < 0.1:
        print("최대 낙폭: 양호 ✅ (10% 미만)")
    elif abs(results['max_drawdown']) < 0.2:
        print("최대 낙폭: 보통 ⚠️  (10-20%)")
    else:
        print("최대 낙폭: 주의 ❌ (20% 이상)")
    
    print()
    
    # 개선 제안
    print("💡 개선 제안:")
    print("-" * 50)
    
    if results['win_rate'] < 0.5:
        print("- 승률 개선: 신호 품질 향상 필요")
    
    if results['cost_ratio'] > 0.02:
        print("- 거래 비용 절감: 거래 빈도 조정 고려")
    
    if results['sharpe_ratio'] < 0.5:
        print("- 리스크 조정 수익률 개선: 포지션 사이징 최적화")
    
    if abs(results['max_drawdown']) > 0.15:
        print("- 손실 관리 강화: 스탑로스 규칙 개선")
    
    print()
    print("⚠️  주의사항:")
    print("이 백테스트는 개념 검증용 예제입니다.")
    print("실제 투자에서는 생존편향, 전진편향, 슬리피지 등을 더 정밀하게 고려해야 합니다.")
    print()
    print("✅ 백테스트 예제 완료!")


if __name__ == "__main__":
    main()