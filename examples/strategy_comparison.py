"""
Strategy Comparison Example
전략 비교 분석 예제: 4단계 전략 vs 기존 단순 전략 성과 비교
"""

import sys
import os
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
import asyncio
import pandas as pd

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

logger = get_logger(__name__)


class SimpleStrategy:
    """간단한 비교 전략 (RSI + 이동평균선만 사용)"""
    
    def __init__(self):
        self.data_manager = MarketDataManager()
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """단순 전략 분석"""
        try:
            # 데이터 수집
            end_date = date.today()
            start_date = end_date - timedelta(days=50)
            data = self.data_manager.get_stock_data(symbol, start_date, end_date)
            
            if data.empty:
                return {
                    'symbol': symbol,
                    'recommendation': 'NO_DATA',
                    'confidence': 0.0,
                    'strategy': 'simple'
                }
            
            # 간단한 지표 계산
            data['ma5'] = data['close'].rolling(window=5).mean()
            data['ma20'] = data['close'].rolling(window=20).mean()
            data['rsi'] = self._calculate_rsi(data['close'])
            
            current_price = data['close'].iloc[-1]
            ma5 = data['ma5'].iloc[-1]
            ma20 = data['ma20'].iloc[-1] 
            rsi = data['rsi'].iloc[-1]
            
            # 단순한 매수 조건
            buy_signals = 0
            total_signals = 3
            
            # 1. 가격 > 단기 이동평균
            if current_price > ma5:
                buy_signals += 1
                
            # 2. 단기 > 장기 이동평균 (상승 추세)
            if ma5 > ma20:
                buy_signals += 1
                
            # 3. RSI 과매도 구간
            if 30 <= rsi <= 50:
                buy_signals += 1
            
            confidence = buy_signals / total_signals
            
            if confidence >= 0.67:  # 2/3 이상
                recommendation = 'BUY'
            elif confidence >= 0.33:  # 1/3 이상
                recommendation = 'WATCH'
            else:
                recommendation = 'NO_TRADE'
            
            return {
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': confidence,
                'strategy': 'simple',
                'signals': buy_signals,
                'total_signals': total_signals,
                'rsi': rsi,
                'price_vs_ma5': current_price > ma5,
                'ma5_vs_ma20': ma5 > ma20
            }
            
        except Exception as e:
            logger.error(f"Error in simple strategy for {symbol}: {e}")
            return {
                'symbol': symbol,
                'recommendation': 'ERROR',
                'confidence': 0.0,
                'strategy': 'simple',
                'error': str(e)
            }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


class AdvancedStrategy:
    """4단계 고도화 전략"""
    
    def __init__(self):
        # Core 4-stage components
        self.market_analyzer = MarketFlowAnalyzer()
        self.support_detector = SupportDetector()
        self.signal_confirmator = SignalConfirmator()
        self.risk_manager = RiskManager()
        
        # Critical modules
        self.fundamental_filter = FundamentalEventFilter()
        self.portfolio_manager = PortfolioConcentrationManager()
        self.gap_strategy = GapTradingStrategy()
        
        self.data_manager = MarketDataManager()
    
    def analyze(self, symbol: str, portfolio: Dict = None) -> Dict[str, Any]:
        """고도화 전략 분석"""
        try:
            # 데이터 수집
            end_date = date.today()
            start_date = end_date - timedelta(days=100)
            
            stock_data = self.data_manager.get_stock_data(symbol, start_date, end_date)
            kospi_data = self.data_manager.get_stock_data('KOSPI', start_date, end_date)
            kosdaq_data = self.data_manager.get_stock_data('KOSDAQ', start_date, end_date)
            
            # Mock portfolio if not provided
            if portfolio is None:
                portfolio = {
                    'cash': 5000000,
                    'total_value': 10000000,
                    'positions': []
                }
            
            # 1. Market Flow Analysis
            market_condition = self.market_analyzer.analyze_market_flow(
                kospi_data, kosdaq_data, {}, {}
            )
            
            if not market_condition.tradable:
                return {
                    'symbol': symbol,
                    'recommendation': 'NO_TRADE',
                    'confidence': 0.1,
                    'strategy': 'advanced',
                    'reason': 'Market conditions unfavorable',
                    'market_score': market_condition.score
                }
            
            # 2. Support Detection
            support_analysis = self.support_detector.detect_support_levels(stock_data, symbol)
            
            if not support_analysis.strongest_support:
                return {
                    'symbol': symbol,
                    'recommendation': 'NO_TRADE',
                    'confidence': 0.2,
                    'strategy': 'advanced',
                    'reason': 'No reliable support levels'
                }
            
            # 3. Fundamental Filter
            fundamental_decision = self.fundamental_filter.get_filter_decision(symbol)
            
            if not fundamental_decision.can_buy:
                return {
                    'symbol': symbol,
                    'recommendation': 'BLOCKED',
                    'confidence': 0.0,
                    'strategy': 'advanced',
                    'reason': fundamental_decision.reason
                }
            
            # 4. Portfolio Concentration Check
            planned_investment = portfolio['total_value'] * 0.15
            concentration_check = self.portfolio_manager.can_add_position(
                symbol, planned_investment, portfolio
            )
            
            if not concentration_check.can_add:
                return {
                    'symbol': symbol,
                    'recommendation': 'BLOCKED',
                    'confidence': 0.0,
                    'strategy': 'advanced',
                    'reason': 'Portfolio concentration limits'
                }
            
            # 5. Signal Confirmation
            entry_price = stock_data['close'].iloc[-1]
            signal_confirmation = self.signal_confirmator.confirm_signal(
                stock_data,
                symbol,
                support_analysis.strongest_support.price,
                market_condition.regime
            )
            
            # 6. Final Decision
            base_confidence = signal_confirmation.weighted_score / 100  # Convert to 0-1
            adjusted_confidence = base_confidence * fundamental_decision.position_adjustment
            
            if signal_confirmation.action == 'immediate_buy':
                recommendation = 'BUY'
            elif signal_confirmation.action == 'split_entry':
                recommendation = 'BUY_PARTIAL'
            elif signal_confirmation.action == 'wait':
                recommendation = 'WATCH'
            else:
                recommendation = 'NO_TRADE'
            
            return {
                'symbol': symbol,
                'recommendation': recommendation,
                'confidence': adjusted_confidence,
                'strategy': 'advanced',
                'market_score': market_condition.score,
                'support_levels': len(support_analysis.individual_supports),
                'signal_score': signal_confirmation.weighted_score,
                'fundamental_adjustment': fundamental_decision.position_adjustment,
                'portfolio_adjustment': concentration_check.position_adjustment_factor,
                'reasoning': [
                    f"시장 점수: {market_condition.score:.1f}",
                    f"지지선 개수: {len(support_analysis.individual_supports)}",
                    f"신호 점수: {signal_confirmation.weighted_score:.1f}",
                    f"펀더멘털 조정: {fundamental_decision.position_adjustment:.1%}",
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in advanced strategy for {symbol}: {e}")
            return {
                'symbol': symbol,
                'recommendation': 'ERROR',
                'confidence': 0.0,
                'strategy': 'advanced',
                'error': str(e)
            }


class StrategyComparison:
    """전략 비교 분석 시스템"""
    
    def __init__(self):
        self.simple_strategy = SimpleStrategy()
        self.advanced_strategy = AdvancedStrategy()
    
    def compare_strategies(self, symbols: List[str], portfolio: Dict = None) -> Dict[str, Any]:
        """두 전략 성과 비교"""
        logger.info(f"Comparing strategies for {len(symbols)} symbols")
        
        results = {
            'symbols': symbols,
            'simple_results': [],
            'advanced_results': [],
            'comparison': {},
            'timestamp': datetime.now()
        }
        
        # 각 전략으로 분석
        for symbol in symbols:
            try:
                # Simple strategy
                simple_result = self.simple_strategy.analyze(symbol)
                results['simple_results'].append(simple_result)
                
                # Advanced strategy
                advanced_result = self.advanced_strategy.analyze(symbol, portfolio)
                results['advanced_results'].append(advanced_result)
                
                logger.info(f"Analyzed {symbol} - Simple: {simple_result['recommendation']}, "
                           f"Advanced: {advanced_result['recommendation']}")
                
            except Exception as e:
                logger.error(f"Error comparing strategies for {symbol}: {e}")
        
        # 비교 분석
        results['comparison'] = self._analyze_comparison(
            results['simple_results'], 
            results['advanced_results']
        )
        
        return results
    
    def _analyze_comparison(self, simple_results: List[Dict], 
                          advanced_results: List[Dict]) -> Dict[str, Any]:
        """전략 비교 분석"""
        
        comparison = {
            'total_symbols': len(simple_results),
            'simple_stats': {},
            'advanced_stats': {},
            'agreement_analysis': {},
            'performance_analysis': {}
        }
        
        # Simple strategy 통계
        simple_recs = [r['recommendation'] for r in simple_results if r['recommendation'] != 'ERROR']
        simple_confidences = [r['confidence'] for r in simple_results if r['confidence'] > 0]
        
        comparison['simple_stats'] = {
            'buy_signals': len([r for r in simple_recs if r == 'BUY']),
            'watch_signals': len([r for r in simple_recs if r == 'WATCH']),
            'no_trade': len([r for r in simple_recs if r == 'NO_TRADE']),
            'avg_confidence': sum(simple_confidences) / len(simple_confidences) if simple_confidences else 0,
            'error_count': len([r for r in simple_results if r['recommendation'] == 'ERROR'])
        }
        
        # Advanced strategy 통계
        advanced_recs = [r['recommendation'] for r in advanced_results if r['recommendation'] != 'ERROR']
        advanced_confidences = [r['confidence'] for r in advanced_results if r['confidence'] > 0]
        
        comparison['advanced_stats'] = {
            'buy_signals': len([r for r in advanced_recs if r in ['BUY', 'BUY_PARTIAL']]),
            'buy_partial': len([r for r in advanced_recs if r == 'BUY_PARTIAL']),
            'watch_signals': len([r for r in advanced_recs if r == 'WATCH']),
            'blocked': len([r for r in advanced_recs if r == 'BLOCKED']),
            'no_trade': len([r for r in advanced_recs if r == 'NO_TRADE']),
            'avg_confidence': sum(advanced_confidences) / len(advanced_confidences) if advanced_confidences else 0,
            'error_count': len([r for r in advanced_results if r['recommendation'] == 'ERROR'])
        }
        
        # 일치도 분석
        agreements = 0
        disagreements = 0
        
        for simple, advanced in zip(simple_results, advanced_results):
            if simple['recommendation'] == 'ERROR' or advanced['recommendation'] == 'ERROR':
                continue
                
            simple_bullish = simple['recommendation'] in ['BUY']
            advanced_bullish = advanced['recommendation'] in ['BUY', 'BUY_PARTIAL']
            
            if simple_bullish == advanced_bullish:
                agreements += 1
            else:
                disagreements += 1
        
        total_comparable = agreements + disagreements
        agreement_rate = agreements / total_comparable if total_comparable > 0 else 0
        
        comparison['agreement_analysis'] = {
            'agreement_count': agreements,
            'disagreement_count': disagreements,
            'agreement_rate': agreement_rate,
            'total_comparable': total_comparable
        }
        
        # 성과 분석 (신호 품질 기준)
        simple_high_confidence = len([r for r in simple_results 
                                    if r.get('confidence', 0) > 0.7 and r['recommendation'] == 'BUY'])
        advanced_high_confidence = len([r for r in advanced_results 
                                      if r.get('confidence', 0) > 0.7 and r['recommendation'] in ['BUY', 'BUY_PARTIAL']])
        
        comparison['performance_analysis'] = {
            'simple_high_confidence_signals': simple_high_confidence,
            'advanced_high_confidence_signals': advanced_high_confidence,
            'simple_selectivity': (comparison['simple_stats']['no_trade'] / 
                                 comparison['total_symbols']) if comparison['total_symbols'] > 0 else 0,
            'advanced_selectivity': ((comparison['advanced_stats']['no_trade'] + 
                                    comparison['advanced_stats']['blocked']) / 
                                   comparison['total_symbols']) if comparison['total_symbols'] > 0 else 0
        }
        
        return comparison


def main():
    """메인 실행 함수"""
    print("=== Strategy Comparison Analysis ===")
    print()
    
    # 테스트 종목들
    test_symbols = ['005930', '000660', '035420', '055550', '005380', '012330', '000270']
    symbol_names = {
        '005930': '삼성전자',
        '000660': 'SK하이닉스',
        '035420': 'NAVER',
        '055550': '신한지주',
        '005380': '현대차',
        '012330': '현대모비스',
        '000270': '기아'
    }
    
    # Mock 포트폴리오
    mock_portfolio = {
        'cash': 5000000,  # 500만원 현금
        'total_value': 10000000,  # 총 1000만원
        'positions': [
            {
                'symbol': '005930',
                'quantity': 50,
                'avg_price': 80000,
                'current_price': 85000,
                'sector': 'technology',
                'entry_date': datetime.now() - timedelta(days=30)
            }
        ]
    }
    
    # 전략 비교 실행
    comparison_system = StrategyComparison()
    results = comparison_system.compare_strategies(test_symbols, mock_portfolio)
    
    print(f"📊 Analyzed {results['comparison']['total_symbols']} symbols")
    print()
    
    # Simple Strategy 결과
    print("🔵 Simple Strategy Results:")
    print("-" * 50)
    simple_stats = results['comparison']['simple_stats']
    print(f"BUY signals: {simple_stats['buy_signals']}")
    print(f"WATCH signals: {simple_stats['watch_signals']}")
    print(f"NO_TRADE: {simple_stats['no_trade']}")
    print(f"Average confidence: {simple_stats['avg_confidence']:.1%}")
    print(f"Errors: {simple_stats['error_count']}")
    print()
    
    # Advanced Strategy 결과
    print("🟢 Advanced Strategy Results:")
    print("-" * 50)
    advanced_stats = results['comparison']['advanced_stats']
    print(f"BUY signals: {advanced_stats['buy_signals']}")
    print(f"BUY_PARTIAL signals: {advanced_stats['buy_partial']}")
    print(f"WATCH signals: {advanced_stats['watch_signals']}")
    print(f"BLOCKED: {advanced_stats['blocked']}")
    print(f"NO_TRADE: {advanced_stats['no_trade']}")
    print(f"Average confidence: {advanced_stats['avg_confidence']:.1%}")
    print(f"Errors: {advanced_stats['error_count']}")
    print()
    
    # 일치도 분석
    print("🔄 Agreement Analysis:")
    print("-" * 50)
    agreement = results['comparison']['agreement_analysis']
    print(f"Agreement rate: {agreement['agreement_rate']:.1%}")
    print(f"Agreements: {agreement['agreement_count']}")
    print(f"Disagreements: {agreement['disagreement_count']}")
    print()
    
    # 성과 분석
    print("📈 Performance Analysis:")
    print("-" * 50)
    performance = results['comparison']['performance_analysis']
    print(f"Simple high-confidence signals: {performance['simple_high_confidence_signals']}")
    print(f"Advanced high-confidence signals: {performance['advanced_high_confidence_signals']}")
    print(f"Simple selectivity: {performance['simple_selectivity']:.1%}")
    print(f"Advanced selectivity: {performance['advanced_selectivity']:.1%}")
    print()
    
    # 개별 종목 상세 비교
    print("📋 Individual Symbol Comparison:")
    print("-" * 80)
    print(f"{'Symbol':<8} {'Name':<12} {'Simple':<12} {'Advanced':<12} {'S_Conf':<8} {'A_Conf':<8}")
    print("-" * 80)
    
    for simple, advanced in zip(results['simple_results'], results['advanced_results']):
        symbol = simple['symbol']
        name = symbol_names.get(symbol, symbol)
        simple_rec = simple['recommendation'][:8]
        advanced_rec = advanced['recommendation'][:8]
        simple_conf = f"{simple.get('confidence', 0):.1%}"
        advanced_conf = f"{advanced.get('confidence', 0):.1%}"
        
        print(f"{symbol:<8} {name:<12} {simple_rec:<12} {advanced_rec:<12} {simple_conf:<8} {advanced_conf:<8}")
    
    print()
    
    # 결론
    print("📊 Strategy Comparison Summary:")
    print("-" * 50)
    
    if advanced_stats['avg_confidence'] > simple_stats['avg_confidence']:
        print("✅ Advanced strategy shows higher average confidence")
    else:
        print("⚠️  Simple strategy shows higher average confidence")
    
    if performance['advanced_selectivity'] > performance['simple_selectivity']:
        print("✅ Advanced strategy is more selective (fewer false positives)")
    else:
        print("⚠️  Simple strategy is more selective")
    
    if advanced_stats['blocked'] > 0:
        print(f"✅ Advanced strategy blocked {advanced_stats['blocked']} risky positions")
    
    print()
    print("🎯 Key Insights:")
    print(f"- Simple strategy focuses on technical indicators only")
    print(f"- Advanced strategy incorporates market flow, fundamentals, and risk management")
    print(f"- Agreement rate: {agreement['agreement_rate']:.1%} shows strategy coherence")
    print(f"- Advanced strategy's blocking mechanism prevents risky trades")
    
    print()
    print("✅ Strategy comparison completed successfully!")


if __name__ == "__main__":
    main()