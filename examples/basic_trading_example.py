"""
Basic Trading Example - 4-Stage Strategy Integration
기본 매매 예제: 4단계 전략 통합 시연
"""

import sys
import os
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, Any

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
from oepnstock.config import config

logger = get_logger(__name__)


class BasicTradingSystem:
    """
    기본 매매 시스템 - 4단계 전략 + Phase 1 필수 모듈 통합
    """
    
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
        
        # Data manager
        self.data_manager = MarketDataManager()
        
        # Portfolio state
        self.portfolio = {
            'cash': 10000000,  # 1천만원
            'total_value': 10000000,
            'positions': []
        }
        
        logger.info("BasicTradingSystem initialized")
    
    def analyze_trading_opportunity(self, symbol: str) -> Dict[str, Any]:
        """
        매매 기회 종합 분석
        
        Args:
            symbol: 분석할 종목 코드
            
        Returns:
            Dict: 종합 분석 결과
        """
        logger.info(f"Analyzing trading opportunity for {symbol}")
        
        try:
            # 1. 데이터 수집
            end_date = date.today()
            start_date = end_date - timedelta(days=100)
            
            stock_data = self.data_manager.get_stock_data(symbol, start_date, end_date)
            kospi_data = self.data_manager.get_stock_data('KOSPI', start_date, end_date)
            kosdaq_data = self.data_manager.get_stock_data('KOSDAQ', start_date, end_date)
            
            # 섹터 데이터 (Mock)
            sector_data = {
                'technology': self.data_manager.get_sector_data('technology', start_date, end_date),
                'finance': self.data_manager.get_sector_data('finance', start_date, end_date)
            }
            
            # 외부 시장 데이터 (Mock)
            external_data = {
                'us_markets': {'sp500_change': 0.01, 'nasdaq_change': 0.015},
                'usd_krw': 1350.0,
                'vix': 22.0
            }
            
            logger.info(f"Data collected for {symbol} - {len(stock_data)} data points")
            
            # 2. Stage 1: 시장 흐름 분석
            logger.info("Stage 1: Market Flow Analysis")
            market_condition = self.market_analyzer.analyze_market_flow(
                kospi_data, kosdaq_data, sector_data, external_data
            )
            
            # 거래 가능 여부 1차 확인
            if not market_condition.tradable:
                return {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'recommendation': 'NO_TRADE',
                    'reason': 'Market conditions unfavorable',
                    'confidence': 0.0,
                    'market_condition': market_condition,
                    'details': market_condition.warnings
                }
            
            # 3. Stage 2: 지지선 분석
            logger.info("Stage 2: Support Level Detection")
            support_analysis = self.support_detector.detect_support_levels(stock_data, symbol)
            
            if not support_analysis.strongest_support:
                return {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'recommendation': 'NO_TRADE',
                    'reason': 'No reliable support levels found',
                    'confidence': 0.0,
                    'support_analysis': support_analysis
                }
            
            # 4. Fundamental Event Filter (Critical Module)
            logger.info("Critical Check: Fundamental Event Filter")
            fundamental_decision = self.fundamental_filter.get_filter_decision(symbol)
            
            if not fundamental_decision.can_buy:
                return {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'recommendation': 'BLOCKED',
                    'reason': fundamental_decision.reason,
                    'confidence': 0.0,
                    'retry_after': fundamental_decision.retry_after,
                    'warnings': fundamental_decision.warnings
                }
            
            # 5. Portfolio Concentration Check
            logger.info("Critical Check: Portfolio Concentration")
            entry_price = stock_data['close'].iloc[-1]
            planned_investment = self.portfolio['total_value'] * 0.15  # 15% 투자 계획
            
            concentration_check = self.portfolio_manager.can_add_position(
                symbol, planned_investment, self.portfolio
            )
            
            if not concentration_check.can_add:
                return {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'recommendation': 'BLOCKED',
                    'reason': 'Portfolio concentration limits',
                    'confidence': 0.0,
                    'details': concentration_check.blocking_reasons,
                    'max_allowed': concentration_check.max_allowed_size
                }
            
            # 6. Gap Analysis (if applicable)
            logger.info("Critical Check: Gap Analysis")
            yesterday_close = stock_data['close'].iloc[-2]
            today_open = stock_data['open'].iloc[-1]
            
            gap_analysis = self.gap_strategy.analyze_gap(symbol, yesterday_close, today_open)
            gap_strategy_plan = self.gap_strategy.determine_gap_strategy(gap_analysis)
            
            # 7. Stage 3: 신호 확인
            logger.info("Stage 3: Signal Confirmation")
            signal_confirmation = self.signal_confirmator.confirm_signal(
                stock_data,
                symbol,
                support_analysis.strongest_support.price,
                market_condition.regime
            )
            
            # 8. Stage 4: 리스크 관리 계획
            logger.info("Stage 4: Risk Management")
            support_levels = [s.price for s in support_analysis.individual_supports[:3]]
            resistance_levels = [entry_price * 1.05, entry_price * 1.10, entry_price * 1.15]  # Mock
            
            risk_plan = self.risk_manager.create_risk_management_plan(
                symbol,
                entry_price,
                support_levels,
                resistance_levels,
                market_condition.regime,
                self.portfolio['total_value'],
                self.portfolio
            )
            
            # 9. 최종 추천 결정
            final_recommendation = self._make_final_recommendation(
                signal_confirmation,
                risk_plan,
                fundamental_decision,
                concentration_check,
                gap_strategy_plan
            )
            
            # 10. 종합 결과 반환
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'recommendation': final_recommendation['action'],
                'confidence': final_recommendation['confidence'],
                'entry_price': entry_price,
                'position_size': risk_plan.position_size.shares,
                'investment_amount': risk_plan.position_size.investment_amount,
                'stop_loss': risk_plan.stop_loss.price,
                'target_prices': [t.price for t in risk_plan.target_prices],
                'risk_reward_ratio': risk_plan.risk_reward_ratio,
                'expected_return': risk_plan.expected_return,
                
                # Stage results
                'market_condition': market_condition,
                'support_analysis': support_analysis,
                'signal_confirmation': signal_confirmation,
                'risk_plan': risk_plan,
                
                # Critical module results
                'fundamental_check': fundamental_decision,
                'concentration_check': concentration_check,
                'gap_analysis': gap_analysis,
                'gap_strategy': gap_strategy_plan,
                
                # Additional info
                'warnings': final_recommendation['warnings'],
                'reasoning': final_recommendation['reasoning']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'recommendation': 'ERROR',
                'reason': str(e),
                'confidence': 0.0,
                'timestamp': datetime.now()
            }
    
    def _make_final_recommendation(self, signal_confirmation, risk_plan, 
                                 fundamental_decision, concentration_check,
                                 gap_strategy_plan) -> Dict[str, Any]:
        """최종 매매 추천 결정"""
        
        # 신호 강도 기반 기본 결정
        if signal_confirmation.action == 'immediate_buy':
            base_action = 'BUY'
            base_confidence = 0.8
        elif signal_confirmation.action == 'split_entry':
            base_action = 'BUY_PARTIAL'
            base_confidence = 0.6
        elif signal_confirmation.action == 'wait':
            base_action = 'WATCH'
            base_confidence = 0.4
        else:
            base_action = 'NO_TRADE'
            base_confidence = 0.2
        
        # 조정 요소들
        warnings = []
        reasoning = [f"신호 강도: {signal_confirmation.weighted_score:.1f}점"]
        
        # 펀더멘털 필터 조정
        if fundamental_decision.position_adjustment < 1.0:
            base_confidence *= fundamental_decision.position_adjustment
            warnings.extend(fundamental_decision.warnings or [])
            reasoning.append(f"펀더멘털 조정: {fundamental_decision.position_adjustment:.1%}")
        
        # 포트폴리오 집중도 조정  
        if concentration_check.position_adjustment_factor < 1.0:
            base_confidence *= concentration_check.position_adjustment_factor
            warnings.extend(concentration_check.warnings)
            reasoning.append(f"포지션 크기 조정: {concentration_check.position_adjustment_factor:.1%}")
        
        # 갭 전략 조정
        if gap_strategy_plan.strategy_type == 'wait_for_pullback':
            base_action = 'WAIT_PULLBACK'
            reasoning.append("갭 되돌림 대기 전략 적용")
        elif gap_strategy_plan.strategy_type == 'recalculate_all':
            base_confidence *= 0.5
            warnings.append("하락 갭으로 인한 신중 대기")
        
        # 리스크 리워드 비율 고려
        if risk_plan.risk_reward_ratio < 1.5:
            base_confidence *= 0.8
            warnings.append(f"낮은 리스크 리워드 비율: {risk_plan.risk_reward_ratio:.2f}")
        
        # 최종 신뢰도 조정
        final_confidence = max(0.1, min(0.9, base_confidence))
        
        # 신뢰도가 너무 낮으면 거래 중지
        if final_confidence < 0.3:
            base_action = 'NO_TRADE'
            reasoning.append("종합 신뢰도 부족")
        
        return {
            'action': base_action,
            'confidence': final_confidence,
            'warnings': warnings,
            'reasoning': reasoning
        }
    
    def run_screening(self, symbols: list) -> Dict[str, Any]:
        """여러 종목 스크리닝"""
        logger.info(f"Running screening for {len(symbols)} symbols")
        
        results = {}
        buy_candidates = []
        
        for symbol in symbols:
            try:
                analysis = self.analyze_trading_opportunity(symbol)
                results[symbol] = analysis
                
                if analysis['recommendation'] in ['BUY', 'BUY_PARTIAL']:
                    buy_candidates.append({
                        'symbol': symbol,
                        'confidence': analysis['confidence'],
                        'expected_return': analysis.get('expected_return', 0),
                        'risk_reward': analysis.get('risk_reward_ratio', 0)
                    })
                    
            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
                results[symbol] = {
                    'recommendation': 'ERROR',
                    'reason': str(e),
                    'confidence': 0.0
                }
        
        # 매수 후보 정렬 (신뢰도 * 기대수익률)
        buy_candidates.sort(
            key=lambda x: x['confidence'] * max(x['expected_return'], 0), 
            reverse=True
        )
        
        return {
            'total_screened': len(symbols),
            'buy_candidates': buy_candidates[:5],  # Top 5
            'detailed_results': results,
            'screening_time': datetime.now()
        }


def main():
    """메인 실행 함수"""
    print("=== oepnStock Basic Trading System Demo ===")
    print()
    
    # 시스템 초기화
    trading_system = BasicTradingSystem()
    
    # 테스트 종목들 (Mock Korean stock codes)
    test_symbols = ['005930', '000660', '035420', '055550', '005380']  
    symbol_names = {
        '005930': '삼성전자',
        '000660': 'SK하이닉스', 
        '035420': 'NAVER',
        '055550': '신한지주',
        '005380': '현대차'
    }
    
    print(f"📊 Testing {len(test_symbols)} symbols...")
    print()
    
    # 1. 개별 종목 상세 분석
    test_symbol = test_symbols[0]
    print(f"🔍 Detailed Analysis: {test_symbol} ({symbol_names[test_symbol]})")
    print("-" * 60)
    
    detailed_result = trading_system.analyze_trading_opportunity(test_symbol)
    
    print(f"추천: {detailed_result['recommendation']}")
    print(f"신뢰도: {detailed_result.get('confidence', 0):.1%}")
    
    if detailed_result['recommendation'] in ['BUY', 'BUY_PARTIAL']:
        print(f"진입가: {detailed_result['entry_price']:,.0f}원")
        print(f"투자금액: {detailed_result['investment_amount']:,.0f}원")
        print(f"수량: {detailed_result['position_size']:,}주")
        print(f"손절가: {detailed_result['stop_loss']:,.0f}원") 
        print(f"목표가: {', '.join([f'{p:,.0f}원' for p in detailed_result['target_prices']])}")
        print(f"리스크/리워드: {detailed_result['risk_reward_ratio']:.2f}")
        print(f"기대수익률: {detailed_result['expected_return']:.2%}")
    
    if detailed_result.get('warnings'):
        print("⚠️  경고사항:")
        for warning in detailed_result['warnings']:
            print(f"   - {warning}")
    
    if detailed_result.get('reasoning'):
        print("📈 판단근거:")
        for reason in detailed_result['reasoning']:
            print(f"   - {reason}")
    
    print()
    
    # 2. 종목 스크리닝
    print("🔍 Multi-Symbol Screening")
    print("-" * 60)
    
    screening_result = trading_system.run_screening(test_symbols)
    
    print(f"스크리닝 완료: {screening_result['total_screened']}개 종목")
    print(f"매수 후보: {len(screening_result['buy_candidates'])}개")
    print()
    
    if screening_result['buy_candidates']:
        print("📋 Top Buy Candidates:")
        for i, candidate in enumerate(screening_result['buy_candidates'], 1):
            symbol = candidate['symbol']
            name = symbol_names.get(symbol, symbol)
            print(f"{i}. {symbol} ({name})")
            print(f"   신뢰도: {candidate['confidence']:.1%}")
            print(f"   기대수익률: {candidate['expected_return']:.2%}")
            print(f"   리스크/리워드: {candidate['risk_reward']:.2f}")
            print()
    
    # 3. 시스템 통계
    print("📊 System Statistics")
    print("-" * 60)
    
    recommendations = [r['recommendation'] for r in screening_result['detailed_results'].values()]
    
    from collections import Counter
    rec_counts = Counter(recommendations)
    
    for rec, count in rec_counts.items():
        print(f"{rec}: {count}개 ({count/len(test_symbols):.1%})")
    
    print()
    print("✅ Demo completed successfully!")


if __name__ == "__main__":
    main()