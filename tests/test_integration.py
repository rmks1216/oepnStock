"""
Integration Tests for oepnStock Trading System
통합 테스트: 4단계 전략 + Phase 1 필수 모듈 통합 검증
"""

import unittest
import sys
import os
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oepnstock.core.stage1_market_flow import MarketFlowAnalyzer, MarketCondition
from oepnstock.core.stage2_support_detection import SupportDetector
from oepnstock.core.stage3_signal_confirmation import SignalConfirmator
from oepnstock.core.stage4_risk_management import RiskManager
from oepnstock.modules.critical import (
    FundamentalEventFilter,
    PortfolioConcentrationManager, 
    GapTradingStrategy,
    FilterDecision
)
from oepnstock.utils import MarketDataManager
from examples.basic_trading_example import BasicTradingSystem


class TestIntegration(unittest.TestCase):
    """통합 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.trading_system = BasicTradingSystem()
        
        # Mock 데이터 생성
        self.mock_stock_data = self._create_mock_stock_data()
        self.mock_index_data = self._create_mock_index_data()
        self.mock_portfolio = {
            'cash': 5000000,
            'total_value': 10000000,
            'positions': []
        }
    
    def _create_mock_stock_data(self) -> pd.DataFrame:
        """Mock 주식 데이터 생성"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
        np.random.seed(42)  # 재현 가능한 랜덤 데이터
        
        # 상승 추세 데이터 생성
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0.002, 0.02)  # 평균 0.2% 상승, 2% 변동성
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.8))  # 최소 20% 하락 제한
        
        # OHLCV 데이터 생성
        data = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.randint(100000, 1000000) for _ in dates]
        })
        
        data.set_index('date', inplace=True)
        return data
    
    def _create_mock_index_data(self) -> pd.DataFrame:
        """Mock 지수 데이터 생성"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
        base_value = 2500
        
        values = [base_value * (1 + np.random.normal(0.001, 0.015)) 
                 for _ in range(len(dates))]
        
        return pd.DataFrame({
            'date': dates,
            'close': values,
            'volume': [np.random.randint(1000000, 10000000) for _ in dates]
        }).set_index('date')
    
    @patch('oepnstock.utils.market_data.MarketDataManager.get_stock_data')
    def test_complete_trading_workflow(self, mock_get_data):
        """완전한 매매 워크플로우 통합 테스트"""
        
        # Mock 데이터 설정
        mock_get_data.return_value = self.mock_stock_data
        
        # 매매 기회 분석 실행
        symbol = '005930'
        result = self.trading_system.analyze_trading_opportunity(symbol)
        
        # 기본 구조 검증
        self.assertIn('symbol', result)
        self.assertIn('recommendation', result)
        self.assertIn('confidence', result)
        self.assertIn('timestamp', result)
        
        # 결과가 유효한 추천인지 확인
        valid_recommendations = ['BUY', 'BUY_PARTIAL', 'WATCH', 'NO_TRADE', 'BLOCKED', 'ERROR']
        self.assertIn(result['recommendation'], valid_recommendations)
        
        # 신뢰도가 0-1 범위인지 확인
        if 'confidence' in result:
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
    
    @patch('oepnstock.utils.market_data.MarketDataManager.get_stock_data')
    def test_market_flow_integration(self, mock_get_data):
        """시장 흐름 분석 통합 테스트"""
        
        mock_get_data.return_value = self.mock_index_data
        
        # 시장 흐름 분석기 테스트
        market_analyzer = MarketFlowAnalyzer()
        market_condition = market_analyzer.analyze_market_flow(
            self.mock_index_data, 
            self.mock_index_data,
            {'technology': self.mock_index_data},
            {'us_markets': {'sp500_change': 0.01}}
        )
        
        # 결과 검증
        self.assertIsInstance(market_condition, MarketCondition)
        self.assertIn(market_condition.regime, ['bull', 'bear', 'sideways'])
        self.assertIsInstance(market_condition.score, (int, float))
        self.assertIsInstance(market_condition.tradable, bool)
    
    def test_support_detection_integration(self):
        """지지선 감지 통합 테스트"""
        
        support_detector = SupportDetector()
        support_analysis = support_detector.detect_support_levels(self.mock_stock_data, '005930')
        
        # 결과 검증
        self.assertIsNotNone(support_analysis)
        self.assertIsInstance(support_analysis.individual_supports, list)
        
        # 지지선이 있다면 가격 범위가 합리적인지 확인
        if support_analysis.individual_supports:
            for support in support_analysis.individual_supports:
                min_price = self.mock_stock_data['low'].min()
                max_price = self.mock_stock_data['high'].max()
                self.assertGreaterEqual(support.price, min_price * 0.8)
                self.assertLessEqual(support.price, max_price * 1.2)
    
    def test_fundamental_filter_integration(self):
        """펀더멘털 필터 통합 테스트"""
        
        fundamental_filter = FundamentalEventFilter()
        filter_decision = fundamental_filter.get_filter_decision('005930')
        
        # 결과 검증
        self.assertIsInstance(filter_decision, FilterDecision)
        self.assertIsInstance(filter_decision.can_buy, bool)
        self.assertIsInstance(filter_decision.reason, str)
        self.assertIsInstance(filter_decision.risk_events, list)
        
        # 포지션 조정 배율이 합리적인지 확인
        self.assertGreaterEqual(filter_decision.position_adjustment, 0.0)
        self.assertLessEqual(filter_decision.position_adjustment, 1.0)
    
    def test_portfolio_concentration_integration(self):
        """포트폴리오 집중도 관리 통합 테스트"""
        
        portfolio_manager = PortfolioConcentrationManager()
        
        # 새 포지션 추가 테스트
        result = portfolio_manager.can_add_position(
            '005930', 
            1500000,  # 150만원 투자
            self.mock_portfolio
        )
        
        # 결과 검증
        self.assertIsInstance(result.can_add, bool)
        self.assertIsInstance(result.max_allowed_size, (int, float))
        self.assertIsInstance(result.recommended_size, (int, float))
        self.assertIsInstance(result.blocking_reasons, list)
        self.assertIsInstance(result.warnings, list)
        
        # 금액이 합리적인지 확인
        if result.can_add:
            self.assertGreater(result.max_allowed_size, 0)
            self.assertLessEqual(result.max_allowed_size, self.mock_portfolio['cash'])
    
    def test_gap_strategy_integration(self):
        """갭 전략 통합 테스트"""
        
        gap_strategy = GapTradingStrategy()
        
        # 갭 분석 테스트
        yesterday_close = 50000
        today_open = 52000  # 4% 갭 업
        
        gap_analysis = gap_strategy.analyze_gap('005930', yesterday_close, today_open)
        
        # 결과 검증
        self.assertEqual(gap_analysis.symbol, '005930')
        self.assertEqual(gap_analysis.yesterday_close, yesterday_close)
        self.assertEqual(gap_analysis.today_open, today_open)
        self.assertAlmostEqual(gap_analysis.gap_ratio, 0.04, places=2)
        
        # 갭 전략 결정 테스트
        strategy_decision = gap_strategy.determine_gap_strategy(gap_analysis)
        
        self.assertIsNotNone(strategy_decision.strategy_type)
        self.assertIsInstance(strategy_decision.entry_points, list)
        self.assertIsInstance(strategy_decision.confidence, (int, float))
    
    @patch('oepnstock.utils.market_data.MarketDataManager.get_stock_data')
    def test_risk_management_integration(self, mock_get_data):
        """리스크 관리 통합 테스트"""
        
        mock_get_data.return_value = self.mock_stock_data
        
        risk_manager = RiskManager()
        
        # 리스크 관리 계획 생성 테스트
        entry_price = 50000
        support_levels = [48000, 46000, 44000]
        resistance_levels = [52000, 54000, 56000]
        
        risk_plan = risk_manager.create_risk_management_plan(
            '005930',
            entry_price,
            support_levels,
            resistance_levels,
            'bull',
            self.mock_portfolio['total_value'],
            self.mock_portfolio
        )
        
        # 결과 검증
        self.assertIsNotNone(risk_plan.position_size)
        self.assertIsNotNone(risk_plan.stop_loss)
        self.assertIsNotNone(risk_plan.target_prices)
        self.assertIsInstance(risk_plan.risk_reward_ratio, (int, float))
        
        # 포지션 크기가 합리적인지 확인
        self.assertGreater(risk_plan.position_size.shares, 0)
        self.assertLessEqual(
            risk_plan.position_size.investment_amount, 
            self.mock_portfolio['cash']
        )
        
        # 손절가가 합리적인지 확인
        self.assertLess(risk_plan.stop_loss.price, entry_price)
        
        # 목표가가 합리적인지 확인
        for target in risk_plan.target_prices:
            self.assertGreater(target.price, entry_price)
    
    @patch('oepnstock.utils.market_data.MarketDataManager.get_stock_data')
    def test_screening_integration(self, mock_get_data):
        """스크리닝 통합 테스트"""
        
        mock_get_data.return_value = self.mock_stock_data
        
        # 여러 종목 스크리닝 테스트
        symbols = ['005930', '000660', '035420']
        screening_result = self.trading_system.run_screening(symbols)
        
        # 결과 검증
        self.assertEqual(screening_result['total_screened'], len(symbols))
        self.assertIsInstance(screening_result['buy_candidates'], list)
        self.assertIsInstance(screening_result['detailed_results'], dict)
        
        # 각 종목 결과가 있는지 확인
        for symbol in symbols:
            self.assertIn(symbol, screening_result['detailed_results'])
            result = screening_result['detailed_results'][symbol]
            self.assertIn('recommendation', result)
    
    def test_error_handling(self):
        """에러 처리 통합 테스트"""
        
        # 잘못된 심볼로 테스트
        result = self.trading_system.analyze_trading_opportunity('INVALID')
        
        # 에러가 적절히 처리되는지 확인
        self.assertIn('recommendation', result)
        # 시스템이 크래시하지 않고 결과를 반환하는지 확인
        
    def test_configuration_integration(self):
        """설정 통합 테스트"""
        
        # 설정이 모든 모듈에서 일관되게 사용되는지 확인
        from oepnstock.config import config
        
        self.assertIsNotNone(config.trading)
        self.assertIsNotNone(config.trading.max_positions)
        self.assertIsNotNone(config.trading.max_single_position_ratio)
        
        # 리스크 관리 설정 검증
        self.assertGreater(config.trading.max_positions, 0)
        self.assertGreater(config.trading.max_single_position_ratio, 0)
        self.assertLessEqual(config.trading.max_single_position_ratio, 1.0)
    
    def test_data_consistency(self):
        """데이터 일관성 테스트"""
        
        # 모든 모듈이 동일한 데이터 형식을 기대하는지 확인
        data_manager = MarketDataManager()
        
        # 데이터 매니저가 올바른 형식으로 데이터를 반환하는지 확인
        # (실제 API 호출 없이 구조만 확인)
        try:
            # Mock 함수가 정의되어 있는지 확인
            self.assertTrue(hasattr(data_manager, 'get_stock_data'))
            self.assertTrue(hasattr(data_manager, 'get_sector_data'))
        except Exception as e:
            self.fail(f"Data manager structure test failed: {e}")


class TestSystemReliability(unittest.TestCase):
    """시스템 안정성 테스트"""
    
    def setUp(self):
        self.trading_system = BasicTradingSystem()
    
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 대량 데이터로 시스템 테스트
        symbols = [f'00{i:04d}' for i in range(100)]  # 100개 가상 종목
        
        for symbol in symbols[:10]:  # 메모리 제한으로 10개만
            try:
                result = self.trading_system.analyze_trading_opportunity(symbol)
                # 각 분석이 완료된 후 가비지 컬렉션
                gc.collect()
            except Exception:
                pass  # 에러는 무시하고 메모리 테스트에 집중
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 메모리 증가가 합리적인 범위인지 확인 (100MB 이하)
        self.assertLess(memory_increase, 100 * 1024 * 1024)
    
    def test_concurrent_analysis(self):
        """동시 분석 처리 테스트"""
        import threading
        import time
        
        results = []
        errors = []
        
        def analyze_symbol(symbol):
            try:
                result = self.trading_system.analyze_trading_opportunity(symbol)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # 동시에 여러 종목 분석
        threads = []
        symbols = ['005930', '000660', '035420']
        
        for symbol in symbols:
            thread = threading.Thread(target=analyze_symbol, args=(symbol,))
            threads.append(thread)
            thread.start()
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join(timeout=30)  # 30초 타임아웃
        
        # 결과 검증
        self.assertGreaterEqual(len(results), 1)  # 최소 하나는 성공
        
        # 치명적인 에러가 없는지 확인
        for error in errors:
            self.assertNotIn('SystemError', error)
            self.assertNotIn('MemoryError', error)


if __name__ == '__main__':
    # 테스트 실행
    print("=== oepnStock Integration Tests ===")
    print()
    
    # 기본 테스트 스위트
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 테스트 클래스 추가
    suite.addTest(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTest(loader.loadTestsFromTestCase(TestSystemReliability))
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 결과 요약
    print()
    print("=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2].strip()}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed successfully!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")