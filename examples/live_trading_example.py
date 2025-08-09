"""
Live Trading Example - 실시간 단기매매 테스트
실제 API 연결 전 페이퍼 트레이딩으로 안전하게 테스트
"""

import asyncio
import sys
import os
from datetime import datetime, time, timedelta
from typing import Dict, List, Any
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
from oepnstock.utils.free_data_sources import get_data_provider
from oepnstock.config import config

logger = get_logger(__name__)


class LiveTradingSystem:
    """
    실시간 단기매매 시스템
    페이퍼 트레이딩 모드로 안전하게 테스트 가능
    """
    
    def __init__(self, paper_trading=True):
        # Core components
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
        
        # Trading state
        self.paper_trading = paper_trading
        self.is_active = False
        self.portfolio = {
            'cash': 10000000,  # 1천만원 시작
            'total_value': 10000000,
            'positions': [],
            'daily_pnl': 0,
            'trades_today': 0
        }
        
        # Watchlist (monitoring symbols)
        self.watchlist = [
            '005930',  # 삼성전자
            '000660',  # SK하이닉스
            '035420',  # NAVER
            '055550',  # 신한지주
            '005380'   # 현대차
        ]
        
        logger.info(f"LiveTradingSystem initialized - Paper Trading: {paper_trading}")
    
    async def start_live_trading(self):
        """실시간 거래 시작"""
        logger.info("🚀 Starting live trading system...")
        
        self.is_active = True
        
        # 한국 증시 시간 체크
        trading_hours = self.get_korean_trading_hours()
        
        try:
            while self.is_active:
                current_time = datetime.now()
                
                # 장중 시간 체크
                if self.is_market_open(current_time, trading_hours):
                    logger.info(f"📊 Market is open - Running analysis at {current_time}")
                    
                    # 1. 시장 상황 체크
                    market_condition = await self.analyze_market_condition()
                    
                    if market_condition['tradable']:
                        # 2. 종목 스크리닝
                        opportunities = await self.screen_opportunities()
                        
                        # 3. 매매 실행 (상위 3개만)
                        for opportunity in opportunities[:3]:
                            await self.execute_trade(opportunity)
                    
                    # 4. 포지션 모니터링
                    await self.monitor_positions()
                    
                    # 5분마다 체크
                    await asyncio.sleep(300)
                else:
                    logger.info(f"💤 Market closed - Next check in 30 minutes")
                    # 장외 시간엔 30분마다 체크
                    await asyncio.sleep(1800)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Stopping live trading system...")
            await self.stop_trading()
    
    def get_korean_trading_hours(self) -> Dict[str, time]:
        """한국 증시 거래 시간"""
        return {
            'market_open': time(9, 0),    # 09:00
            'market_close': time(15, 30), # 15:30
            'lunch_start': time(11, 30),  # 11:30 (점심시간 피하기)
            'lunch_end': time(13, 0)      # 13:00
        }
    
    def is_market_open(self, current_time: datetime, trading_hours: Dict) -> bool:
        """장중 시간인지 확인"""
        # 주말 체크
        if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        current_time_only = current_time.time()
        
        # 장 시작 전이거나 장 마감 후
        if (current_time_only < trading_hours['market_open'] or 
            current_time_only > trading_hours['market_close']):
            return False
        
        # 점심시간 체크 (거래량 낮음)
        if (trading_hours['lunch_start'] <= current_time_only <= trading_hours['lunch_end']):
            return False
        
        return True
    
    async def analyze_market_condition(self) -> Dict[str, Any]:
        """현재 시장 상황 분석"""
        logger.info("📈 Analyzing current market condition...")
        
        # 지수 데이터 수집
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        kospi_data = self.data_manager.get_stock_data('KOSPI', start_date, end_date)
        kosdaq_data = self.data_manager.get_stock_data('KOSDAQ', start_date, end_date)
        
        # 섹터 데이터 (Mock)
        sector_data = {
            'technology': self.data_manager.get_sector_data('technology', start_date, end_date),
            'finance': self.data_manager.get_sector_data('finance', start_date, end_date)
        }
        
        # 외부 시장 데이터 (실제로는 API에서 수집)
        external_data = {
            'us_markets': {'sp500_change': 0.01, 'nasdaq_change': 0.015},
            'usd_krw': 1350.0,
            'vix': 22.0
        }
        
        # 시장 흐름 분석
        market_condition = self.market_analyzer.analyze_market_flow(
            kospi_data, kosdaq_data, sector_data, external_data
        )
        
        return {
            'tradable': market_condition.tradable,
            'score': market_condition.score,
            'regime': market_condition.regime,
            'warnings': market_condition.warnings
        }
    
    async def screen_opportunities(self) -> List[Dict[str, Any]]:
        """매매 기회 스크리닝"""
        logger.info("🔍 Screening trading opportunities...")
        
        opportunities = []
        
        for symbol in self.watchlist:
            try:
                # 기본 분석 (실제로는 analyze_trading_opportunity 사용)
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=60)
                
                stock_data = self.data_manager.get_stock_data(symbol, start_date, end_date)
                current_price = stock_data['close'].iloc[-1]
                
                # 간단한 스크리닝 로직
                rsi = self.calculate_rsi(stock_data['close'])
                volume_ratio = stock_data['volume'].iloc[-1] / stock_data['volume'].rolling(20).mean().iloc[-1]
                
                if rsi < 35 and volume_ratio > 1.5:  # 과매도 + 거래량 증가
                    opportunities.append({
                        'symbol': symbol,
                        'current_price': current_price,
                        'rsi': rsi,
                        'volume_ratio': volume_ratio,
                        'score': (50 - rsi) * volume_ratio  # 간단한 점수
                    })
                    
            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
        
        # 점수순 정렬
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Found {len(opportunities)} trading opportunities")
        return opportunities
    
    async def execute_trade(self, opportunity: Dict[str, Any]):
        """매매 실행 (페이퍼 트레이딩)"""
        symbol = opportunity['symbol']
        current_price = opportunity['current_price']
        
        logger.info(f"💰 Executing trade for {symbol} at {current_price:,.0f}")
        
        if self.paper_trading:
            # 페이퍼 트레이딩: 실제 주문 없이 기록만
            investment_amount = self.portfolio['total_value'] * 0.1  # 10% 투자
            shares = int(investment_amount / current_price)
            
            # 포지션 추가
            position = {
                'symbol': symbol,
                'entry_price': current_price,
                'shares': shares,
                'investment': shares * current_price,
                'entry_time': datetime.now(),
                'stop_loss': current_price * 0.97,  # 3% 손절
                'target_price': current_price * 1.06  # 6% 목표
            }
            
            self.portfolio['positions'].append(position)
            self.portfolio['cash'] -= position['investment']
            self.portfolio['trades_today'] += 1
            
            logger.info(f"📝 Paper trade executed: {shares} shares of {symbol}")
        else:
            # 실제 거래 (API 연결 필요)
            logger.warning("Real trading not implemented - use paper trading mode")
    
    async def monitor_positions(self):
        """포지션 모니터링 및 청산 관리"""
        if not self.portfolio['positions']:
            return
        
        logger.info(f"📊 Monitoring {len(self.portfolio['positions'])} positions...")
        
        positions_to_close = []
        
        for i, position in enumerate(self.portfolio['positions']):
            symbol = position['symbol']
            
            try:
                # 현재가 조회
                current_data = self.data_manager.get_stock_data(
                    symbol, datetime.now().date(), datetime.now().date()
                )
                current_price = current_data['close'].iloc[-1]
                
                # 손익 계산
                pnl = (current_price - position['entry_price']) / position['entry_price']
                
                # 청산 조건 체크
                should_close = False
                close_reason = ""
                
                if current_price <= position['stop_loss']:
                    should_close = True
                    close_reason = "Stop Loss"
                elif current_price >= position['target_price']:
                    should_close = True
                    close_reason = "Target Reached"
                elif (datetime.now() - position['entry_time']).days >= 3:
                    should_close = True
                    close_reason = "Max Holding Period"
                
                if should_close:
                    positions_to_close.append((i, position, current_price, close_reason))
                    
                logger.info(f"{symbol}: {pnl:+.2%} PnL (Current: {current_price:,.0f})")
                
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
        
        # 포지션 청산
        for i, position, exit_price, reason in reversed(positions_to_close):
            await self.close_position(i, position, exit_price, reason)
    
    async def close_position(self, index: int, position: Dict, exit_price: float, reason: str):
        """포지션 청산"""
        symbol = position['symbol']
        pnl_amount = (exit_price - position['entry_price']) * position['shares']
        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        
        # 포트폴리오 업데이트
        self.portfolio['cash'] += position['shares'] * exit_price
        self.portfolio['daily_pnl'] += pnl_amount
        del self.portfolio['positions'][index]
        
        logger.info(f"🔒 Position closed: {symbol} - {reason}")
        logger.info(f"   PnL: {pnl_amount:+,.0f}원 ({pnl_pct:+.2%})")
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    async def stop_trading(self):
        """거래 중단"""
        self.is_active = False
        
        # 포지션 현황 출력
        total_value = self.portfolio['cash']
        for pos in self.portfolio['positions']:
            # 현재가 조회 (간단히 진입가로 계산)
            total_value += pos['shares'] * pos['entry_price']
        
        self.portfolio['total_value'] = total_value
        
        logger.info("📊 Trading stopped. Portfolio summary:")
        logger.info(f"   Cash: {self.portfolio['cash']:,.0f}원")
        logger.info(f"   Positions: {len(self.portfolio['positions'])}개")
        logger.info(f"   Total Value: {total_value:,.0f}원")
        logger.info(f"   Daily PnL: {self.portfolio['daily_pnl']:+,.0f}원")


async def main():
    """메인 실행 함수"""
    print("=== oepnStock Live Trading System ===")
    print("⚠️  Paper Trading Mode - 실제 거래 없음")
    print()
    
    # 실시간 거래 시스템 시작
    trading_system = LiveTradingSystem(paper_trading=True)
    
    try:
        await trading_system.start_live_trading()
    except KeyboardInterrupt:
        print("\n👋 Trading system stopped by user")


if __name__ == "__main__":
    asyncio.run(main())