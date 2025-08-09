"""
Free Data Sources for Paper Trading
페이퍼 트레이딩용 무료 데이터 소스
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import time
import requests
from .logger import get_logger

logger = get_logger(__name__)


class FreeDataProvider:
    """무료 데이터 제공자 통합 클래스"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # 1분 캐시
        
        # 한국 주요 종목 매핑 (Yahoo Finance용)
        self.korean_tickers = {
            '005930': '005930.KS',  # 삼성전자
            '000660': '000660.KS',  # SK하이닉스
            '035420': '035420.KS',  # NAVER
            '055550': '055550.KS',  # 신한지주
            '005380': '005380.KS',  # 현대차
            '068270': '068270.KS',  # 셀트리온
            '006400': '006400.KS',  # 삼성SDI
            '035720': '035720.KS',  # 카카오
            '207940': '207940.KS',  # 삼성바이오로직스
            '051910': '051910.KS',  # LG화학
        }
        
        # 지수 매핑
        self.index_mapping = {
            'KOSPI': '^KS11',
            'KOSDAQ': '^KQ11'
        }
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """현재가 조회 (실시간 근사)"""
        cache_key = f"price_{symbol}"
        
        # 캐시 확인
        if cache_key in self.cache:
            timestamp, price = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_timeout:
                return price
        
        try:
            # Yahoo Finance로 현재가 조회
            ticker_symbol = self.korean_tickers.get(symbol, f"{symbol}.KS")
            ticker = yf.Ticker(ticker_symbol)
            
            # 최근 데이터 조회
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                
                # 캐시 저장
                self.cache[cache_key] = (datetime.now(), current_price)
                
                logger.debug(f"Current price for {symbol}: {current_price:,.0f}")
                return float(current_price)
            
        except Exception as e:
            logger.warning(f"Failed to get current price for {symbol}: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """과거 데이터 조회"""
        try:
            if symbol in ['KOSPI', 'KOSDAQ']:
                ticker_symbol = self.index_mapping.get(symbol)
            else:
                ticker_symbol = self.korean_tickers.get(symbol, f"{symbol}.KS")
            
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                # 컬럼명 소문자로 변경
                hist.columns = hist.columns.str.lower()
                hist.index.name = 'date'
                
                logger.info(f"Retrieved {len(hist)} days of data for {symbol}")
                return hist
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
        
        return None
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """여러 종목 데이터 한번에 조회"""
        market_data = {}
        
        for symbol in symbols:
            data = self.get_historical_data(symbol)
            if data is not None:
                market_data[symbol] = data
            else:
                logger.warning(f"No data available for {symbol}")
        
        logger.info(f"Retrieved market data for {len(market_data)} symbols")
        return market_data
    
    def get_sector_data(self, sector: str) -> Dict[str, pd.DataFrame]:
        """섹터별 데이터 조회 (Mock)"""
        # 실제로는 섹터별 종목 리스트를 정의하고 데이터를 수집
        sector_stocks = {
            'technology': ['005930', '000660', '035420'],  # 삼성전자, SK하이닉스, 네이버
            'finance': ['055550', '316140', '086790'],      # 신한지주, 우리금융, 하나금융
            'auto': ['005380', '012330'],                   # 현대차, 현대모비스
            'bio': ['068270', '207940', '326030']           # 셀트리온, 삼성바이오, SK바이오팜
        }
        
        if sector not in sector_stocks:
            logger.warning(f"Unknown sector: {sector}")
            return {}
        
        return self.get_market_data(sector_stocks[sector])
    
    def check_connection(self) -> bool:
        """데이터 연결 상태 확인"""
        try:
            # KOSPI 지수로 연결 테스트
            ticker = yf.Ticker("^KS11")
            hist = ticker.history(period="1d")
            return not hist.empty
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


class MockDataProvider:
    """실시간 데이터가 없을 때 사용할 Mock 데이터"""
    
    def __init__(self):
        self.base_prices = {
            '005930': 75000,   # 삼성전자
            '000660': 120000,  # SK하이닉스
            '035420': 180000,  # 네이버
            '055550': 45000,   # 신한지주
            '005380': 200000,  # 현대차
        }
        self.last_update = {}
    
    def generate_mock_price(self, symbol: str) -> float:
        """실시간처럼 보이는 Mock 가격 생성"""
        base_price = self.base_prices.get(symbol, 50000)
        
        # 작은 랜덤 변동 추가 (-1% ~ +1%)
        variation = np.random.normal(0, 0.005)  # 0.5% 표준편차
        current_price = base_price * (1 + variation)
        
        # 가격 업데이트
        self.base_prices[symbol] = current_price
        
        return current_price
    
    def generate_mock_history(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Mock 과거 데이터 생성"""
        base_price = self.base_prices.get(symbol, 50000)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # 랜덤 워크로 가격 생성
        returns = np.random.normal(0.001, 0.02, days)  # 일평균 0.1% 상승, 2% 변동
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1000))  # 최소 1,000원
        
        # OHLCV 데이터 생성
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            volume = int(np.random.lognormal(np.log(1000000), 0.5))
            
            data.append({
                'open': open_price,
                'high': max(high, price, open_price),
                'low': min(low, price, open_price),
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        return df


# 전역 데이터 제공자 인스턴스
free_data = FreeDataProvider()
mock_data = MockDataProvider()


def get_data_provider():
    """사용 가능한 데이터 제공자 반환"""
    if free_data.check_connection():
        logger.info("Using free data provider (Yahoo Finance)")
        return free_data
    else:
        logger.warning("Using mock data provider")
        return mock_data