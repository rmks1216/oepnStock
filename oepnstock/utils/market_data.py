"""
Market data management and utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

from ..config import config
from .logger import get_logger
from .korean_market import KoreanMarketUtils

logger = get_logger(__name__)


@dataclass
class MarketDataPoint:
    """Single market data point"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


@dataclass
class MarketSnapshot:
    """Market snapshot at a point in time"""
    timestamp: datetime
    kospi_level: float
    kosdaq_level: float
    usd_krw: float
    vix: Optional[float]
    sector_performances: Dict[str, float]
    top_gainers: List[Dict]
    top_losers: List[Dict]


class MarketDataManager:
    """
    Market data management and caching system
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 300  # 5 minutes default TTL
        
        # Data source priorities (실제 API 연동시 사용)
        self.data_sources = [
            'korea_investment',  # Primary
            'kiwoom',           # Secondary  
            'yahoo_finance',    # Fallback
            'mock'              # Development/testing
        ]
        
        # Real-time data subscriptions
        self.subscriptions = set()
        self.realtime_callbacks = {}
        
    def get_stock_data(self, symbol: str, 
                      start_date: Optional[date] = None,
                      end_date: Optional[date] = None,
                      interval: str = '1d') -> pd.DataFrame:
        """
        Get historical stock data
        
        Args:
            symbol: Stock symbol (Korean format)
            start_date: Start date for data
            end_date: End date for data  
            interval: Data interval ('1d', '1h', '5m')
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached data for {symbol}")
            return self.cache[cache_key]
        
        try:
            # Try data sources in priority order
            for source in self.data_sources:
                try:
                    data = self._fetch_from_source(source, symbol, start_date, end_date, interval)
                    if data is not None and not data.empty:
                        self._update_cache(cache_key, data)
                        return data
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source}: {e}")
                    continue
            
            # All sources failed - return mock data for development
            logger.warning(f"All data sources failed for {symbol}, returning mock data")
            return self._generate_mock_data(symbol, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def get_market_snapshot(self) -> MarketSnapshot:
        """Get current market snapshot"""
        cache_key = "market_snapshot"
        
        if self._is_cache_valid(cache_key, ttl=60):  # 1 minute TTL
            return self.cache[cache_key]
        
        try:
            # In real implementation, this would fetch from market data APIs
            snapshot = self._generate_mock_market_snapshot()
            self._update_cache(cache_key, snapshot, ttl=60)
            return snapshot
            
        except Exception as e:
            logger.error(f"Error fetching market snapshot: {e}")
            # Return fallback snapshot
            return MarketSnapshot(
                timestamp=datetime.now(),
                kospi_level=2500.0,
                kosdaq_level=800.0,
                usd_krw=1350.0,
                vix=20.0,
                sector_performances={},
                top_gainers=[],
                top_losers=[]
            )
    
    def get_sector_data(self, sector: str, 
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> pd.DataFrame:
        """Get sector index data"""
        cache_key = f"sector_{sector}_{start_date}_{end_date}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        # Mock sector data for development
        data = self._generate_mock_sector_data(sector, start_date, end_date)
        self._update_cache(cache_key, data)
        return data
    
    def get_real_time_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time price data"""
        try:
            # In production, this would connect to real-time data feed
            # For now, return mock real-time data
            current_time = datetime.now()
            
            # Simulate market hours check
            if not KoreanMarketUtils.is_market_open(current_time):
                return None
            
            # Mock real-time data
            base_price = 50000 + hash(symbol) % 10000  # Deterministic base price
            random_change = np.random.normal(0, 0.01)  # 1% volatility
            
            current_price = base_price * (1 + random_change)
            volume = int(np.random.normal(100000, 30000))
            
            return {
                'symbol': symbol,
                'price': current_price,
                'change': random_change,
                'change_percent': random_change * 100,
                'volume': max(volume, 10000),
                'timestamp': current_time,
                'bid': current_price * 0.999,
                'ask': current_price * 1.001,
                'bid_size': np.random.randint(100, 1000),
                'ask_size': np.random.randint(100, 1000)
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {e}")
            return None
    
    def subscribe_real_time(self, symbol: str, callback: callable) -> bool:
        """Subscribe to real-time price updates"""
        try:
            self.subscriptions.add(symbol)
            self.realtime_callbacks[symbol] = callback
            
            # In production, this would establish WebSocket connection
            logger.info(f"Subscribed to real-time data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to {symbol}: {e}")
            return False
    
    def unsubscribe_real_time(self, symbol: str) -> bool:
        """Unsubscribe from real-time updates"""
        try:
            self.subscriptions.discard(symbol)
            self.realtime_callbacks.pop(symbol, None)
            
            logger.info(f"Unsubscribed from real-time data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol}: {e}")
            return False
    
    async def fetch_multiple_stocks(self, symbols: List[str],
                                  start_date: Optional[date] = None,
                                  end_date: Optional[date] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks concurrently"""
        tasks = []
        
        async def fetch_single(symbol: str) -> Tuple[str, pd.DataFrame]:
            try:
                # Run synchronous data fetch in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    data = await loop.run_in_executor(
                        executor, self.get_stock_data, symbol, start_date, end_date
                    )
                return symbol, data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, pd.DataFrame()
        
        # Create tasks for all symbols
        tasks = [fetch_single(symbol) for symbol in symbols]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        stock_data = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            
            symbol, data = result
            stock_data[symbol] = data
        
        return stock_data
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Validate data quality and completeness"""
        if data.empty:
            return {'valid': False, 'issues': ['Empty dataset']}
        
        issues = []
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for missing values
        null_counts = data[required_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check for impossible values
        if (data['high'] < data['low']).any():
            issues.append("High prices below low prices detected")
        
        if (data['volume'] < 0).any():
            issues.append("Negative volume values detected")
        
        # Check for extreme outliers
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                pct_change = data[col].pct_change().abs()
                extreme_changes = (pct_change > 0.3).sum()  # >30% daily change
                if extreme_changes > len(data) * 0.05:  # More than 5% of data
                    issues.append(f"Excessive extreme price changes in {col}")
        
        # Check data continuity (gaps)
        if len(data) > 1:
            date_gaps = pd.to_datetime(data.index).to_series().diff()
            large_gaps = (date_gaps > pd.Timedelta(days=7)).sum()  # Week+ gaps
            if large_gaps > 0:
                issues.append(f"{large_gaps} large date gaps detected")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'data_points': len(data),
            'date_range': (data.index[0], data.index[-1]) if len(data) > 0 else None,
            'completeness': 1 - (data.isnull().sum().sum() / data.size)
        }
    
    def resample_data(self, data: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Resample data to different time intervals"""
        try:
            # Define resampling rules
            rules = {
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Filter rules to available columns
            available_rules = {col: rule for col, rule in rules.items() if col in data.columns}
            
            # Resample
            resampled = data.resample(target_interval).agg(available_rules)
            
            # Drop rows with NaN (incomplete periods)
            resampled = resampled.dropna()
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return data
    
    def calculate_derived_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate common derived indicators"""
        derived = data.copy()
        
        try:
            # Price-based indicators
            derived['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
            derived['price_range'] = data['high'] - data['low']
            derived['price_change'] = data['close'].diff()
            derived['price_change_pct'] = data['close'].pct_change()
            
            # Volume indicators
            if 'volume' in data.columns:
                derived['volume_ma'] = data['volume'].rolling(window=20).mean()
                derived['volume_ratio'] = data['volume'] / derived['volume_ma']
                
                # VWAP
                derived['vwap'] = (derived['typical_price'] * data['volume']).cumsum() / data['volume'].cumsum()
            
            # Volatility measures
            derived['volatility'] = data['close'].pct_change().rolling(window=20).std()
            
            # Gap analysis
            derived['gap_up'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
            derived['gap_up_significant'] = derived['gap_up'] > 0.02  # 2%+ gaps
            
            # Support/resistance touches (simplified)
            window = 20
            derived['resistance_test'] = (
                data['high'] >= data['high'].rolling(window=window, center=True).max()
            ).astype(int)
            
            derived['support_test'] = (
                data['low'] <= data['low'].rolling(window=window, center=True).min()
            ).astype(int)
            
            return derived
            
        except Exception as e:
            logger.error(f"Error calculating derived indicators: {e}")
            return data
    
    # Private helper methods
    
    def _is_cache_valid(self, key: str, ttl: Optional[int] = None) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        
        if key not in self.cache_expiry:
            return True  # No expiry set
        
        expiry_time = self.cache_expiry[key]
        return datetime.now() < expiry_time
    
    def _update_cache(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Update cache with new data"""
        self.cache[key] = data
        
        if ttl is None:
            ttl = self.cache_ttl
        
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=ttl)
    
    def _fetch_from_source(self, source: str, symbol: str,
                          start_date: Optional[date], end_date: Optional[date],
                          interval: str) -> Optional[pd.DataFrame]:
        """Fetch data from specific source"""
        if source == 'mock':
            return self._generate_mock_data(symbol, start_date, end_date)
        
        # Real API implementations would go here
        elif source == 'korea_investment':
            # Korea Investment Securities API
            return self._fetch_from_korea_investment(symbol, start_date, end_date, interval)
        
        elif source == 'kiwoom':
            # Kiwoom Open API
            return self._fetch_from_kiwoom(symbol, start_date, end_date, interval)
        
        elif source == 'yahoo_finance':
            # Yahoo Finance (limited Korean stocks)
            return self._fetch_from_yahoo(symbol, start_date, end_date, interval)
        
        return None
    
    def _generate_mock_data(self, symbol: str, 
                           start_date: Optional[date] = None,
                           end_date: Optional[date] = None) -> pd.DataFrame:
        """Generate mock stock data for development/testing"""
        if end_date is None:
            end_date = date.today()
        
        if start_date is None:
            start_date = end_date - timedelta(days=100)
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Filter to trading days only (rough approximation)
        trading_days = [d for d in date_range if d.weekday() < 5]
        
        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)  # Deterministic based on symbol
        
        n_days = len(trading_days)
        base_price = 50000 + (hash(symbol) % 50000)  # Base price 50k-100k
        
        # Random walk with drift
        daily_returns = np.random.normal(0.001, 0.02, n_days)  # Slight upward bias
        price_path = base_price * np.exp(np.cumsum(daily_returns))
        
        # Generate OHLCV data
        data = []
        for i, date in enumerate(trading_days):
            close_price = price_path[i]
            
            # Generate realistic OHLC
            daily_volatility = np.random.normal(0.015, 0.005)  # 1.5% avg daily range
            high_low_range = close_price * abs(daily_volatility)
            
            high_price = close_price + np.random.uniform(0, high_low_range)
            low_price = close_price - np.random.uniform(0, high_low_range)
            
            # Open price influenced by previous close and some randomness
            if i == 0:
                open_price = close_price * np.random.uniform(0.995, 1.005)
            else:
                gap = np.random.normal(0, 0.005)  # Small overnight gaps
                open_price = price_path[i-1] * (1 + gap)
            
            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume (correlated with price movement)
            base_volume = 1000000 + (hash(symbol + str(i)) % 2000000)
            price_change = abs(close_price - (price_path[i-1] if i > 0 else close_price))
            volume_multiplier = 1 + (price_change / close_price) * 5  # Higher volume on big moves
            volume = int(base_volume * volume_multiplier)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=trading_days)
        return df
    
    def _generate_mock_market_snapshot(self) -> MarketSnapshot:
        """Generate mock market snapshot"""
        return MarketSnapshot(
            timestamp=datetime.now(),
            kospi_level=2500 + np.random.normal(0, 50),
            kosdaq_level=800 + np.random.normal(0, 20),
            usd_krw=1350 + np.random.normal(0, 10),
            vix=20 + np.random.normal(0, 5),
            sector_performances={
                'technology': np.random.normal(0.005, 0.02),
                'finance': np.random.normal(0.002, 0.015),
                'manufacturing': np.random.normal(0.001, 0.018),
                'consumer': np.random.normal(0.003, 0.012)
            },
            top_gainers=[
                {'symbol': 'GAINER1', 'change_percent': 15.2},
                {'symbol': 'GAINER2', 'change_percent': 12.8},
                {'symbol': 'GAINER3', 'change_percent': 11.5}
            ],
            top_losers=[
                {'symbol': 'LOSER1', 'change_percent': -8.3},
                {'symbol': 'LOSER2', 'change_percent': -7.1},
                {'symbol': 'LOSER3', 'change_percent': -6.8}
            ]
        )
    
    def _generate_mock_sector_data(self, sector: str,
                                  start_date: Optional[date] = None,
                                  end_date: Optional[date] = None) -> pd.DataFrame:
        """Generate mock sector index data"""
        # Reuse stock data generation logic with sector-specific parameters
        return self._generate_mock_data(f"SECTOR_{sector}", start_date, end_date)
    
    # API-specific methods (placeholders for real implementation)
    
    def _fetch_from_korea_investment(self, symbol: str, start_date: Optional[date],
                                   end_date: Optional[date], interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Korea Investment Securities API"""
        # TODO: Implement real API call
        logger.info(f"Would fetch {symbol} from Korea Investment API")
        return None
    
    def _fetch_from_kiwoom(self, symbol: str, start_date: Optional[date],
                          end_date: Optional[date], interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Kiwoom Open API"""
        # TODO: Implement real API call
        logger.info(f"Would fetch {symbol} from Kiwoom API")
        return None
    
    def _fetch_from_yahoo(self, symbol: str, start_date: Optional[date],
                         end_date: Optional[date], interval: str) -> Optional[pd.DataFrame]:
        """Fetch from Yahoo Finance"""
        # TODO: Implement real API call
        logger.info(f"Would fetch {symbol} from Yahoo Finance")
        return None
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        self.cache_expiry.clear()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache),
            'expired_entries': sum(1 for key in self.cache_expiry 
                                 if not self._is_cache_valid(key)),
            'memory_usage_mb': sum(
                df.memory_usage(deep=True).sum() if isinstance(df, pd.DataFrame) else 0
                for df in self.cache.values()
            ) / (1024 * 1024)
        }