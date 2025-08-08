"""
Pytest configuration and fixtures for oepnStock testing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List
from unittest.mock import Mock, MagicMock

from oepnstock.config import config
from oepnstock.utils import get_logger


@pytest.fixture(scope="session")
def test_config():
    """Test configuration with safe values"""
    test_conf = config
    # Override with test-safe values
    test_conf.trading.max_positions = 3
    test_conf.trading.initial_risk_per_trade = 0.01  # 1% for testing
    test_conf.enable_paper_trading = True
    return test_conf


@pytest.fixture
def logger():
    """Test logger instance"""
    return get_logger("test")


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    base_price = 50000
    price_data = []
    current_price = base_price
    
    np.random.seed(42)  # For reproducible tests
    
    for i in range(100):
        # Random walk with slight upward bias
        change = np.random.normal(0.005, 0.02)  # 0.5% average gain, 2% volatility
        current_price = current_price * (1 + change)
        
        # OHLC for the day
        open_price = current_price
        high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
        low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
        close_price = low_price + (high_price - low_price) * np.random.random()
        
        volume = int(np.random.normal(1000000, 300000))  # Average 1M volume
        
        price_data.append({
            'open': open_price,
            'high': high_price, 
            'low': low_price,
            'close': close_price,
            'volume': max(volume, 100000)  # Minimum volume
        })
        
        current_price = close_price
    
    df = pd.DataFrame(price_data, index=dates)
    return df


@pytest.fixture 
def sample_market_data():
    """Generate sample market index data"""
    return {
        'kospi': sample_ohlcv_data(),
        'kosdaq': sample_ohlcv_data()
    }


@pytest.fixture
def sample_sector_data():
    """Generate sample sector data"""
    sectors = ['technology', 'finance', 'manufacturing', 'consumer']
    sector_data = {}
    
    for sector in sectors:
        # Use different seeds for variation
        np.random.seed(hash(sector) % 1000)
        sector_data[sector] = sample_ohlcv_data()
    
    return sector_data


@pytest.fixture
def mock_api_client():
    """Mock API client for external data sources"""
    mock_client = MagicMock()
    
    # Mock responses
    mock_client.get_stock_data.return_value = sample_ohlcv_data()
    mock_client.get_market_data.return_value = sample_market_data()
    mock_client.is_connected.return_value = True
    
    return mock_client


@pytest.fixture
def sample_support_levels():
    """Generate sample support level data"""
    return [
        {
            'price': 48000,
            'strength': 0.8,
            'touch_count': 3,
            'support_type': 'horizontal',
            'in_cluster': True
        },
        {
            'price': 49500,
            'strength': 0.6, 
            'touch_count': 2,
            'support_type': 'ma20',
            'in_cluster': False
        },
        {
            'price': 50000,
            'strength': 0.7,
            'touch_count': 2,
            'support_type': 'round_figure',
            'in_cluster': True
        }
    ]


@pytest.fixture
def market_conditions():
    """Sample market conditions for testing"""
    return {
        'strong_uptrend': {
            'score': 85,
            'regime': 'strong_uptrend',
            'tradable': True,
            'warnings': []
        },
        'weak_downtrend': {
            'score': 45,
            'regime': 'downtrend', 
            'tradable': False,
            'warnings': ['Market score below threshold']
        },
        'sideways': {
            'score': 65,
            'regime': 'sideways',
            'tradable': False,  # Below 70 threshold
            'warnings': []
        }
    }


@pytest.fixture
def trading_session_times():
    """Korean market trading session times for testing"""
    base_date = date.today()
    return {
        'pre_opening': datetime.combine(base_date, datetime.strptime('08:45', '%H:%M').time()),
        'opening_auction': datetime.combine(base_date, datetime.strptime('09:05', '%H:%M').time()),
        'morning_trend': datetime.combine(base_date, datetime.strptime('10:30', '%H:%M').time()),
        'lunch_break': datetime.combine(base_date, datetime.strptime('12:00', '%H:%M').time()),
        'afternoon_trend': datetime.combine(base_date, datetime.strptime('14:00', '%H:%M').time()),
        'closing': datetime.combine(base_date, datetime.strptime('15:10', '%H:%M').time()),
        'after_hours': datetime.combine(base_date, datetime.strptime('16:00', '%H:%M').time())
    }


@pytest.fixture
def portfolio_data():
    """Sample portfolio data for testing"""
    return [
        {
            'symbol': 'TEST001',
            'quantity': 100,
            'avg_price': 50000,
            'current_price': 52000,
            'weight': 0.15,
            'sector': 'technology'
        },
        {
            'symbol': 'TEST002', 
            'quantity': 200,
            'avg_price': 25000,
            'current_price': 24000,
            'weight': 0.12,
            'sector': 'finance'
        },
        {
            'symbol': 'TEST003',
            'quantity': 50,
            'avg_price': 100000,
            'current_price': 105000,
            'weight': 0.13,
            'sector': 'technology'  # Same sector as TEST001
        }
    ]


# Test data generators
def generate_price_series(length: int = 100, start_price: float = 50000, 
                         volatility: float = 0.02, trend: float = 0.001) -> pd.Series:
    """Generate realistic price series for testing"""
    prices = [start_price]
    
    for i in range(1, length):
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 100))  # Minimum price of 100
    
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    return pd.Series(prices, index=dates)


def generate_volume_series(length: int = 100, avg_volume: int = 1000000,
                          volatility: float = 0.3) -> pd.Series:
    """Generate realistic volume series for testing"""
    volumes = []
    
    for i in range(length):
        volume = int(np.random.lognormal(np.log(avg_volume), volatility))
        volumes.append(max(volume, 10000))  # Minimum volume
    
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    return pd.Series(volumes, index=dates)


# Markers for different test categories
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
]