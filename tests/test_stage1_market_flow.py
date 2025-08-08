"""
Tests for Stage 1: Market Flow Analysis
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from oepnstock.core.stage1_market_flow import MarketFlowAnalyzer
from oepnstock.core.stage1_market_flow.market_flow_analyzer import MarketCondition, SectorAnalysis


@pytest.fixture
def market_flow_analyzer():
    """Market flow analyzer instance"""
    return MarketFlowAnalyzer()


@pytest.fixture
def mock_external_data():
    """Mock external market data"""
    return {
        'us_markets': {
            'sp500_change': 0.015,  # 1.5% up
            'nasdaq_change': 0.020   # 2% up  
        },
        'usd_krw': 1350.0,
        'vix': 20.5
    }


class TestMarketFlowAnalyzer:
    
    @pytest.mark.unit
    def test_analyzer_initialization(self, market_flow_analyzer):
        """Test analyzer initialization"""
        analyzer = market_flow_analyzer
        
        assert analyzer.scoring_weights['index_position'] == 0.4
        assert analyzer.scoring_weights['ma_slope'] == 0.3
        assert analyzer.scoring_weights['volatility'] == 0.3
        assert len(analyzer.market_regimes) == 4
    
    @pytest.mark.unit
    def test_analyze_index(self, market_flow_analyzer, sample_ohlcv_data):
        """Test individual index analysis"""
        analyzer = market_flow_analyzer
        
        result = analyzer._analyze_index(sample_ohlcv_data, "KOSPI")
        
        assert 'current_price' in result
        assert 'ma5' in result
        assert 'ma20' in result
        assert 'ma_slope' in result
        assert 'daily_change' in result
        assert 'position_score' in result
        
        # Check score range
        assert 0 <= result['position_score'] <= 40
        
        # Check MA values are reasonable
        assert result['ma5'] > 0
        assert result['ma20'] > 0
    
    @pytest.mark.unit
    def test_calculate_ma_slope(self, market_flow_analyzer):
        """Test MA slope calculation"""
        analyzer = market_flow_analyzer
        
        # Test upward slope
        upward_ma = pd.Series([100, 101, 102, 103, 104, 105])
        slope_up = analyzer._calculate_ma_slope(upward_ma, lookback=5)
        assert slope_up > 0
        
        # Test downward slope  
        downward_ma = pd.Series([105, 104, 103, 102, 101, 100])
        slope_down = analyzer._calculate_ma_slope(downward_ma, lookback=5)
        assert slope_down < 0
        
        # Test flat slope
        flat_ma = pd.Series([100, 100, 100, 100, 100, 100])
        slope_flat = analyzer._calculate_ma_slope(flat_ma, lookback=5)
        assert abs(slope_flat) < 0.01
    
    @pytest.mark.unit
    def test_analyze_sectors(self, market_flow_analyzer, sample_sector_data):
        """Test sector analysis"""
        analyzer = market_flow_analyzer
        
        result = analyzer._analyze_sectors(sample_sector_data)
        
        assert isinstance(result, list)
        assert len(result) == len(sample_sector_data)
        
        for sector in result:
            assert isinstance(sector, SectorAnalysis)
            assert hasattr(sector, 'name')
            assert hasattr(sector, 'five_day_return')
            assert hasattr(sector, 'volume_ratio')
            assert hasattr(sector, 'status')
            assert sector.status in ['normal', 'hot', 'overheated']
        
        # Check ranking
        returns = [s.five_day_return for s in result]
        assert returns == sorted(returns, reverse=True)
    
    @pytest.mark.unit
    def test_check_sector_overheating(self, market_flow_analyzer):
        """Test sector overheating detection"""
        analyzer = market_flow_analyzer
        
        # Normal sector
        status_normal = analyzer._check_sector_overheating(0.10, 1.5)
        assert status_normal == 'normal'
        
        # Hot sector
        status_hot = analyzer._check_sector_overheating(0.18, 2.0)
        assert status_hot == 'hot'
        
        # Overheated sector
        status_overheated = analyzer._check_sector_overheating(0.25, 3.5)
        assert status_overheated == 'overheated'
    
    @pytest.mark.unit
    def test_calculate_market_score(self, market_flow_analyzer):
        """Test market score calculation"""
        analyzer = market_flow_analyzer
        
        # Mock analysis results
        kospi_analysis = {
            'position_score': 40,  # Maximum
            'ma_slope': 1.0,       # Strong upward
            'daily_change': 0.01   # Positive
        }
        
        kosdaq_analysis = {
            'position_score': 20,  # Half maximum
            'ma_slope': 0.5,       # Moderate upward  
            'daily_change': 0.005  # Small positive
        }
        
        sector_analysis = [
            Mock(status='normal'),
            Mock(status='hot'),
            Mock(status='normal')
        ]
        
        sentiment = {'vix': {'warning': False}}
        
        score = analyzer._calculate_market_score(
            kospi_analysis, kosdaq_analysis, sector_analysis, sentiment
        )
        
        assert 0 <= score <= 100
        assert score > 70  # Should be high given good inputs
    
    @pytest.mark.unit 
    def test_classify_market_regime(self, market_flow_analyzer):
        """Test market regime classification"""
        analyzer = market_flow_analyzer
        
        # Strong uptrend
        kospi_strong = {'ma_slope': 1.0, 'daily_change': 0.02}
        kosdaq_strong = {'ma_slope': 0.8, 'daily_change': 0.015}
        regime = analyzer._classify_market_regime(85, kospi_strong, kosdaq_strong)
        assert regime == 'strong_uptrend'
        
        # Weak uptrend
        kospi_weak = {'ma_slope': 0.2, 'daily_change': 0.005}
        kosdaq_weak = {'ma_slope': 0.1, 'daily_change': 0.003}
        regime = analyzer._classify_market_regime(75, kospi_weak, kosdaq_weak)
        assert regime == 'weak_uptrend'
        
        # Sideways
        kospi_side = {'ma_slope': -0.1, 'daily_change': 0.002}
        kosdaq_side = {'ma_slope': 0.05, 'daily_change': -0.001}
        regime = analyzer._classify_market_regime(65, kospi_side, kosdaq_side)
        assert regime == 'sideways'
        
        # Downtrend
        kospi_down = {'ma_slope': -0.5, 'daily_change': -0.02}
        kosdaq_down = {'ma_slope': -0.8, 'daily_change': -0.025}
        regime = analyzer._classify_market_regime(40, kospi_down, kosdaq_down)
        assert regime == 'downtrend'
    
    @pytest.mark.unit
    def test_generate_warnings(self, market_flow_analyzer):
        """Test warning generation"""
        analyzer = market_flow_analyzer
        
        # Low market score
        warnings1 = analyzer._generate_warnings(
            60, [], {'vix': {'warning': False}}
        )
        assert any('below threshold' in w for w in warnings1)
        
        # Overheated sectors
        overheated_sectors = [Mock(name='tech', status='overheated')]
        warnings2 = analyzer._generate_warnings(
            80, overheated_sectors, {'vix': {'warning': False}}
        )
        assert any('Overheated sectors' in w for w in warnings2)
        
        # High VIX
        warnings3 = analyzer._generate_warnings(
            80, [], {'vix': {'warning': True}}
        )
        assert any('High VIX' in w for w in warnings3)
    
    @pytest.mark.integration
    def test_analyze_market_flow_complete(self, market_flow_analyzer, 
                                        sample_ohlcv_data, sample_sector_data,
                                        mock_external_data):
        """Test complete market flow analysis"""
        analyzer = market_flow_analyzer
        
        # Ensure data has sufficient length
        assert len(sample_ohlcv_data) >= 60
        
        result = analyzer.analyze_market_flow(
            kospi_data=sample_ohlcv_data,
            kosdaq_data=sample_ohlcv_data,
            sector_data=sample_sector_data,
            external_data=mock_external_data
        )
        
        # Check result structure
        assert isinstance(result, MarketCondition)
        assert hasattr(result, 'score')
        assert hasattr(result, 'regime')
        assert hasattr(result, 'tradable')
        assert hasattr(result, 'warnings')
        
        # Check value ranges
        assert 0 <= result.score <= 100
        assert result.regime in ['strong_uptrend', 'weak_uptrend', 'sideways', 'downtrend']
        assert isinstance(result.tradable, bool)
        assert isinstance(result.warnings, list)
        
        # Check nested data
        assert 'kospi' in result.index_position
        assert 'kosdaq' in result.index_position
        assert len(result.leading_sectors) <= 3
        assert 'us_markets' in result.market_sentiment
    
    @pytest.mark.unit
    def test_get_regime_parameters(self, market_flow_analyzer):
        """Test regime parameter retrieval"""
        analyzer = market_flow_analyzer
        
        # Test each regime
        strong_params = analyzer.get_regime_parameters('strong_uptrend')
        assert strong_params['position_multiplier'] == 1.2
        assert strong_params['min_score'] == 80
        
        weak_params = analyzer.get_regime_parameters('weak_uptrend')
        assert weak_params['position_multiplier'] == 1.0
        
        sideways_params = analyzer.get_regime_parameters('sideways')
        assert sideways_params['position_multiplier'] == 0.8
        
        down_params = analyzer.get_regime_parameters('downtrend')
        assert down_params['position_multiplier'] == 0.5
        assert down_params['min_score'] == 90  # Very strict
        
        # Test invalid regime
        default_params = analyzer.get_regime_parameters('invalid_regime')
        assert default_params == analyzer.market_regimes['sideways']
    
    @pytest.mark.unit
    def test_insufficient_data_error(self, market_flow_analyzer):
        """Test error handling for insufficient data"""
        analyzer = market_flow_analyzer
        
        # Create insufficient data (less than ma_long period)
        short_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107], 
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.analyze_market_flow(
                kospi_data=short_data,
                kosdaq_data=short_data,
                sector_data={'test': short_data}
            )
    
    @pytest.mark.unit
    def test_analyze_market_sentiment(self, market_flow_analyzer,
                                    sample_ohlcv_data, mock_external_data):
        """Test market sentiment analysis"""
        analyzer = market_flow_analyzer
        
        sentiment = analyzer._analyze_market_sentiment(
            sample_ohlcv_data, sample_ohlcv_data, mock_external_data
        )
        
        # Check US markets data
        assert 'us_markets' in sentiment
        assert sentiment['us_markets']['sp500_change'] == 0.015
        assert sentiment['us_markets']['warning'] == False
        
        # Check VIX data
        assert 'vix' in sentiment
        assert sentiment['vix']['level'] == 20.5
        assert sentiment['vix']['warning'] == False  # Below 30
        
        # Check USD/KRW
        assert 'usd_krw' in sentiment
        
    @pytest.mark.unit
    def test_score_adjustments(self, market_flow_analyzer):
        """Test score adjustments for various conditions"""
        analyzer = market_flow_analyzer
        
        base_analysis = {
            'position_score': 30,
            'ma_slope': 0.3,
            'daily_change': 0.01
        }
        
        # Test with overheated sectors
        overheated_sectors = [Mock(status='overheated') for _ in range(3)]
        sentiment_high_vix = {'vix': {'warning': True}}
        
        score = analyzer._calculate_market_score(
            base_analysis, base_analysis, overheated_sectors, sentiment_high_vix
        )
        
        # Score should be reduced due to overheated sectors and high VIX
        base_score = analyzer._calculate_market_score(
            base_analysis, base_analysis, [], {}
        )
        
        assert score < base_score