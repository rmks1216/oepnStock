"""
Technical analysis calculations and utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

from ..config import config
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class TechnicalIndicators:
    """Container for technical indicator values"""
    rsi: float
    macd: Dict[str, float]  # line, signal, histogram
    stochastic: Dict[str, float]  # k, d
    bollinger_bands: Dict[str, float]  # upper, middle, lower
    atr: float
    moving_averages: Dict[str, float]  # ma5, ma10, ma20, ma60


class PriceCalculations:
    """
    Price-related calculations for technical analysis
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
        """Calculate returns for given periods"""
        return prices.pct_change(periods=periods)
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns"""
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility (annualized)"""
        returns = PriceCalculations.calculate_returns(prices)
        return returns.rolling(window).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """
        Calculate support and resistance levels using local extrema
        
        Args:
            data: OHLCV DataFrame
            window: Window for finding local extrema
            
        Returns:
            Dict with 'support' and 'resistance' lists
        """
        highs = data['high']
        lows = data['low']
        
        # Find local maxima (resistance)
        resistance_levels = []
        for i in range(window, len(highs) - window):
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                resistance_levels.append(highs.iloc[i])
        
        # Find local minima (support)
        support_levels = []
        for i in range(window, len(lows) - window):
            if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                support_levels.append(lows.iloc[i])
        
        return {
            'support': sorted(list(set(support_levels))),
            'resistance': sorted(list(set(resistance_levels)), reverse=True)
        }
    
    @staticmethod
    def calculate_fibonacci_levels(high_price: float, low_price: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high_price - low_price
        
        return {
            '0.0': high_price,
            '23.6': high_price - 0.236 * diff,
            '38.2': high_price - 0.382 * diff,
            '50.0': high_price - 0.5 * diff,
            '61.8': high_price - 0.618 * diff,
            '78.6': high_price - 0.786 * diff,
            '100.0': low_price
        }
    
    @staticmethod
    def calculate_pivot_points(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate pivot points for support/resistance"""
        if len(data) == 0:
            return {}
            
        last_candle = data.iloc[-1]
        high, low, close = last_candle['high'], last_candle['low'], last_candle['close']
        
        pivot = (high + low + close) / 3
        
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }


class TechnicalAnalysis:
    """
    Technical analysis indicators and calculations
    """
    
    def __init__(self):
        self.config = config.technical
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators"""
        if len(data) < 60:  # Need enough data for calculations
            logger.warning("Insufficient data for full technical analysis")
            return self._get_default_indicators()
        
        try:
            rsi = self.calculate_rsi(data['close'])
            macd = self.calculate_macd(data['close'])
            stochastic = self.calculate_stochastic(data)
            bollinger = self.calculate_bollinger_bands(data['close'])
            atr = self.calculate_atr(data)
            mas = self.calculate_moving_averages(data['close'])
            
            return TechnicalIndicators(
                rsi=rsi,
                macd=macd,
                stochastic=stochastic,
                bollinger_bands=bollinger,
                atr=atr,
                moving_averages=mas
            )
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._get_default_indicators()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def calculate_macd(self, prices: pd.Series, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow + signal:
            return {'line': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'line': float(macd_line.iloc[-1]),
            'signal': float(signal_line.iloc[-1]),
            'histogram': float(histogram.iloc[-1])
        }
    
    def calculate_stochastic(self, data: pd.DataFrame, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Calculate Stochastic Oscillator"""
        if len(data) < k_period:
            return {'k': 50.0, 'd': 50.0}
        
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': float(k_percent.iloc[-1]) if not np.isnan(k_percent.iloc[-1]) else 50.0,
            'd': float(d_percent.iloc[-1]) if not np.isnan(d_percent.iloc[-1]) else 50.0
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, 
                                period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = float(prices.iloc[-1])
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98
            }
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': float(upper_band.iloc[-1]),
            'middle': float(sma.iloc[-1]),
            'lower': float(lower_band.iloc[-1])
        }
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(data) < period:
            return 0.02  # Default 2% volatility
        
        # True Range calculation
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        
        # Return as ratio to current price
        current_price = data['close'].iloc[-1]
        atr_value = atr.iloc[-1]
        
        return float(atr_value / current_price) if current_price > 0 else 0.02
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate multiple moving averages"""
        mas = {}
        periods = [5, 10, 20, 60]
        
        for period in periods:
            if len(prices) >= period:
                ma_value = prices.rolling(window=period).mean().iloc[-1]
                mas[f'ma{period}'] = float(ma_value) if not np.isnan(ma_value) else float(prices.iloc[-1])
            else:
                mas[f'ma{period}'] = float(prices.iloc[-1])
        
        return mas
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based indicators"""
        if len(data) < 20:
            return {'vwap': 0.0, 'volume_ratio': 1.0, 'obv': 0.0}
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).sum() / data['volume'].sum()
        
        # Volume Ratio
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # OBV (On Balance Volume) - simplified
        price_change = data['close'].diff()
        obv_series = (np.sign(price_change) * data['volume']).cumsum()
        obv = obv_series.iloc[-1]
        
        return {
            'vwap': float(vwap),
            'volume_ratio': float(volume_ratio),
            'obv': float(obv)
        }
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators"""
        if len(data) < 14:
            return {'roc': 0.0, 'momentum': 0.0, 'cci': 0.0}
        
        prices = data['close']
        
        # Rate of Change (ROC)
        roc = ((prices.iloc[-1] - prices.iloc[-12]) / prices.iloc[-12]) * 100
        
        # Momentum
        momentum = prices.iloc[-1] - prices.iloc[-10]
        
        # Commodity Channel Index (CCI) - simplified
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=14).mean()
        mad = typical_price.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price.iloc[-1] - sma_tp.iloc[-1]) / (0.015 * mad.iloc[-1])
        
        return {
            'roc': float(roc),
            'momentum': float(momentum / prices.iloc[-1] * 100),  # As percentage
            'cci': float(cci) if not np.isnan(cci) else 0.0
        }
    
    def identify_chart_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify basic chart patterns"""
        if len(data) < 50:
            return {'patterns': [], 'confidence': 0.0}
        
        patterns = []
        
        # Double bottom pattern (simplified)
        lows = data['low'].rolling(window=5, center=True).min()
        recent_lows = lows.tail(20)
        
        if len(recent_lows) >= 10:
            # Find two significant lows
            min_indices = []
            for i in range(2, len(recent_lows) - 2):
                if (recent_lows.iloc[i] < recent_lows.iloc[i-2:i].min() and 
                    recent_lows.iloc[i] < recent_lows.iloc[i+1:i+3].min()):
                    min_indices.append(i)
            
            if len(min_indices) >= 2:
                # Check if the lows are similar (within 3%)
                low1, low2 = recent_lows.iloc[min_indices[-2]], recent_lows.iloc[min_indices[-1]]
                if abs(low1 - low2) / min(low1, low2) < 0.03:
                    patterns.append({
                        'pattern': 'double_bottom',
                        'confidence': 0.7,
                        'support_level': min(low1, low2)
                    })
        
        # Head and shoulders (very simplified)
        highs = data['high'].rolling(window=5, center=True).max()
        recent_highs = highs.tail(30)
        
        if len(recent_highs) >= 15:
            max_indices = []
            for i in range(2, len(recent_highs) - 2):
                if (recent_highs.iloc[i] > recent_highs.iloc[i-2:i].max() and 
                    recent_highs.iloc[i] > recent_highs.iloc[i+1:i+3].max()):
                    max_indices.append(i)
            
            if len(max_indices) >= 3:
                # Check for head and shoulders pattern
                if (recent_highs.iloc[max_indices[-2]] > recent_highs.iloc[max_indices[-3]] and
                    recent_highs.iloc[max_indices[-2]] > recent_highs.iloc[max_indices[-1]]):
                    patterns.append({
                        'pattern': 'head_and_shoulders',
                        'confidence': 0.6,
                        'resistance_level': recent_highs.iloc[max_indices[-2]]
                    })
        
        return {
            'patterns': patterns,
            'confidence': max([p['confidence'] for p in patterns]) if patterns else 0.0
        }
    
    def _get_default_indicators(self) -> TechnicalIndicators:
        """Return default indicator values when calculation fails"""
        return TechnicalIndicators(
            rsi=50.0,
            macd={'line': 0.0, 'signal': 0.0, 'histogram': 0.0},
            stochastic={'k': 50.0, 'd': 50.0},
            bollinger_bands={'upper': 0.0, 'middle': 0.0, 'lower': 0.0},
            atr=0.02,
            moving_averages={'ma5': 0.0, 'ma10': 0.0, 'ma20': 0.0, 'ma60': 0.0}
        )


class PatternRecognition:
    """
    Advanced pattern recognition for candlesticks and chart patterns
    """
    
    @staticmethod
    def identify_candlestick_patterns(data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify candlestick patterns in recent data"""
        if len(data) < 5:
            return []
        
        patterns = []
        recent_data = data.tail(5)  # Look at last 5 candles
        
        for i in range(1, len(recent_data)):
            current = recent_data.iloc[i]
            previous = recent_data.iloc[i-1]
            
            pattern = PatternRecognition._analyze_single_candle(current, previous)
            if pattern:
                patterns.append({
                    'date': current.name,
                    'pattern': pattern['name'],
                    'strength': pattern['strength'],
                    'bullish': pattern['bullish']
                })
        
        return patterns
    
    @staticmethod
    def _analyze_single_candle(current: pd.Series, previous: pd.Series) -> Optional[Dict]:
        """Analyze single candlestick pattern"""
        # Current candle components
        open_price = current['open']
        high_price = current['high']
        low_price = current['low']
        close_price = current['close']
        
        body = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        if total_range == 0:
            return None
        
        # Body and shadow ratios
        body_ratio = body / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        
        # Hammer pattern
        if (lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1 and
            body_ratio < 0.3):
            return {
                'name': 'hammer',
                'strength': min(lower_shadow_ratio * 2, 1.0),
                'bullish': True
            }
        
        # Shooting star pattern
        if (upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1 and
            body_ratio < 0.3):
            return {
                'name': 'shooting_star', 
                'strength': min(upper_shadow_ratio * 2, 1.0),
                'bullish': False
            }
        
        # Doji pattern
        if body_ratio < 0.1:
            return {
                'name': 'doji',
                'strength': 1.0 - body_ratio * 10,
                'bullish': None  # Neutral, depends on context
            }
        
        # Engulfing patterns
        prev_body = abs(previous['close'] - previous['open'])
        if (body > prev_body * 1.1 and  # Current body engulfs previous
            ((close_price > open_price and previous['close'] < previous['open']) or  # Bullish engulfing
             (close_price < open_price and previous['close'] > previous['open']))):  # Bearish engulfing
            
            is_bullish = close_price > open_price
            return {
                'name': 'engulfing',
                'strength': min(body / prev_body / 2, 1.0),
                'bullish': is_bullish
            }
        
        return None
    
    @staticmethod
    def calculate_pattern_reliability(patterns: List[Dict], market_trend: str) -> float:
        """Calculate overall pattern reliability based on market context"""
        if not patterns:
            return 0.0
        
        total_reliability = 0.0
        weight_sum = 0.0
        
        for pattern in patterns:
            base_reliability = pattern['strength']
            
            # Adjust based on market trend
            if market_trend == 'uptrend' and pattern.get('bullish'):
                base_reliability *= 1.2
            elif market_trend == 'downtrend' and not pattern.get('bullish'):
                base_reliability *= 1.2
            elif market_trend == 'sideways':
                base_reliability *= 1.1
            else:
                base_reliability *= 0.8  # Against trend
            
            total_reliability += base_reliability
            weight_sum += 1.0
        
        return min(total_reliability / weight_sum, 1.0) if weight_sum > 0 else 0.0