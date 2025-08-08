# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **oepnStock** project - a documentation repository for implementing a sophisticated 4-stage checklist strategy for Korean stock market trading. The repository contains comprehensive guides for building an algorithmic trading system optimized for KOSPI/KOSDAQ markets.

## Architecture Strategy

### Core 4-Stage Trading Framework

The system should implement these sequential stages:

1. **Market Flow Analysis (시장의 흐름 파악)**: Market-wide trend scoring with 70+ point threshold for trade eligibility
2. **Support Level Detection (매수 후보 지점)**: Advanced support clustering algorithm identifying overlapping zones  
3. **Signal Confirmation (진짜 반등 신호)**: Dynamic weighted validation with market condition adjustments
4. **Risk Management (리스크 관리)**: Comprehensive position sizing and exit strategies

### Essential Module Implementation Priority

**Phase 1 (Critical - 1-2 weeks)**:
- `FundamentalEventFilter`: News/disclosure filtering system
- `PortfolioConcentrationManager`: Risk diversification with correlation limits
- `GapTradingStrategy`: Pre-market gap analysis and response

**Phase 2 (Performance - 3-4 weeks)**:  
- `VolatilityAdaptiveSystem`: ATR-based parameter adjustment
- `OrderBookAnalyzer`: Real-time supply/demand analysis
- `CorrelationRiskManager`: Portfolio correlation clustering

**Phase 3 (Advanced - 1-2 months)**:
- `MLSignalValidator`: Machine learning signal verification  
- `IntradayTimeStrategy`: Korean market time-zone optimization

## Key Implementation Guidelines

### Market Scoring Algorithm
```python
def calculate_market_score():
    # Index position (40 points): Above MA5/MA20
    # MA slope (30 points): 20-day moving average momentum  
    # Volatility (30 points): Daily change > -2%
    # Threshold: 70+ points for trading eligibility
```

### Support Level Clustering
```python  
def analyze_support_clustering():
    # Identify overlapping support zones within 1% range
    # Weight by: touch count, volume, MA proximity
    # Apply cluster bonus for 3+ overlapping supports
```

### Dynamic Signal Weighting
```python
def calculate_dynamic_weights(market_condition):
    # Strong uptrend: Lower volume weight, higher candle pattern weight
    # Sideways: Higher volume weight, emphasize oversold indicators  
    # Downtrend: Maximum volume weight (60%), tight confirmation
```

## Korean Market Specifications

### Trading Parameters
- **Position Limits**: Max 5 positions, 20% single position, 40% sector exposure
- **Risk Management**: 2% rule for first 100 trades, then Kelly formula transition
- **Time Constraints**: 3-day holding limit, avoid 11:00-13:00 lunch lull
- **Liquidity Filter**: Market cap >500억 KRW minimum

### Market Condition Thresholds
- **Sector Overheating**: >20% 5-day returns with 3x volume = avoid entry
- **Gap Classification**: Major gap ±3%, minor gap ±1% with specific strategies
- **Signal Strength**: 80+ immediate buy, 60-80 split entry, <60 wait

## Testing Requirements

### Multi-Scenario Backtesting
- **Bull Market**: 2020-04 to 2021-06 (weight: 0.2)
- **Bear Market**: 2022-01 to 2022-10 (weight: 0.3) 
- **Sideways**: 2023-01 to 2023-06 (weight: 0.3)
- **High Volatility**: 2020-03 to 2020-04 (weight: 0.2)

### Walk-Forward Analysis
- Training window: 252 days (1 year)
- Testing window: 63 days (3 months)
- Step size: 63 days to prevent over-optimization

## Critical Safety Mechanisms

### Fundamental Event Filtering
- Earnings blackout: D-3 to D+1
- Major disclosure monitoring with risk level classification
- News sentiment analysis with -0.3 threshold

### Emergency Procedures
```python
def emergency_liquidation():
    # Automatic position closure on system failures
    # Health check every minute (API, data feed, positions)
    # Failsafe with exponential backoff retry logic
```

### Cost Integration
```python  
TRADING_COSTS = {
    'commission_buy': 0.00015, 'commission_sell': 0.00015,
    'tax': 0.0023, 'slippage_market': 0.002
}
# All profit calculations must include these costs
```

## Architecture Notes

When implementing this system:

1. **Module Independence**: Each component must function standalone for testing
2. **Fail-Safe Design**: All trading operations require circuit breakers  
3. **Korean Market Tuning**: All parameters specifically calibrated for KOSPI/KOSDAQ
4. **Risk-First Philosophy**: Prioritize capital preservation over profit optimization
5. **Real-time Adaptation**: System must adjust parameters based on current market volatility and regime

The documentation provides extensive implementation details for each module, including specific algorithms, thresholds, and Korean market microstructure considerations in GUIDE.md and MODULE_GUIDE.md.